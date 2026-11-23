"""Replay published serving traces with their real arrival times (issue #6).

Supported sources (verified against the public files):

* **Mooncake** (FAST '25, `kvcache-ai/Mooncake`; JSONL or the parquet mirror):
  ``timestamp`` (ms, relative), ``input_length``, ``output_length`` and
  ``hash_ids`` — chained hashes of 512-token blocks mapped to global ids. Two
  requests sharing hash id ``h`` at position ``i`` share *all* blocks up to
  and including ``i``. Replay renders every id deterministically as its own
  block of synthetic text, so the served prefix-cache sees exactly the sharing
  structure of the production trace (system prompt + shared history + unique
  tail), not a cache hit rate "somewhere in between".
* **Azure LLM inference trace** (`Azure/AzurePublicDataset`, ``AzureLLMInferenceTrace_{conv,code}.csv``):
  ``TIMESTAMP`` (wall clock), ``ContextTokens``, ``GeneratedTokens``. No
  prefix information: each request becomes an independent prompt of its
  recorded size.

Token-exact rendering needs a tokenizer (``transformers``); without one the
char-mode sizer is used (``chars_per_token``, approximate).
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import logging
import os
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable, Optional

from benchmaker.workloads import _tokens
from benchmaker.workloads.datasets import Workload

_logger = logging.getLogger(__name__)

MOONCAKE_BLOCK_SIZE = 512  # tokens per hash block in the published trace


# --------------------------------------------------------------------------- #
# Records                                                                      #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TraceRecord:
    """One request of a serving trace."""

    t_s: float                              # relative arrival time (seconds)
    input_tokens: int
    output_tokens: int
    hash_ids: tuple = ()                    # chained block hashes, empty if unknown
    meta: dict = field(default_factory=dict)  # source-specific extras

    def to_dict(self) -> dict:
        return {
            "t_s": self.t_s, "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens, "hash_ids": list(self.hash_ids),
            "meta": self.meta,
        }


# --------------------------------------------------------------------------- #
# Loaders                                                                      #
# --------------------------------------------------------------------------- #


def _open_text(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    lower = path.lower()
    if lower.endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8")
    if lower.endswith(".zip"):
        zf = zipfile.ZipFile(path)
        names = [n for n in zf.namelist() if n.lower().endswith(".jsonl")]
        if not names:
            raise ValueError(f"{path}: no .jsonl member in zip")
        return io.TextIOWrapper(zf.open(names[0]), encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def _first_int(d: dict, keys: Iterable[str]) -> Optional[int]:
    for k in keys:
        v = d.get(k)
        if v is None or v == "":
            continue
        try:
            return int(v)
        except (TypeError, ValueError):
            continue
    return None


def _as_hash_ids(v: Any) -> tuple:
    """Accept ``[ids]`` or ``{"hash_prefix": [...], "hash_ids": [...]}``."""
    if v is None:
        return ()
    if isinstance(v, dict):
        prefix = v.get("hash_prefix") or v.get("hash_prefix_ids") or []
        rest = v.get("hash_ids") or []
        return tuple(int(x) for x in list(prefix) + list(rest))
    if isinstance(v, (list, tuple)):
        return tuple(int(x) for x in v)
    return ()


def _detect_ts_unit(values: list[float]) -> str:
    """``ms`` if the trace spans more than 100k units (a >27h trace in seconds
    is implausible; a 1h trace in ms is 3.6M)."""
    if values and max(values) > 100_000:
        return "ms"
    return "s"


def _limit(records: list[TraceRecord], *, max_records: Optional[int],
           max_duration_s: Optional[float], max_input_tokens: Optional[int],
           max_output_tokens: Optional[int]) -> list[TraceRecord]:
    out = []
    for r in records:
        if max_duration_s is not None and r.t_s > max_duration_s:
            break
        if max_input_tokens is not None and r.input_tokens > max_input_tokens:
            continue
        if max_output_tokens is not None and r.output_tokens > max_output_tokens:
            continue
        out.append(r)
        if max_records is not None and len(out) >= max_records:
            break
    return out


def load_mooncake_trace(path: str, *, timestamp_unit: str = "ms",
                        block_size: int = MOONCAKE_BLOCK_SIZE,
                        max_records: Optional[int] = None,
                        max_duration_s: Optional[float] = None,
                        max_input_tokens: Optional[int] = None,
                        max_output_tokens: Optional[int] = None) -> list[TraceRecord]:
    """Load the Mooncake trace (JSONL / JSONL.gz / parquet).

    Timestamps are relative arrival times in **milliseconds** in the published
    data (the default ``timestamp_unit``). ``"s"`` is for a file already in
    seconds; ``"auto"`` guesses ``"ms"`` when the trace spans more than 100k
    units. Records are sorted by arrival time.
    """
    lower = path.lower()
    rows: list[dict] = []
    if lower.endswith(".parquet"):
        try:
            import pyarrow.parquet as pq  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise ImportError("reading .parquet traces needs pyarrow (pip install pyarrow)") from e
        rows = pq.read_table(path).to_pylist()
    else:
        with _open_text(path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))

    raw_ts: list[float] = []
    parsed: list[Optional[tuple]] = []
    for row in rows:
        if not isinstance(row, dict):
            parsed.append(None)
            raw_ts.append(0.0)
            continue
        ts = row.get("timestamp")
        try:
            ts = float(ts)
        except (TypeError, ValueError):
            parsed.append(None)
            raw_ts.append(0.0)
            continue
        inp = _first_int(row, ("input_length", "input_tokens", "prompt_tokens"))
        out = _first_int(row, ("output_length", "output_tokens", "generated_tokens",
                               "completion_tokens", "max_new_tokens"))
        if inp is None:
            parsed.append(None)
            raw_ts.append(0.0)
            continue
        raw_ts.append(ts)
        parsed.append((ts, inp, out if out is not None else 0, _as_hash_ids(row.get("hash_ids"))))

    if timestamp_unit == "auto":
        timestamp_unit = _detect_ts_unit(raw_ts)
    scale = 0.001 if timestamp_unit == "ms" else 1.0
    t0 = min(raw_ts) if raw_ts else 0.0

    records: list[TraceRecord] = []
    for item in parsed:
        if item is None:
            continue
        ts, inp, out, hashes = item
        if inp <= 0:
            continue
        if hashes and inp > len(hashes) * block_size:
            _logger.debug("mooncake: input_length %d exceeds %d blocks x %d; tail is padded",
                          inp, len(hashes), block_size)
        records.append(TraceRecord(
            t_s=(ts - t0) * scale, input_tokens=inp, output_tokens=max(out, 1),
            hash_ids=hashes, meta={"block_size": block_size},
        ))
    records.sort(key=lambda r: r.t_s)
    records = _limit(records, max_records=max_records, max_duration_s=max_duration_s,
                     max_input_tokens=max_input_tokens, max_output_tokens=max_output_tokens)
    return records


_AZURE_TS_FORMATS = ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f",
                     "%Y-%m-%dT%H:%M:%S")


def _parse_wallclock(v: str) -> Optional[float]:
    v = v.strip()
    for fmt in _AZURE_TS_FORMATS:
        try:
            return datetime.strptime(v, fmt).timestamp()
        except ValueError:
            continue
    try:  # already relative seconds?
        return float(v)
    except ValueError:
        return None


def load_azure_trace(path: str, *, max_records: Optional[int] = None,
                     max_duration_s: Optional[float] = None,
                     max_input_tokens: Optional[int] = None,
                     max_output_tokens: Optional[int] = None) -> list[TraceRecord]:
    """Load ``AzureLLMInferenceTrace_*.csv`` (optionally gzipped).

    ``TIMESTAMP`` is wall clock; it is converted to seconds relative to the
    first request. Column names are matched case-insensitively
    (``ContextTokens`` / ``GeneratedTokens``).
    """
    if path.lower().endswith(".parquet"):
        import pyarrow.parquet as pq  # type: ignore
        rows = pq.read_table(path).to_pylist()
    else:
        with _open_text(path) as fh:
            rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(fh)]

    parsed: list[tuple[float, int, int]] = []
    for row in rows:
        lower = {str(k).lower(): v for k, v in row.items()}
        ts = _parse_wallclock(str(row.get("TIMESTAMP") or lower.get("timestamp") or ""))
        inp = _first_int(lower, ("contexttokens", "prompt_tokens", "input_tokens"))
        out = _first_int(lower, ("generatedtokens", "completion_tokens", "output_tokens"))
        if ts is None or inp is None or inp <= 0:
            continue
        parsed.append((ts, inp, out if out is not None else 0))
    if not parsed:
        raise ValueError(f"{path}: no usable rows (expected TIMESTAMP, ContextTokens, GeneratedTokens)")
    t0 = min(t for t, _, _ in parsed)
    records = [
        TraceRecord(t_s=t - t0, input_tokens=inp, output_tokens=max(out, 1))
        for t, inp, out in parsed
    ]
    records.sort(key=lambda r: r.t_s)
    return _limit(records, max_records=max_records, max_duration_s=max_duration_s,
                  max_input_tokens=max_input_tokens, max_output_tokens=max_output_tokens)


def detect_source(path: str) -> str:
    """``mooncake`` | ``azure`` from extension/header."""
    lower = path.lower()
    if lower.endswith(".parquet"):
        import pyarrow.parquet as pq  # type: ignore
        names = set(pq.read_schema(path).names)
        return "mooncake" if "hash_ids" in names else "azure"
    with _open_text(path) as fh:
        head = fh.read(4096)
    first = head.split("\n", 1)[0]
    if "TIMESTAMP" in first and ("ContextTokens" in first or "GeneratedTokens" in first):
        return "azure"
    return "mooncake"


def load_trace(path: str, source: str = "auto", **kwargs) -> list[TraceRecord]:
    """Load by ``source`` (``mooncake`` | ``azure`` | ``auto``)."""
    if source in ("auto", None):
        source = detect_source(path)
    if source in ("mooncake", "mooncake-trace"):
        return load_mooncake_trace(path, **kwargs)
    if source in ("azure", "azure-trace", "azure-llm"):
        kwargs.pop("timestamp_unit", None)
        kwargs.pop("block_size", None)
        return load_azure_trace(path, **kwargs)
    raise ValueError(f"unknown trace source {source!r} (expected mooncake | azure | auto)")


# --------------------------------------------------------------------------- #
# Workload                                                                     #
# --------------------------------------------------------------------------- #


class TraceReplayWorkload(Workload):
    """Turn trace records into chat requests with trace-faithful prompt structure.

    Args:
        records: :class:`TraceRecord` list (sorted by arrival).
        tokenizer: HF tokenizer id for token-exact prompts; ``None`` uses
            char-mode ``chars_per_token`` (default 4.0, approximate).
        block_size: tokens per hash block (512 for Mooncake).
        match_output_tokens: send the trace's output length as ``max_tokens``
            (generation may stop earlier at EOS; pair with ``ignore_eos: true``
            in the workload-type for exact lengths).
        max_tokens: cap when ``match_output_tokens`` is false (or as a cap).
        loop: restart from the first record when exhausted (default False).

    Item shape: ``{"messages": [...], "max_tokens": n, "meta": {...}}`` with
    ``meta.prompt_tokens_hint`` and the trace fields, so the OpenAI
    workload-type records them as metrics.
    """

    def __init__(self, records: list[TraceRecord], *, tokenizer: Optional[str] = None,
                 chars_per_token: float = 4.0, block_size: int = MOONCAKE_BLOCK_SIZE,
                 match_output_tokens: bool = True, max_tokens: Optional[int] = None,
                 loop: bool = False, name: str = "trace-replay"):
        if not records:
            raise ValueError("trace has no records")
        self.name = name
        self._records = list(records)
        self._sizer = _tokens.build_sizer(tokenizer, chars_per_token)
        self._block_size = int(block_size)
        self._match_output = bool(match_output_tokens)
        self._max_tokens = max_tokens
        self._loop = loop
        self._i = 0
        self._cache: dict[tuple, str] = {}

    # -- provenance -------------------------------------------------------
    @property
    def records(self) -> list[TraceRecord]:
        return self._records

    @property
    def arrival_times(self) -> list[float]:
        """Relative arrival times (s), one per record — feed ``TraceArrivalsLoad``."""
        return [r.t_s for r in self._records]

    def item_count(self) -> Optional[int]:
        return None if self._loop else len(self._records)

    def content_hash(self) -> Optional[str]:
        import hashlib
        payload = {
            "records": [r.to_dict() for r in self._records],
            "exact": getattr(self._sizer, "exact", False),
            "block_size": self._block_size,
            "match_output_tokens": self._match_output,
            "max_tokens": self._max_tokens,
        }
        return "sha256:" + hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()

    # -- rendering --------------------------------------------------------
    def _block_text(self, block_id: Any, n_tokens: int) -> str:
        key = ("blk", str(block_id), n_tokens)
        text = self._cache.get(key)
        if text is None:
            text = self._sizer.block(("mooncake", str(block_id)), n_tokens)
            self._cache[key] = text
        return text

    def _render_prefix(self, record: TraceRecord, index: int) -> str:
        """Prompt whose blocks follow ``record.hash_ids`` (issue #6).

        Block ``i`` (a full block) is ``block_size`` tokens of text derived
        only from ``hash_ids[i]``; the last block gets the remainder
        ``input_tokens - i*block_size`` (the trace's partial block, unique by
        construction). If the hash list is shorter than the input, the tail is
        padded with request-unique filler. Requests without hash ids get one
        request-unique block of ``input_tokens`` tokens.
        """
        remaining = record.input_tokens
        if not record.hash_ids:
            return self._sizer.block((self.name, index), remaining)
        parts: list[str] = []
        for i, h in enumerate(record.hash_ids):
            if remaining <= 0:
                break
            n = min(self._block_size, remaining)
            parts.append(self._block_text(h, n))
            remaining -= n
        if remaining > 0:
            parts.append(self._sizer.block((self.name, "tail", index), remaining))
        return " ".join(parts)

    async def next_item(self) -> Any:
        if self._i >= len(self._records):
            if not self._loop:
                raise StopAsyncIteration
            self._i = 0
        rec = self._records[self._i]
        index = self._i
        self._i += 1
        prompt = self._render_prefix(rec, index)
        out_tokens = rec.output_tokens if self._match_output else (self._max_tokens or 128)
        if self._max_tokens is not None:
            out_tokens = min(out_tokens, self._max_tokens)
        return {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": int(max(out_tokens, 1)),
            "meta": {
                "trace_index": index,
                "trace_t_s": rec.t_s,
                "trace_input_tokens": rec.input_tokens,
                "trace_output_tokens": rec.output_tokens,
                "trace_blocks": len(rec.hash_ids),
                "prompt_tokens_hint": rec.input_tokens,
                **({"trace_meta": rec.meta} if rec.meta else {}),
            },
        }


def trace_replay_from_spec(spec: dict) -> TraceReplayWorkload:
    """Build from a YAML ``workload:`` block (``path`` required)."""
    kwargs = {k: v for k, v in spec.items() if k not in ("type", "path", "source")}
    path = spec.get("path")
    if not path:
        raise ValueError("trace workload needs 'path'")
    source = spec.get("source", "auto")
    load_keys = ("timestamp_unit", "max_records", "max_duration_s", "max_input_tokens",
                 "max_output_tokens", "block_size")
    records = load_trace(path, source=source,
                         **{k: kwargs.pop(k) for k in list(kwargs) if k in load_keys})
    name = kwargs.pop("name", None) or f"{source if source != 'auto' else detect_source(path)}:{os.path.basename(path)}"
    return TraceReplayWorkload(records, name=name, **kwargs)


def MooncakeTraceWorkload(path: str, **kwargs: Any) -> TraceReplayWorkload:  # noqa: N802
    kwargs.pop("source", None)
    return trace_replay_from_spec({"type": "mooncake", "path": path, "source": "mooncake", **kwargs})


def AzureTraceWorkload(path: str, **kwargs: Any) -> TraceReplayWorkload:  # noqa: N802
    kwargs.pop("source", None)
    return trace_replay_from_spec({"type": "azure", "path": path, "source": "azure", **kwargs})
