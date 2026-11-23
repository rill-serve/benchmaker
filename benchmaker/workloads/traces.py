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
char-mode sizer is used (``chars_per_token``, approximate). With
``send: token_ids`` (issue #11) no text is rendered at all: every block is a
seeded list of whitelisted token ids sent as ``prompt: [ids]`` to
``/v1/completions`` (workload-type ``openai-completions``), which keeps blocks
page-aligned and free of tokenizer merges and chat-template shifts.
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
           max_output_tokens: Optional[int], max_seq: Optional[int] = None,
           dropped: Optional[dict[str, int]] = None) -> list[TraceRecord]:
    """Apply the size/duration limits. Rows removed by ``max_input_tokens``,
    ``max_output_tokens`` or ``max_seq`` are counted in ``dropped`` by reason
    (``max_duration`` / ``max_records`` only truncate the stream)."""
    dropped = dropped if dropped is not None else {}

    def _drop(reason: str) -> None:
        dropped[reason] = dropped.get(reason, 0) + 1

    out = []
    for r in records:
        if max_duration_s is not None and r.t_s > max_duration_s:
            break
        if max_input_tokens is not None and r.input_tokens > max_input_tokens:
            _drop("max_input_tokens")
            continue
        if max_output_tokens is not None and r.output_tokens > max_output_tokens:
            _drop("max_output_tokens")
            continue
        if max_seq is not None and r.input_tokens + r.output_tokens > max_seq:
            _drop("max_seq")
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
                        max_output_tokens: Optional[int] = None,
                        max_seq: Optional[int] = None,
                        stats: Optional[dict] = None) -> list[TraceRecord]:
    """Load the Mooncake trace (JSONL / JSONL.gz / parquet).

    Timestamps are relative arrival times in **milliseconds** in the published
    data (the default ``timestamp_unit``). ``"s"`` is for a file already in
    seconds; ``"auto"`` guesses ``"ms"`` when the trace spans more than 100k
    units. Records are sorted by arrival time.

    ``max_seq`` (the server's ``max_model_len``) drops rows whose
    ``input + output`` would not fit; ``stats`` (an optional dict) receives
    ``{"rows", "kept", "dropped": {reason: n}}`` for the run manifest.
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
    dropped: dict[str, int] = {}
    for item in parsed:
        if item is None:
            dropped["malformed"] = dropped.get("malformed", 0) + 1
            continue
        ts, inp, out, hashes = item
        if inp <= 0:
            dropped["zero_input"] = dropped.get("zero_input", 0) + 1
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
                     max_input_tokens=max_input_tokens, max_output_tokens=max_output_tokens,
                     max_seq=max_seq, dropped=dropped)
    if stats is not None:
        stats.update({"rows": len(rows), "kept": len(records), "dropped": dropped})
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
                     max_output_tokens: Optional[int] = None,
                     max_seq: Optional[int] = None,
                     stats: Optional[dict] = None) -> list[TraceRecord]:
    """Load ``AzureLLMInferenceTrace_*.csv`` (optionally gzipped).

    ``TIMESTAMP`` is wall clock; it is converted to seconds relative to the
    first request. Column names are matched case-insensitively
    (``ContextTokens`` / ``GeneratedTokens``). ``max_seq`` / ``stats`` as in
    :func:`load_mooncake_trace`.
    """
    if path.lower().endswith(".parquet"):
        import pyarrow.parquet as pq  # type: ignore
        rows = pq.read_table(path).to_pylist()
    else:
        with _open_text(path) as fh:
            rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(fh)]

    parsed: list[tuple[float, int, int]] = []
    dropped: dict[str, int] = {}
    for row in rows:
        lower = {str(k).lower(): v for k, v in row.items()}
        ts = _parse_wallclock(str(row.get("TIMESTAMP") or lower.get("timestamp") or ""))
        inp = _first_int(lower, ("contexttokens", "prompt_tokens", "input_tokens"))
        out = _first_int(lower, ("generatedtokens", "completion_tokens", "output_tokens"))
        if ts is None or inp is None or inp <= 0:
            dropped["malformed"] = dropped.get("malformed", 0) + 1
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
    records = _limit(records, max_records=max_records, max_duration_s=max_duration_s,
                     max_input_tokens=max_input_tokens, max_output_tokens=max_output_tokens,
                     max_seq=max_seq, dropped=dropped)
    if stats is not None:
        stats.update({"rows": len(rows), "kept": len(records), "dropped": dropped})
    return records


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
    """Turn trace records into requests with trace-faithful prompt structure.

    Args:
        records: :class:`TraceRecord` list (sorted by arrival).
        tokenizer: HF tokenizer id (or an already-loaded tokenizer) for
            token-exact prompts; ``None`` uses char-mode ``chars_per_token``
            (default 4.0, approximate). In ``token_ids`` mode it defines the
            whitelist of sendable ids instead.
        chars_per_token: char-mode approximation (text mode only).
        block_size: tokens per hash block (512 for Mooncake).
        match_output_tokens: send the trace's output length as ``max_tokens``
            (generation may stop earlier at EOS; pair with ``ignore_eos: true``
            in the workload-type for exact lengths).
        max_tokens: cap when ``match_output_tokens`` is false (or as a cap).
        loop: restart from the first record when exhausted (default False).
        send: ``"text"`` (default; chat messages) or ``"token_ids"`` (issue #11):
            every hash block is rendered as exactly ``block_size`` whitelisted
            token ids and sent as ``prompt: [ids]`` — pair it with the
            ``openai-completions`` workload-type. Nothing is tokenized server
            side, so blocks never merge across boundaries and no chat template
            shifts the alignment: equal hash ids give byte-identical,
            page-aligned prefixes.
        namespace: mixed into every block seed so that runs serving several
            models from one trace get disjoint block content (``namespace:
            model`` resolves to the workload-type's model). ``None`` keeps the
            historical seeds (same prompt-set hash as before #11).
        unique_tail: append this many request-unique tokens after the last
            block. Requests with identical hash lists (whole-prompt duplicates
            in the trace) then keep their shared prefix but stop being
            byte-identical prompts. Appended *in addition to* the recorded
            input length; ``meta.prompt_tokens_hint`` reflects the real count.
        vocab_size / exclude_ids / exclude_last: whitelist for ``token_ids``
            mode without a tokenizer (``[0, vocab_size)`` minus the reserved
            ids — state your tokenizer's special range, usually at the end).
        source / path / dropped / max_seq: provenance recorded in the manifest
            (``prompt_set.provenance``): where the records came from and how
            many rows each limit removed.

    Item shape: ``{"messages": [...], "max_tokens": n, "meta": {...}}`` in
    text mode and ``{"prompt": [ids], "max_tokens": n, "meta": {...}}`` in
    token_ids mode, with ``meta.prompt_tokens_hint`` and the trace fields so
    the workload-types record them as metrics.
    """

    def __init__(self, records: list[TraceRecord], *, tokenizer: Any = None,
                 chars_per_token: float = 4.0, block_size: int = MOONCAKE_BLOCK_SIZE,
                 match_output_tokens: bool = True, max_tokens: Optional[int] = None,
                 loop: bool = False, name: str = "trace-replay", send: str = "text",
                 namespace: Optional[str] = None, unique_tail: int = 0,
                 vocab_size: Optional[int] = None, exclude_ids: Any = (),
                 exclude_last: int = 0, source: Optional[str] = None,
                 path: Optional[str] = None, dropped: Optional[dict] = None,
                 max_seq: Optional[int] = None):
        if not records:
            raise ValueError("trace has no records")
        if send not in ("text", "token_ids"):
            raise ValueError(f"send={send!r} (expected 'text' or 'token_ids')")
        if int(unique_tail) < 0:
            raise ValueError("unique_tail must be >= 0")
        self.name = name
        self._records = list(records)
        self.send = send
        self.namespace = namespace or None
        self._unique_tail = int(unique_tail)
        if send == "token_ids":
            if tokenizer is None and vocab_size is None:
                raise ValueError("send: token_ids needs tokenizer=<hf id> or vocab_size=<int>")
            self._ids = (_tokens.TokenIdSampler.from_tokenizer(tokenizer) if tokenizer is not None
                         else _tokens.TokenIdSampler.from_vocab(vocab_size, exclude_ids, exclude_last))
            self._sizer = None
        else:
            if vocab_size is not None or exclude_ids or exclude_last:
                raise ValueError("vocab_size/exclude_ids/exclude_last only apply to send: token_ids")
            self._ids = None
            self._sizer = _tokens.build_sizer(tokenizer, chars_per_token)
        self._tokenizer_name = tokenizer if isinstance(tokenizer, str) else (
            f"object:{type(tokenizer).__name__}" if tokenizer is not None else None)
        self._chars_per_token = chars_per_token
        self._block_size = int(block_size)
        self._match_output = bool(match_output_tokens)
        self._max_tokens = max_tokens
        self._loop = loop
        self._i = 0
        self._cache: dict[tuple, Any] = {}
        # provenance for the manifest
        self._source = source
        self._path = path
        self._dropped = dict(dropped or {})
        self._max_seq = max_seq

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

    def _params(self) -> dict:
        p = {
            "send": self.send, "namespace": self.namespace, "unique_tail": self._unique_tail,
            "block_size": self._block_size, "match_output_tokens": self._match_output,
            "max_tokens": self._max_tokens, "chars_per_token": self._chars_per_token,
            "tokenizer": self._tokenizer_name, "max_seq": self._max_seq,
        }
        if self._ids is not None:
            p["token_whitelist"] = {"size": self._ids.size, "source": self._ids.source}
        return p

    def content_hash(self) -> Optional[str]:
        import hashlib
        payload = {
            "records": [r.to_dict() for r in self._records],
            "exact": getattr(self._sizer, "exact", False),
            "block_size": self._block_size,
            "match_output_tokens": self._match_output,
            "max_tokens": self._max_tokens,
        }
        # Only the #11 knobs that change prompt content join the hash, so a
        # default text-mode config keeps its pre-#11 prompt-set hash.
        if self.send != "text" or self.namespace or self._unique_tail:
            payload.update({k: v for k, v in self._params().items()
                            if k in ("send", "namespace", "unique_tail", "token_whitelist")})
        return "sha256:" + hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()

    def provenance(self) -> dict:
        """What the run manifest records under ``prompt_set.provenance``: the
        replay parameters and the rows dropped by each limit (issue #11)."""
        return {
            "source": self._source, "path": self._path, "records": len(self._records),
            **self._params(), "dropped": dict(self._dropped),
        }

    def set_namespace(self, namespace: Optional[str]) -> None:
        """Called by ``build_config`` to resolve ``namespace: model``; must happen
        before the first item is rendered."""
        if self._i != 0 or self._cache:
            raise RuntimeError("cannot change namespace after rendering started")
        self.namespace = namespace or None

    # -- rendering --------------------------------------------------------
    def _block(self, block_id: Any, n_tokens: int):
        key = ("blk", self.namespace, str(block_id), n_tokens)
        hit = self._cache.get(key)
        if hit is None:
            seed = (("mooncake", self.namespace, str(block_id)) if self.namespace
                    else ("mooncake", str(block_id)))
            hit = self._ids.block(seed, n_tokens) if self._ids is not None else self._sizer.block(seed, n_tokens)
            self._cache[key] = hit
        return hit

    def _request_block(self, index: int, kind: str, n_tokens: int):
        """Request-unique content: ``kind`` is ``""`` (no hashes), ``"tail"``
        (input longer than its blocks) or ``"unique"`` (``unique_tail``)."""
        key = ("req", self.namespace, index, kind, n_tokens)
        hit = self._cache.get(key)
        if hit is None:
            seed = ((self.name, self.namespace, kind, index) if self.namespace
                    else ((self.name, index) if kind == "" else (self.name, kind, index)))
            hit = self._ids.block(seed, n_tokens) if self._ids is not None else self._sizer.block(seed, n_tokens)
            self._cache[key] = hit
        return hit

    def _render_prefix(self, record: TraceRecord, index: int) -> Any:
        """Prompt whose blocks follow ``record.hash_ids`` (issue #6).

        Block ``i`` (a full block) is ``block_size`` tokens derived only from
        ``hash_ids[i]``; the last block gets the remainder
        ``input_tokens - i*block_size`` — in token_ids mode a true prefix of
        the full block. If the hash list is shorter than the input, the tail
        is padded with request-unique content; ``unique_tail`` adds more.
        Requests without hash ids get one request-unique block of
        ``input_tokens`` tokens. Returns ``str`` (text mode) or ``list[int]``
        (token_ids mode).
        """
        remaining = record.input_tokens
        parts: list = []
        if not record.hash_ids:
            parts.append(self._request_block(index, "", remaining))
            remaining = 0
        else:
            for h in record.hash_ids:
                if remaining <= 0:
                    break
                n = min(self._block_size, remaining)
                parts.append(self._block(h, n))
                remaining -= n
        if remaining > 0:
            parts.append(self._request_block(index, "tail", remaining))
        if self._unique_tail > 0:
            parts.append(self._request_block(index, "unique", self._unique_tail))
        if self._ids is not None:
            out: list[int] = []
            for p in parts:
                out.extend(p)
            return out
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
        meta = {
            "trace_index": index,
            "trace_t_s": rec.t_s,
            "trace_input_tokens": rec.input_tokens,
            "trace_output_tokens": rec.output_tokens,
            "trace_blocks": len(rec.hash_ids),
            "prompt_tokens_hint": rec.input_tokens + self._unique_tail,
            "trace_unique_tail": self._unique_tail,
            **({"trace_meta": rec.meta} if rec.meta else {}),
        }
        if self._ids is not None:
            return {"prompt": prompt, "max_tokens": int(max(out_tokens, 1)), "meta": meta}
        return {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": int(max(out_tokens, 1)),
            "meta": meta,
        }


def trace_replay_from_spec(spec: dict) -> TraceReplayWorkload:
    """Build from a YAML ``workload:`` block (``path`` required)."""
    kwargs = {k: v for k, v in spec.items() if k not in ("type", "path", "source")}
    path = spec.get("path")
    if not path:
        raise ValueError("trace workload needs 'path'")
    source = spec.get("source", "auto")
    load_keys = ("timestamp_unit", "max_records", "max_duration_s", "max_input_tokens",
                 "max_output_tokens", "block_size", "max_seq")
    load_kwargs = {k: kwargs.pop(k) for k in list(kwargs) if k in load_keys}
    stats: dict = {}
    records = load_trace(path, source=source, stats=stats, **load_kwargs)
    if source == "auto":
        source = detect_source(path)
    name = kwargs.pop("name", None) or f"{source}:{os.path.basename(path)}"
    return TraceReplayWorkload(
        records, name=name, source=source, path=path, dropped=stats.get("dropped"),
        max_seq=load_kwargs.get("max_seq"), **kwargs)


def MooncakeTraceWorkload(path: str, **kwargs: Any) -> TraceReplayWorkload:  # noqa: N802
    kwargs.pop("source", None)
    return trace_replay_from_spec({"type": "mooncake", "path": path, "source": "mooncake", **kwargs})


def AzureTraceWorkload(path: str, **kwargs: Any) -> TraceReplayWorkload:  # noqa: N802
    kwargs.pop("source", None)
    return trace_replay_from_spec({"type": "azure", "path": path, "source": "azure", **kwargs})
