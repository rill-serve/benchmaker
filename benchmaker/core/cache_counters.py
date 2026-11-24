"""Aggregate host KV restores from versioned Prometheus counter families."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from benchmaker.core.types import Sample

Series = dict[str, float]


@dataclass(frozen=True)
class Counter:
    name: str
    labels: tuple[tuple[str, str], ...] = ()
    # Scheduler-side counters are logical copies on each TP rank.
    rank_zero: tuple[str, ...] = ()
    estimate: str | None = None

    @property
    def selector(self) -> str:
        if not self.labels:
            return self.name
        pairs = ",".join(f'{k}="{v}"' for k, v in self.labels)
        return f"{self.name}{{{pairs}}}"


@dataclass(frozen=True)
class Engine:
    name: str
    tokens: tuple[Counter, ...]
    prompt: tuple[Counter, ...]
    bytes: tuple[Counter, ...] = ()
    blocks: tuple[Counter, ...] = ()


# LMCache v0.3/v0.4 observability.py and v0.5 MP metrics.rst; SGLang metrics_collector.py
# (scheduler/HiCache and earlier tokenizer metrics); vLLM V1 loggers.py,
# offloading/metrics.py and simple_kv_offload/metrics.py.
_ENGINES = (
    Engine("LMCache", (Counter("lmcache:num_hit_tokens_total"),),
           (Counter("lmcache:num_prompt_tokens_total"),)),
    Engine(
        "LMCache MP",
        (Counter("lmcache_mp_lookup_hit_tokens_total", estimate="lookup hits"),),
        (Counter("lmcache_mp_lookup_prompt_tokens_total"),),
        (Counter("lmcache_mp_transfer_phase_bytes_total", (("direction", "h2d"), ("phase", "staging"))),),
    ),
    Engine(
        "SGLang HiCache",
        (Counter("sglang:prefill_effective_tokens_total", (("mode", "host_hit"),), ("tp_rank",)),
         Counter("sglang:cached_tokens_total", (("cache_source", "host"),)),
         Counter("sglang:load_back_tokens_total", (("pool", "kv"),), ("tp_rank",), "load-back"),
         Counter("sglang:load_back_tokens_total", (("pool", ""),), ("tp_rank",), "load-back")),
        (Counter("sglang:prefill_effective_tokens_total", rank_zero=("tp_rank",)),
         Counter("sglang:prompt_tokens_total")),
        (Counter("sglang:load_back_bytes_total"),),
    ),
    Engine(
        "vLLM native offload",
        (Counter("vllm:prompt_tokens_cached_by_source_total", (("source", "host"),)),),
        (Counter("vllm:prompt_tokens_total"),),
        (Counter("vllm:kv_offload_load_bytes_total"),
         Counter("vllm:kv_offload_total_bytes_total", (("transfer_type", "CPU_to_GPU"),))),
        (Counter("vllm:simple_kv_offload_load_blocks_total"),),
    ),
)
CACHE_COUNTER_NAMES = frozenset(
    counter.name
    for engine in _ENGINES
    for counter in (*engine.tokens, *engine.prompt, *engine.bytes, *engine.blocks)
)

_LABEL = re.compile(r'([a-zA-Z_][a-zA-Z_0-9]*)="((?:[^"\\]|\\.)*)"')


def _matching(values: Series, counter: Counter) -> dict[tuple, float]:
    result = {}
    for key, value in values.items():
        name, _, raw_labels = key.partition("{")
        if name != counter.name:
            continue
        labels = dict(_LABEL.findall(raw_labels))
        if any(labels.get(k, "") != v for k, v in counter.labels):
            continue
        if any(labels.get(rank, "0") != "0" for rank in counter.rank_zero):
            continue
        result[tuple(sorted(labels.items()))] = value
    return result


def _delta(counter: Counter, snapshots: list[Series]) -> float | None:
    first = _matching(snapshots[0], counter)
    if not first:
        return None
    prev = first
    for snapshot in snapshots[1:]:
        current = _matching(snapshot, counter)
        if current.keys() != prev.keys() or any(
            not math.isfinite(value) or value < prev[key] for key, value in current.items()
        ):
            return None
        prev = current
    if any(not math.isfinite(value) or value < 0 for value in first.values()):
        return None
    return sum(prev[key] - value for key, value in first.items())


def _first_delta(counters: tuple[Counter, ...], snapshots: list[Series]) -> tuple[float | None, Counter | None]:
    for counter in counters:
        delta = _delta(counter, snapshots)
        if delta is not None:
            return delta, counter
    return None, None


def counter_restores(
    snapshots: list[Series], *, bytes_per_token: float | None = None, block_size_tokens: int | None = None,
) -> dict:
    """Read matching series at both boundaries and reject observed resets.

    Shares use restored prompt tokens divided by total prompt tokens in the
    same scope. Physical transfer bytes can include padding and duplicate loads.
    """
    if len(snapshots) < 2:
        return {"attribution": "unavailable", "reason": "missing boundary scrape"}
    for engine in _ENGINES:
        tokens, token_counter = _first_delta(engine.tokens, snapshots)
        restored_bytes, byte_counter = _first_delta(engine.bytes, snapshots)
        blocks, block_counter = _first_delta(engine.blocks, snapshots)
        counters = {
            k: v.selector for k, v in {
                "restored_tokens": token_counter, "restored_bytes": byte_counter, "restored_blocks": block_counter,
            }.items() if v is not None
        }
        if not counters:
            continue
        byte_source = "counter" if restored_bytes is not None else None
        token_estimate = token_counter.estimate if token_counter is not None else None
        if tokens is None and blocks is not None and block_size_tokens:
            tokens = blocks * block_size_tokens
            counters["restored_tokens"] = counters["restored_blocks"]
            token_estimate = "estimated from blocks"
        if tokens is None and restored_bytes is not None and bytes_per_token:
            tokens = restored_bytes / bytes_per_token
            counters["restored_tokens"] = counters["restored_bytes"]
            token_estimate = "estimated from bytes"
        if restored_bytes is None and tokens is not None and bytes_per_token:
            restored_bytes = tokens * bytes_per_token
            byte_source = "estimated from tokens"
            counters["restored_bytes"] = counters["restored_tokens"]
        prompt, prompt_counter = _first_delta(engine.prompt, snapshots)
        if prompt_counter:
            counters["prompt_tokens"] = prompt_counter.selector
        # A transfer counter can include padding, prefetch, and repeated loads.
        # Keep the raw delta and omit the share when its scope exceeds the input.
        share = tokens / prompt if tokens is not None and prompt and tokens <= prompt else None
        return {
            "engine": engine.name,
            "attribution": "aggregate only",
            "restored_tokens": tokens,
            "prompt_tokens": prompt,
            "restored_token_share": share,
            "restored_token_estimate": token_estimate,
            "restored_bytes": restored_bytes,
            "restored_bytes_source": byte_source,
            "restored_blocks": blocks,
            "counters": counters,
            "bytes_per_token": bytes_per_token,
            "block_size_tokens": block_size_tokens,
        }
    return {
        "attribution": "unavailable",
        "reason": "no stable supported restore counter",
    }


def response_restores(samples: list[Sample]) -> dict | None:
    """Use explicit response telemetry over the requests with full coverage."""
    if not samples or any(
        "restored_tokens" not in sample.extra or "prompt_tokens" not in sample.extra for sample in samples
    ):
        return None
    restored = sum(sample.extra["restored_tokens"] for sample in samples)
    prompt = sum(sample.extra["prompt_tokens"] for sample in samples)
    return {
        "engine": "response usage",
        "attribution": "per request",
        "restored_tokens": restored,
        "prompt_tokens": prompt,
        "restored_token_share": restored / prompt if 0 <= restored <= prompt and prompt else None,
        "restored_bytes": None,
        "counters": {"restored_tokens": "usage.restored_tokens", "prompt_tokens": "usage.prompt_tokens"},
    }
