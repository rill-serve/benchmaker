"""Per-cache-class TTFT: cache classification and ``group_by`` breakdowns
(issue #2)."""

from __future__ import annotations

import json

import pytest

from benchmaker.core.metrics import ITL_SERIES, MetricsAggregator
from benchmaker.core.types import Response, Sample
from benchmaker.workloads.llm import CacheClassInfo, OpenAIChatWorkloadType, classify_cache
from benchmaker.workloads.sglang import SGLangGenerateWorkloadType


def _sample(extra: dict, series: list[float] | None = None, meta: dict | None = None) -> Sample:
    s = Sample(start_ts=0.0, latency_s=1.0, status=200, ok=True)
    s.extra.update(extra)
    if series is not None:
        s.series[ITL_SERIES] = series
    if meta:
        s.meta.update(meta)
    return s


def _sse(evs):
    raw = b"".join((e + "\n\n").encode() for e, _ in evs)
    chunks = [(e + "\n\n").encode() for e, _ in evs]
    times = [t for _, t in evs]
    return Response(status=200, headers={}, body=raw, elapsed_s=times[-1], ok=True,
                    stream_chunks=chunks, stream_chunk_times=times)


# --------------------------------------------------------------------------- #
# group_by: per-class summary blocks                                           #
# --------------------------------------------------------------------------- #


def test_group_by_adds_share_and_per_class_summary():
    agg = MetricsAggregator(group_by=["cache_class"])
    for cls, ttft in [("device_hit", 0.01), ("device_hit", 0.02), ("device_hit", 0.012),
                      ("miss", 0.30), ("restore", 0.05)]:
        agg.add(_sample({"ttft_s": ttft}, [5.0, 6.0], meta={"cache_class": cls}))
    agg.finalize()
    s = agg.summary()
    g = s["groups"]["cache_class"]
    assert set(g) == {"device_hit", "miss", "restore"}
    assert g["device_hit"]["share"] == pytest.approx(0.6) and g["miss"]["share"] == pytest.approx(0.2)
    assert g["device_hit"]["workload_metrics"]["ttft_s"]["mean"] == pytest.approx(0.014)
    assert g["miss"]["itl_ms_pooled"]["count"] == 2
    # untagged samples land in a "(none)" bucket instead of disappearing
    agg = MetricsAggregator(group_by=["cache_class"])
    agg.add(_sample({"ttft_s": 1.0}))
    agg.finalize()
    assert "(none)" in agg.summary()["groups"]["cache_class"]


# --------------------------------------------------------------------------- #
# Cache classification on the LLM workload-types                               #
# --------------------------------------------------------------------------- #


def _cached(cached: int, total: int = 100, hint=None, header=None):
    evs = [
        ('data: ' + json.dumps({"choices": [{"index": 0, "delta": {"content": "a"}}]}), 0.1),
        ('data: ' + json.dumps({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}), 0.1),
        ('data: ' + json.dumps({"choices": [], "usage": {
            "prompt_tokens": total, "completion_tokens": 1,
            "prompt_tokens_details": {"cached_tokens": cached, **({"cache_source": hint} if hint else {})}}}), 0.1),
        ('data: [DONE]', 0.1),
    ]
    resp = _sse(evs)
    if header:
        resp.headers.update(header)
    return resp

async def test_cache_classes_from_cached_tokens_and_hint():
    wt = OpenAIChatWorkloadType(url="http://x", model="m")
    cases = [
        (_cached(0), "miss"),
        (_cached(100), "device_hit"),
        (_cached(60), "partial"),
        (_cached(100, hint="host"), "restore"),
        (_cached(100, hint="hierarchical:remote"), "restore"),
        (_cached(60, hint="gpu"), "partial"),
        (_cached(100, header={"X-Cache-Source": "cpu"}), "restore"),
    ]
    for resp, expected in cases:
        s = await wt.make_sample("hi", await wt.make_request("hi"), resp, 0.0)
        assert s.meta["cache_class"] == expected, (expected, s.meta)
    s = await wt.make_sample("hi", await wt.make_request("hi"), _cached(100, hint="host"), 0.0)
    assert s.meta["cache_hint"] == "host" and s.extra["cached_ratio"] == 1.0

async def test_cache_classifier_is_pluggable():
    def by_tier(info: CacheClassInfo) -> str:
        return {"host": "restore", "ssd": "restore"}.get(info.hint or "") or ("device_hit" if info.cached_tokens == info.prompt_tokens else "miss")

    wt = OpenAIChatWorkloadType(url="http://x", model="m", cache_classifier=by_tier)
    s = await wt.make_sample("hi", await wt.make_request("hi"), _cached(50, hint="ssd"), 0.0)
    assert s.meta["cache_class"] == "restore"
    s = await wt.make_sample("hi", await wt.make_request("hi"), _cached(50), 0.0)
    assert s.meta["cache_class"] == "miss"  # classifier wins over the default "partial"
    assert classify_cache(None) is None
    info = CacheClassInfo(prompt_tokens=100, cached_tokens=0)
    assert classify_cache(info) == "miss"

async def test_sglang_meta_info_cached_tokens_classified():
    evs = [
        ('data: ' + json.dumps({"text": "a", "meta_info": {"prompt_tokens": 100, "cached_tokens": 100,
                                                            "completion_tokens": 1, "finish_reason": "stop"}}), 0.1),
    ]
    wt = SGLangGenerateWorkloadType(url="http://x", model="m")
    req = await wt.make_request({"text": "hi"})
    s = await wt.make_sample({"text": "hi"}, req, _sse(evs), 0.0)
    assert s.meta["cache_class"] == "device_hit" and s.extra["cached_ratio"] == 1.0

# --------------------------------------------------------------------------- #
