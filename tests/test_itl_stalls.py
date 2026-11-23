"""Token-level ITL pooling and decode-stall metrics (issue #1)."""

from __future__ import annotations

import json

import pytest

from benchmaker.core.metrics import (
    ITL_SERIES, MetricsAggregator, log_histogram, pooled_series_summary, stall_summary,
)
from benchmaker.core.types import Response, Sample
from benchmaker.workloads.llm import OpenAIChatWorkloadType


def _sample(extra: dict, series: list[float] | None = None, meta: dict | None = None) -> Sample:
    s = Sample(start_ts=0.0, latency_s=1.0, status=200, ok=True)
    s.extra.update(extra)
    if series is not None:
        s.series[ITL_SERIES] = series
    if meta:
        s.meta.update(meta)
    return s


# --------------------------------------------------------------------------- #
# Pooled ITL percentiles (#1)                                                  #
# --------------------------------------------------------------------------- #


def test_pooled_percentiles_include_p995_and_p999():
    # 997 gaps of 10ms and 3 gaps of 100ms (0.3%): the tail sits between the
    # p99.5 and p99.9 ranks, so only those percentiles (and max) see it.
    gaps = [10.0] * 997 + [100.0] * 3
    s = pooled_series_summary(gaps)
    assert s["count"] == 1000
    assert s["p99"] == 10.0 and s["p995"] == 10.0 and s["p999"] == 100.0
    assert s["max"] == 100.0 and s["mean"] == pytest.approx(10.27)


def test_summary_pools_gaps_across_requests_not_of_request_p99s():
    agg = MetricsAggregator()
    # Request A: 98 fast gaps + 2 slow ones -> per-request p99 = 100
    agg.add(_sample({"itl_ms_p99": 100.0}, [5.0] * 98 + [100.0] * 2))
    # Requests B..Z: all fast -> per-request p99 = 5
    for _ in range(25):
        agg.add(_sample({"itl_ms_p99": 5.0}, [5.0] * 100))
    agg.finalize()
    s = agg.summary()
    pooled = s["itl_ms_pooled"]
    assert pooled["count"] == 100 + 25 * 100
    # "percentile of per-request p99s" says p99 = 100 (1 of 26 requests has a
    # slow tail) while the token-level pool says p99 = 5, max = 100: the two
    # answer different questions and both are kept, under distinct keys.
    assert pooled["p99"] == 5.0 and pooled["max"] == 100.0
    assert s["workload_metrics"]["itl_ms_p99"]["max"] == 100.0  # old key kept


# --------------------------------------------------------------------------- #
# Stall metrics (#1)                                                           #
# --------------------------------------------------------------------------- #


def test_stall_summary_counts_duration_and_excess():
    per_request = [[10.0, 30.0, 10.0, 10.0], [10.0, 10.0], [50.0, 20.0, 20.0]]
    s = stall_summary(per_request, threshold_ms=12.0)
    assert s["threshold_ms"] == 12.0
    assert s["count"] == 4 and s["requests"] == 3
    assert s["total_time_ms"] == pytest.approx(30 + 50 + 20 + 20)
    assert s["excess_ms"] == pytest.approx((30 - 12) + (50 - 12) + 2 * (20 - 12))
    assert s["per_request"]["mean"] == pytest.approx(4 / 3) and s["per_request"]["max"] == 3.0
    assert s["stall_ms"]["mean"] == pytest.approx(30.0)
    hist = s["histogram"]
    assert sum(b["count"] for b in hist) == 9 and hist[0]["lo_ms"] == 8.0  # empty leading bins dropped
    # the two 50ms gaps land in the [32, 64) bin
    bin_50 = next(b for b in hist if b["lo_ms"] <= 50.0 < b["hi_ms"])
    assert bin_50["count"] == 1 and bin_50["lo_ms"] == 32.0


def test_shorter_stalls_can_raise_p99_but_lower_p995_and_excess():
    # The motivating observation from the issue: shrinking every stall moves
    # the shorter stalls up to p99's rank (p99 up) while the tail beyond falls.
    before = [8.0] * 980 + [50.0] * 20
    after = [8.0] * 980 + [20.0] * 20
    b, a = pooled_series_summary(before), pooled_series_summary(after)
    assert a["p99"] >= b["p99"] or (a["p995"] < b["p995"] and a["max"] < b["max"])
    assert a["p995"] == 20.0 and b["p995"] == 50.0
    sb, sa = stall_summary([before], 12.0), stall_summary([after], 12.0)
    assert sa["count"] == sb["count"] and sa["excess_ms"] < sb["excess_ms"]


def test_log_histogram_edges():
    h = log_histogram([0.5, 1.5, 3.0, 10_000.0], (1, 2, 4, 8))
    assert [(b["lo_ms"], b["hi_ms"]) for b in h][0] == (0.0, 1.0)
    assert h[-1]["hi_ms"] is None and h[-1]["count"] == 1  # 10s in the open bin
    assert [b["count"] for b in h] == [1, 1, 1, 0, 1]


def test_summary_stall_block_and_disable():
    agg = MetricsAggregator()
    agg.add(_sample({}, [10.0, 40.0, 10.0]))
    agg.finalize()
    assert agg.summary()["stalls"]["count"] == 1
    agg = MetricsAggregator(stall_ms=None)
    agg.add(_sample({}, [10.0, 40.0, 10.0]))
    agg.finalize()
    assert "stalls" not in agg.summary() and "itl_ms_pooled" in agg.summary()
    agg = MetricsAggregator(stall_ms=100.0)
    agg.add(_sample({}, [10.0, 40.0, 10.0]))
    agg.finalize()
    assert agg.summary()["stalls"]["count"] == 0 and agg.summary()["stalls"]["threshold_ms"] == 100.0


# --------------------------------------------------------------------------- #
# Bundle / rendering                                                           #
# --------------------------------------------------------------------------- #


def test_dump_series_writes_gaps_to_samples(monkeypatch, tmp_path):
    from benchmaker.io.bundle import write_bundle
    agg = MetricsAggregator(dump_series=True)
    agg.add(_sample({"ttft_s": 0.1}, [1.0, 2.0, 3.0]))
    agg.finalize()
    d = write_bundle(str(tmp_path), agg, workload_type_name="openai-chat", workload_name="static", run_id="r")
    rows = [json.loads(line) for line in open(f"{d}/samples.jsonl")]
    assert rows[0]["series"]["itl_ms"] == [1.0, 2.0, 3.0]
    agg = MetricsAggregator(dump_series=False)
    agg.add(_sample({"ttft_s": 0.1}, [1.0, 2.0, 3.0]))
    agg.finalize()
    d = write_bundle(str(tmp_path), agg, workload_type_name="openai-chat", workload_name="static", run_id="r2")
    assert "series" not in [json.loads(line) for line in open(f"{d}/samples.jsonl")][0]


def test_render_mentions_stalls_and_tail():
    agg = MetricsAggregator()
    agg.add(_sample({"ttft_s": 0.1}, [10.0, 40.0, 10.0]))
    agg.add(_sample({"ttft_s": 0.1}, [10.0, 400.0, 10.0, 12.0]))
    agg.finalize()
    import io
    buf = io.StringIO()
    agg.render(buf)
    text = buf.getvalue()
    assert "p99.5" in text and "stall" in text.lower()


# --------------------------------------------------------------------------- #
# LLM workload-type: raw per-token gaps                                       #
# --------------------------------------------------------------------------- #


def _sse(evs):
    raw = b"".join((e + "\n\n").encode() for e, _ in evs)
    chunks = [(e + "\n\n").encode() for e, _ in evs]
    times = [t for _, t in evs]
    return Response(status=200, headers={}, body=raw, elapsed_s=times[-1], ok=True,
                    stream_chunks=chunks, stream_chunk_times=times)


async def test_llm_sample_carries_raw_gaps():
    evs = [
        ('data: ' + json.dumps({"choices": [{"index": 0, "delta": {"content": "a"}}]}), 0.100),
        ('data: ' + json.dumps({"choices": [{"index": 0, "delta": {"content": "b"}}]}), 0.110),
        ('data: ' + json.dumps({"choices": [{"index": 0, "delta": {"content": "c"}}]}), 0.150),
        ('data: ' + json.dumps({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}), 0.151),
        ('data: ' + json.dumps({"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 3,
                                                         "prompt_tokens_details": {"cached_tokens": 0}}}), 0.152),
        ('data: [DONE]', 0.153),
    ]
    wt = OpenAIChatWorkloadType(url="http://x", model="m")
    sample = await wt.make_sample("hi", await wt.make_request("hi"), _sse(evs), 0.0)
    assert sample.series[ITL_SERIES] == [pytest.approx(10.0), pytest.approx(40.0)]
    assert sample.extra["ttft_s"] == pytest.approx(0.100)
    # per-request stats are unchanged; the pooled ones come from the series
    assert sample.extra["itl_ms_mean"] == pytest.approx(25.0)
