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


# --------------------------------------------------------------------------- #
# Per-token gaps (issue #17)                                                  #
# --------------------------------------------------------------------------- #


def _stream(evs):
    """evs: list of (obj, t) -> Response with one SSE chunk per event."""
    lines = [("data: " + json.dumps(o) if o != "[DONE]" else "data: [DONE]") for o, _ in evs]
    chunks = [(line + "\n\n").encode() for line in lines]
    times = [t for _, t in evs]
    return Response(status=200, headers={}, body=b"".join(chunks), elapsed_s=times[-1], ok=True,
                    stream_chunks=chunks, stream_chunk_times=times)


def _delta(text):
    return {"choices": [{"index": 0, "delta": {"content": text}}]}


def test_token_gaps_modes_and_burst_splitting():
    from benchmaker.workloads.llm import token_gaps

    # one token per event, well separated: plain per-event gaps
    g = token_gaps([1.0, 1.010, 1.020, 1.030])
    assert g["mode"] == "per_event" and g["itl_ms"] == pytest.approx([10, 10, 10])
    # three events read together 6 ms after the first: the read's gap is split
    g = token_gaps([1.0, 1.006, 1.00605, 1.0061, 1.016, 1.026])
    assert g["mode"] == "per_event" and g["coalesced_events"] == 2
    assert g["itl_ms"] == pytest.approx([2.0, 2.0, 2.0, 10.0, 10.0])
    assert g["itl_chunk_ms"] == pytest.approx([6.0, 0.05, 0.05, 9.9, 10.0])
    # usage says 5 tokens over 3 events: the surplus is spread evenly
    g = token_gaps([1.0, 1.01, 1.02], completion_tokens=5)
    assert g["mode"] == "per_token_estimated" and g["tokens_per_event"] == [2, 2, 1]
    assert g["itl_ms"] == pytest.approx([5.0, 5.0, 10.0])
    # exact per-event counts (sglang): the deltas are the tokens per event
    g = token_gaps([1.0, 1.01, 1.02], [0, 1, 3])
    assert g["mode"] == "per_token" and g["itl_ms"] == pytest.approx([10.0, 10 / 3, 10 / 3, 10 / 3])
    # usage undercounting the events (reasoning tokens excluded, say): keep events
    g = token_gaps([1.0, 1.01, 1.02], completion_tokens=1)
    assert g["mode"] == "per_token" and g["usage_mismatch"] and len(g["itl_ms"]) == 2
    # counts that never advance carry no information: fall back to per-event
    g = token_gaps([1.0, 1.01, 1.02], [0, 0, 0])
    assert g["mode"] == "per_event" and g["itl_ms"] == pytest.approx([10, 10])
    assert token_gaps([1.0])["itl_ms"] == []


async def test_chat_itl_is_per_token_with_usage_and_reads_are_split():
    wt = OpenAIChatWorkloadType(url="http://x", model="m")
    resp = _stream([
        (_delta("a"), 1.000),
        (_delta("bb"), 1.006),       # 2 tokens, same read as the next event
        (_delta("cc"), 1.0061),
        ({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
          "usage": {"prompt_tokens": 10, "completion_tokens": 5}}, 1.0062),
        ("[DONE]", 1.0062),
    ])
    s = await wt.make_sample("hi", await wt.make_request("hi"), resp, 0.0)
    assert s.meta["itl_mode"] == "per_token_estimated" and s.extra["tokens_out"] == 5.0
    # 5 tokens over 3 events -> 2/2/1; the last two events came in one read
    # 6 ms after the first, carrying 3 tokens: 2 ms each (was: 6 ms then 0.1)
    assert s.series[ITL_SERIES] == pytest.approx([2.0] * 3)
    assert s.extra["itl_ms_mean"] == pytest.approx(2.0) and s.extra["itl_ms_p99"] == pytest.approx(2.0)
    assert s.extra["itl_coalesced_events"] == 1.0 and s.extra["itl_tokens_per_event_mean"] == pytest.approx(5 / 3)
    assert s.extra["itl_chunk_ms_mean"] == pytest.approx(3.05, abs=0.01)  # (6 + 0.1) / 2
    assert s.extra["itl_chunk_ms_p99"] == pytest.approx(5.94, abs=0.01)   # interpolated between the two
    assert s.extra["tokens_out"] == 5.0 and s.extra["ttft_s"] == 1.0


async def test_summary_reports_itl_modes_and_measured_window():
    agg = MetricsAggregator()
    agg.add(_sample({"itl_coalesced_events": 2.0, "itl_tokens_per_event_mean": 1.5}, [2.0, 2.0, 2.0],
                    meta={"itl_mode": "per_token"}))
    # warm-up request: excluded from the window and the summary (#5)
    warm = _sample({}, [5.0], meta={"warmup": True})
    warm.start_ts, warm.latency_s = 0.0, 3.0
    agg.add(warm)
    agg.start_time = 0.0
    agg.finalize()
    agg.end_time = 20.0  # ramp-up + drain around the measured request
    s = agg.summary()
    assert s["itl_modes"] == {"modes": {"per_token": 1}, "usage_mismatch_requests": 0,
                              "coalesced_events_mean": 2.0, "tokens_per_event_mean": 1.5}
    # throughput over the measured request's own window (start 0 -> end 1),
    # not the 20 s run — while wall_time_s keeps the whole run
    assert s["measured_window_s"] == pytest.approx(1.0) and s["wall_time_s"] == 20.0
    assert s["throughput_rps"] == pytest.approx(1.0)
    assert s["warmup_excluded"] == 1

    # measured requests that start after warm-up: window = first start .. last end
    agg = MetricsAggregator()
    agg.start_time = 0.0
    a = _sample({}, [1.0])
    a.start_ts, a.latency_s = 10.0, 2.0
    b = _sample({}, [1.0])
    b.start_ts, b.latency_s = 11.0, 4.0
    agg.add(a)
    agg.add(b)
    agg.finalize()
    agg.end_time = 30.0
    s = agg.summary()
    assert s["measured_window_s"] == pytest.approx(5.0) and s["measured_window"] == {"start_s": 10.0, "end_s": 15.0}
    assert s["throughput_rps"] == pytest.approx(2 / 5)
    # nothing measured: the whole run
    agg = MetricsAggregator()
    agg.finalize()
    assert agg.summary()["measured_window_s"] == agg.summary()["wall_time_s"]


def test_render_shows_measured_window_when_it_differs():
    from benchmaker.core.metrics import MetricsAggregator as _A
    import io
    agg = _A()
    a = _sample({}, [1.0])
    a.start_ts, a.latency_s = 0.0, 1.0
    agg.add(a)
    agg.start_time = 0.0
    agg.finalize()
    agg.end_time = 10.0
    buf = io.StringIO()
    agg.render(buf)
    assert "(over the measured 1.00s of 10.00s wall)" in buf.getvalue()
    assert "gaps per token : per_event=1;" in buf.getvalue()  # no itl_mode meta: per-event
