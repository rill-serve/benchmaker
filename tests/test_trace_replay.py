"""Replay of published serving traces (issue #6): Mooncake block-hash structure
and the Azure LLM inference trace, with real arrival times."""

from __future__ import annotations

import asyncio
import gzip
import json
import socket

import pytest
import pytest_asyncio
from aiohttp import web

from benchmaker.config import build_config
from benchmaker.core.load import TraceArrivalsLoad
from benchmaker.core.runner import BenchRunner
from benchmaker.workloads.traces import (
    TraceReplayWorkload, AzureTraceWorkload, MooncakeTraceWorkload,
    detect_source, load_azure_trace, load_mooncake_trace, load_trace, trace_replay_from_spec,
)


def _write(path, text: str) -> str:
    if str(path).endswith(".gz"):
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            fh.write(text)
    else:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
    return str(path)


# Real Mooncake field names: timestamp (ms), input_length, output_length, hash_ids.
MOONCAKE_ROWS = [
    {"timestamp": 500, "input_length": 1024, "output_length": 64, "hash_ids": [11, 12]},
    {"timestamp": 120, "input_length": 700, "output_length": 10, "hash_ids": [11, 12]},
    {"timestamp": 0, "input_length": 300, "output_length": 1, "hash_ids": [11]},
    # out-of-order on purpose; input larger than its blocks (tail padded)
    {"timestamp": 900, "input_length": 2000, "output_length": 5, "hash_ids": {"hash_prefix": [11], "hash_ids": [13]}},
    {"timestamp": 950, "input_length": 0, "output_length": 5, "hash_ids": []},   # dropped
    {"timestamp": "bad", "input_length": 10, "output_length": 5},                 # dropped
]


@pytest.fixture
def mooncake(tmp_path):
    return _write(tmp_path / "mooncake.jsonl", "\n".join(json.dumps(r) for r in MOONCAKE_ROWS))


AZURE_CSV = (
    "TIMESTAMP,ContextTokens,GeneratedTokens\n"
    "2024-07-01 00:00:02.500,120,7\n"
    "2024-07-01 00:00:00.000,310,0\n"
    "2024-07-01 00:00:01.000,,3\n"          # no context -> dropped
    "2024-07-01 00:00:01.250,64,12\n"
)


@pytest.fixture
def azure(tmp_path):
    return _write(tmp_path / "AzureLLMInferenceTrace_conv.csv", AZURE_CSV)


# --------------------------------------------------------------------------- #
# Loaders                                                                      #
# --------------------------------------------------------------------------- #


def test_mooncake_loader(mooncake):
    recs = load_mooncake_trace(mooncake)
    assert [r.t_s for r in recs] == [0.0, 0.12, 0.5, 0.9]           # ms -> s, relative to the first
    assert [(r.input_tokens, r.output_tokens) for r in recs] == [(300, 1), (700, 10), (1024, 64), (2000, 5)]
    assert [r.hash_ids for r in recs] == [(11,), (11, 12), (11, 12), (11, 13)]
    assert all(r.meta == {"block_size": 512} for r in recs)
    # the published unit (ms) is the default; a seconds-based file is explicit
    assert load_mooncake_trace(mooncake, timestamp_unit="s")[-1].t_s == pytest.approx(900.0)
    # "auto": a trace spanning > 100k units is ms, a tiny one is kept as seconds
    assert load_mooncake_trace(mooncake, timestamp_unit="auto")[-1].t_s == pytest.approx(900.0)
    big = _write(mooncake.replace(".jsonl", "-big.jsonl"), "\n".join(
        json.dumps({"timestamp": t, "input_length": 10, "output_length": 1}) for t in (3_600_000, 0, 7_200_000)))
    assert [r.t_s for r in load_mooncake_trace(big, timestamp_unit="auto")] == [0.0, 3600.0, 7200.0]


def test_mooncake_limits_and_gzip(mooncake, tmp_path):
    assert len(load_mooncake_trace(mooncake, max_records=2)) == 2
    assert [r.input_tokens for r in load_mooncake_trace(mooncake, max_input_tokens=1000)] == [300, 700]
    assert [r.output_tokens for r in load_mooncake_trace(mooncake, max_output_tokens=10)] == [1, 10, 5]
    assert [r.t_s for r in load_mooncake_trace(mooncake, max_duration_s=0.5)] == [0.0, 0.12, 0.5]
    gz = _write(tmp_path / "mooncake.jsonl.gz", "\n".join(json.dumps(r) for r in MOONCAKE_ROWS))
    assert load_mooncake_trace(gz) == load_mooncake_trace(mooncake)


def test_azure_loader(azure, tmp_path):
    recs = load_azure_trace(azure)
    assert [r.t_s for r in recs] == [0.0, 1.25, 2.5]
    assert [(r.input_tokens, r.output_tokens) for r in recs] == [(310, 1), (64, 12), (120, 7)]
    assert all(r.hash_ids == () for r in recs)
    gz = _write(tmp_path / "azure.csv.gz", AZURE_CSV)
    assert load_azure_trace(gz) == recs
    # lowercase / alternative column names are accepted
    alt = _write(tmp_path / "alt.csv", "timestamp,prompt_tokens,completion_tokens\n0.0,5,1\n1.0,6,2\n")
    assert [(r.t_s, r.input_tokens, r.output_tokens) for r in load_azure_trace(alt)] == [(0.0, 5, 1), (1.0, 6, 2)]
    bad = _write(tmp_path / "bad.csv", "a,b\n1,2\n")
    with pytest.raises(ValueError):
        load_azure_trace(bad)


def test_detect_and_dispatch(mooncake, azure):
    assert detect_source(mooncake) == "mooncake" and detect_source(azure) == "azure"
    assert load_trace(mooncake) == load_mooncake_trace(mooncake)
    assert load_trace(azure, source="azure") == load_azure_trace(azure)
    assert load_trace(azure) == load_azure_trace(azure)
    with pytest.raises(ValueError):
        load_trace(mooncake, source="nope")


# --------------------------------------------------------------------------- #
# Workload: trace-faithful prefix structure                                    #
# --------------------------------------------------------------------------- #


def test_prefix_sharing_follows_hash_ids(mooncake):
    wl = MooncakeTraceWorkload(mooncake)
    items = [asyncio.run(wl.next_item()) for _ in range(4)]
    prompts = [it["messages"][0]["content"] for it in items]
    # record 0: [11] (300 tokens) ; record 1: [11, 12] (700) -> shares block 11
    # and adds block 12 (188 tokens of it); record 2: [11, 12] (1024) -> same
    # blocks, both full; record 3: [11, 13] -> shares block 11 only.
    assert prompts[1].startswith(prompts[0])            # 11 is a full block in both
    assert prompts[2].startswith(prompts[1])            # 11 + (partial 12) is a prefix of 11 + 12
    assert not prompts[3].startswith(prompts[1])        # 13 differs from 12
    common = _common_prefix(prompts[3], prompts[0])
    assert common == prompts[0]                         # record 3 shares exactly block 11
    # char-mode sizing is approximate but in the right ballpark
    assert abs(len(prompts[0]) - 300 * 4) <= 40 and abs(len(prompts[2]) - 1024 * 4) <= 120
    # the 2000-token record: 2 blocks + a request-unique tail
    assert len(prompts[3]) > len(prompts[2])
    assert [it["max_tokens"] for it in items] == [1, 10, 64, 5]
    m = items[0]["meta"]
    assert m["trace_index"] == 0 and m["trace_t_s"] == 0.0 and m["trace_input_tokens"] == 300
    assert m["trace_blocks"] == 1 and m["prompt_tokens_hint"] == 300
    assert wl.arrival_times == [0.0, 0.12, 0.5, 0.9] and wl.item_count() == 4
    with pytest.raises(StopAsyncIteration):
        asyncio.run(wl.next_item())
    assert wl.name.startswith("mooncake:")
    h = wl.content_hash()
    assert h == MooncakeTraceWorkload(mooncake).content_hash() and h != MooncakeTraceWorkload(mooncake, block_size=256).content_hash()


def _common_prefix(a: str, b: str) -> str:
    i = 0
    while i < min(len(a), len(b)) and a[i] == b[i]:
        i += 1
    return a[:i]


def test_records_without_hashes_are_independent(azure):
    wl = AzureTraceWorkload(azure, match_output_tokens=False, max_tokens=16)
    items = [asyncio.run(wl.next_item()) for _ in range(3)]
    prompts = [it["messages"][0]["content"] for it in items]
    assert len(set(prompts)) == 3 and not prompts[1].startswith(prompts[0])
    assert all(it["max_tokens"] == 16 for it in items)
    assert [it["meta"]["trace_output_tokens"] for it in items] == [1, 12, 7]
    assert [it["meta"]["trace_blocks"] for it in items] == [0, 0, 0]
    # loop restarts the trace
    looped = TraceReplayWorkload(load_azure_trace(azure), loop=True)
    assert looped.item_count() is None
    assert [asyncio.run(looped.next_item())["meta"]["trace_index"] for _ in range(4)] == [0, 1, 2, 0]


def test_from_spec_and_config(tmp_path, mooncake, azure):
    wl = trace_replay_from_spec({"path": mooncake, "max_records": 2, "max_input_tokens": 1000, "chars_per_token": 4.0})
    assert wl.item_count() == 2
    with pytest.raises(ValueError):
        trace_replay_from_spec({"source": "mooncake"})
    with pytest.raises(ValueError):
        TraceReplayWorkload([])
    cfg = {
        "workload_type": {"type": "openai", "url": "http://x", "model": "m", "stream": False},
        "workload": {"type": "mooncake-trace", "path": mooncake, "max_records": 3},
        "load": {"type": "trace", "speed": 10},
    }
    bc = build_config(cfg, dotenv_path=None)
    assert isinstance(bc.workload, TraceReplayWorkload) and bc.workload.item_count() == 3
    assert isinstance(bc.load, TraceArrivalsLoad) and bc.load._speed == 10
    cfg2 = {
        "workload_type": {"type": "openai", "url": "http://x", "model": "m"},
        "workload": {"type": "azure", "path": azure},
        "load": {"type": "trace-arrivals", "max_requests": 2},
    }
    bc2 = build_config(cfg2, dotenv_path=None)
    assert isinstance(bc2.load, TraceArrivalsLoad) and bc2.load._max == 2
    with pytest.raises(ValueError):
        build_config({**cfg, "workload": {"type": "static", "items": ["x"]}, "load": {"type": "trace"}}, dotenv_path=None)


# --------------------------------------------------------------------------- #
# End-to-end: arrival times honoured, trace meta on samples                    #
# --------------------------------------------------------------------------- #


async def _chat(request: web.Request) -> web.Response:
    body = await request.json()
    request.app["prompts"].append(body["messages"][0]["content"])
    return web.json_response({
        "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": len(body["messages"][0]["content"]) // 4, "completion_tokens": 1},
    })


@pytest_asyncio.fixture
async def echo_server():
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    app = web.Application()
    app["prompts"] = []
    app.router.add_post("/v1/chat/completions", _chat)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", port)
    await site.start()
    try:
        yield f"http://127.0.0.1:{port}", app
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_trace_replay_run_keeps_arrivals_and_order(echo_server, mooncake):
    base, app = echo_server
    cfg = {
        "workload_type": {"type": "openai", "url": f"{base}/v1/chat/completions", "model": "m", "stream": False},
        "workload": {"type": "trace-replay", "path": mooncake},
        "load": {"type": "trace"},
    }
    runner = BenchRunner(build_config(cfg, dotenv_path=None))
    result = await runner.run()
    samples = sorted(result.samples, key=lambda s: s.start_ts)
    assert len(samples) == 4 and all(s.ok for s in samples)
    # ticket i is record i: the trace order (by arrival) is preserved end to end
    assert [s.meta["trace_index"] for s in samples] == [0, 1, 2, 3]
    t0 = samples[0].start_ts
    gaps = [s.start_ts - t0 for s in samples]
    assert all(abs(g - e) < 0.08 for g, e in zip(gaps, [0.0, 0.12, 0.5, 0.9])), gaps
    # the server saw the shared prefix structure
    seen = app["prompts"]
    assert seen[1].startswith(seen[0]) and seen[2].startswith(seen[1])
    assert samples[0].extra["prompt_tokens_hint"] == 300 and samples[0].extra["prompt_tokens"] > 0
    assert result.summary["workload_metrics"]["prompt_tokens_hint"]["max"] == 2000
    assert result.summary["workload_metrics"]["trace_input_tokens"]["p50"] > 0
