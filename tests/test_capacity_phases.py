"""Phased runs and the capacity-revisit workload (issue #5)."""

from __future__ import annotations

import asyncio
import json
import socket

import pytest
import pytest_asyncio
from aiohttp import web

from benchmaker.config import build_config
from benchmaker.core.runner import BenchPhase, BenchRunner
from benchmaker.workloads.capacity import CapacityRevisitWorkload
from benchmaker.workloads.datasets import IterableWorkload
from benchmaker.core.metrics import ITL_SERIES, MetricsAggregator
from benchmaker.core.types import Request, Sample
from benchmaker.workloads.llm import OpenAIChatWorkloadType


# --------------------------------------------------------------------------- #
# Workload                                                                     #
# --------------------------------------------------------------------------- #


def test_capacity_items_warm_then_revisits_with_unique_suffix():
    docs = ["The quick brown fox jumps over the lazy dog. " * 50 for _ in range(3)]
    wl = CapacityRevisitWorkload(docs, revisits=2, suffix_tokens=4, max_tokens=7)
    # public ordering: warm pass in doc order, then revisit passes
    got = [asyncio.run(wl.next_item()) for _ in range(9)]
    assert [it["meta"]["visit"] for it in got] == ["warm"] * 3 + ["revisit"] * 6
    assert [it["meta"]["doc_id"] for it in got] == [0, 1, 2, 0, 1, 2, 0, 1, 2]
    assert [it["meta"]["pass_index"] for it in got] == [0, 0, 0, 1, 1, 1, 2, 2, 2]
    assert [it["meta"]["phase"] for it in got[:3]] == ["capacity-warm"] * 3
    assert {it["meta"]["phase"] for it in got[3:]} == {"capacity-revisit"}
    # every visit has a unique suffix so a revisit is not a duplicate prompt
    prompts = [it["messages"][0]["content"] for it in got]
    assert len(set(prompts)) == 9
    assert all(p.startswith(docs[i % 3][:40]) for i, p in enumerate(prompts))
    assert all(it["max_tokens"] == 7 for it in got)
    assert all(it["meta"]["prompt_tokens_hint"] == it["meta"]["doc_tokens"] + 4 for it in got)
    with pytest.raises(StopAsyncIteration):
        asyncio.run(wl.next_item())
    assert wl.item_count() == 9 and wl.n_docs == 3
    assert wl.content_hash().startswith("sha256:")
    # deterministic: same docs -> same hash; different seed/order -> different
    assert wl.content_hash() == CapacityRevisitWorkload(docs, revisits=2, suffix_tokens=4, max_tokens=7).content_hash()
    assert wl.content_hash() != CapacityRevisitWorkload(docs, revisits=2, suffix_tokens=4, max_tokens=7,
                                                        revisit_order="random").content_hash()


def test_capacity_random_revisit_order_is_seed_stable():
    docs = ["d%d " % i * 100 for i in range(8)]
    a = CapacityRevisitWorkload(docs, revisits=3, revisit_order="random", seed=11)
    b = CapacityRevisitWorkload(docs, revisits=3, revisit_order="random", seed=11)
    c = CapacityRevisitWorkload(docs, revisits=3, revisit_order="random", seed=12)
    ids_a = [it["meta"]["doc_id"] for it in a.revisit_items()]
    assert ids_a == [it["meta"]["doc_id"] for it in b.revisit_items()]
    assert ids_a != [it["meta"]["doc_id"] for it in c.revisit_items()]
    # the warm pass stays in id order
    assert [it["meta"]["doc_id"] for it in a.warm_items()] == list(range(8))
    # each revisit pass still covers every document exactly once
    for k in range(1, 4):
        assert sorted(it["meta"]["doc_id"] for it in a.revisit_items() if it["meta"]["pass_index"] == k) == list(range(8))
    with pytest.raises(ValueError):
        CapacityRevisitWorkload(docs, revisit_order="lru")


def test_capacity_stage_views_and_loop():
    docs = ["abc " * 200, "def " * 200]
    wl = CapacityRevisitWorkload(docs, revisits=1, stage="warm")
    assert [it["meta"]["visit"] for it in wl._items] == ["warm", "warm"]
    full = CapacityRevisitWorkload(docs, revisits=1)
    assert len(full.warm_items()) == 2 and len(full.revisit_items()) == 2
    w, r = full.warm_workload(), full.revisit_workload()
    assert isinstance(w, IterableWorkload) and w.item_count() == 2 and r.item_count() == 2
    assert w.name == "capacity-revisit-warm"
    loop = CapacityRevisitWorkload(docs, revisits=0, loop=True)
    assert loop.item_count() is None
    got = [asyncio.run(loop.next_item()) for _ in range(3)]  # 2 docs, restarts
    assert [it["meta"]["doc_id"] for it in got] == [0, 1, 0]
    with pytest.raises(ValueError):
        CapacityRevisitWorkload(stage="hot")
    with pytest.raises(ValueError):
        CapacityRevisitWorkload()


def test_capacity_synthesized_docs_are_distinct_and_sized():
    wl = CapacityRevisitWorkload(n_docs=4, doc_tokens=64, chars_per_token=4.0)
    assert wl.n_docs == 4
    texts = [d.text for d in wl._docs]
    assert len(set(texts)) == 4
    assert all(abs(len(t) - 256) <= 8 for t in texts)
    assert all(d.doc_tokens == 64 for d in wl._docs)


def test_capacity_from_jsonl(tmp_path):
    p = tmp_path / "docs.jsonl"
    p.write_text("\n".join(json.dumps({"text": "x" * (100 * (i + 1))}) for i in range(3)) + "\n\n")
    wl = CapacityRevisitWorkload(path=str(p))
    assert wl.n_docs == 3 and [d.doc_tokens for d in wl._docs] == [25, 50, 75]
    (tmp_path / "empty.jsonl").write_text("\n")
    with pytest.raises(ValueError):
        CapacityRevisitWorkload(path=str(tmp_path / "empty.jsonl"))


# --------------------------------------------------------------------------- #
# Config: phases + lane model override                                         #
# --------------------------------------------------------------------------- #


def test_build_config_phases():
    cfg = {
        "workload_type": {"type": "openai", "url": "http://x", "model": "base", "stream": False},
        "phases": [
            {"name": "hot-a", "duration": "0.1s", "drain": True,
             "lanes": [{"name": "a", "workload": {"type": "static", "items": ["hi"]}, "rate": "10"}]},
            {"name": "both", "duration": 0.2,
             "lanes": [{"name": "a", "workload": {"type": "static", "items": ["hi"]}, "rate": "10"},
                       {"name": "b", "workload": {"type": "static", "items": ["ho"]}, "rate": "5",
                        "model": "other-model", "meta": {"tier": "gold"}}]},
        ],
    }
    bc = build_config(cfg, dotenv_path=None)
    assert [p.name for p in bc.phases] == ["hot-a", "both"]
    assert bc.phases[0].drain is True and bc.phases[1].drain is False
    assert bc.phases[0].duration_s == 0.1 and bc.phases[1].duration_s == 0.2
    assert [lane.name for lane in bc.phases[1].lanes] == ["a", "b"]
    assert bc.phases[1].lanes[1].model == "other-model" and bc.phases[1].lanes[0].model is None
    assert bc.phases[1].lanes[1].meta == {"tier": "gold", "phase": "both"}
    assert bc.load is None and bc.lanes == []
    with pytest.raises(ValueError):
        build_config({**cfg, "load": "10"}, dotenv_path=None)
    with pytest.raises(ValueError):
        build_config({"workload_type": cfg["workload_type"], "phases": [{"lanes": []}]}, dotenv_path=None)
    with pytest.raises(ValueError):
        build_config({"workload_type": cfg["workload_type"],
                      "phases": [{"name": "x", "lanes": [{"name": "a", "workload": {"type": "static", "items": ["x"]}, "rate": "1"}]}]},
                     dotenv_path=None)


def test_phase_post_init_validates():
    with pytest.raises(ValueError):
        BenchPhase(name=" ", lanes=[], duration_s=1.0)


# --------------------------------------------------------------------------- #
# End-to-end phased run against a stub with a simulated prefix cache            #
# --------------------------------------------------------------------------- #


class _Cache:
    def __init__(self, docs: list[str]) -> None:
        self.seen: dict[str, int] = {}
        self.models: list[str] = []
        self.docs = docs


async def _chat(request: web.Request) -> web.Response:
    body = await request.json()
    prompt = body["messages"][0]["content"]
    state: _Cache = request.app["state"]
    state.models.append(body.get("model"))
    # Simulated prefix cache: the shared document body is a hit on revisit
    # (the unique per-visit suffix is the only new part).
    doc = next(d for d in state.docs if prompt.startswith(d))
    cached = state.seen.get(doc, 0)
    state.seen[doc] = len(doc) // 4
    prompt_tokens = max(len(prompt) // 4, 1)
    return web.json_response({
        "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 1,
                  "prompt_tokens_details": {"cached_tokens": min(cached, prompt_tokens)}},
    })


@pytest_asyncio.fixture
async def cache_server():
    docs = ["doc %d " % i + "lorem ipsum dolor sit amet " * 30 for i in range(3)]
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    app = web.Application()
    app["state"] = _Cache(docs)
    app.router.add_post("/v1/chat/completions", _chat)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", port)
    await site.start()
    try:
        yield f"http://127.0.0.1:{port}", app["state"]
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_phased_run_tags_samples_and_summarizes_per_phase(cache_server):
    base, state = cache_server
    docs = state.docs
    cfg = {
        "workload_type": {"type": "openai", "url": f"{base}/v1/chat/completions", "model": "base", "stream": False},
        "warmup_requests": 0,
        "group_by": ["cache_class"],
        "phases": [
            {"name": "warm", "duration": "0.15s", "drain": True,
             "lanes": [{"name": "warm", "rate": "40", "model": "warm-model",
                        "workload": {"type": "capacity", "documents": docs, "revisits": 1, "stage": "warm"}}]},
            {"name": "revisit", "duration": "0.15s",
             "lanes": [{"name": "revisit", "rate": "40",
                        "workload": {"type": "capacity", "documents": docs, "revisits": 1, "stage": "revisit"}}]},
        ],
    }
    bc = build_config(cfg, dotenv_path=None)
    runner = BenchRunner(bc)
    result = await runner.run()
    samples = result.samples
    assert samples and all(s.ok for s in samples), [s.error for s in samples if not s.ok]
    warm = [s for s in samples if s.meta["phase"] == "warm"]
    revisit = [s for s in samples if s.meta["phase"] == "revisit"]
    assert warm and revisit
    assert {s.meta["lane"] for s in warm} == {"warm"} and {s.meta["lane"] for s in revisit} == {"revisit"}
    # lane-level model override applied only to the warm lane
    assert set(state.models) == {"warm-model", "base"}
    # the workload's own tags survive next to the phase tag
    assert {s.meta["visit"] for s in warm} == {"warm"} and {s.meta["visit"] for s in revisit} == {"revisit"}
    assert "doc_id" in warm[0].meta and warm[0].extra["prompt_tokens_hint"] > 0
    # simulated cache: warm pass misses, revisit hits from the device
    assert {s.meta["cache_class"] for s in warm} == {"miss"}
    assert {s.meta["cache_class"] for s in revisit} == {"device_hit"}
    s = result.summary
    assert set(s["phases"]) == {"warm", "revisit"}
    assert s["phases"]["warm"]["total_requests"] == len(warm)
    assert s["phases"]["revisit"]["total_requests"] == len(revisit)
    assert s["phases"]["warm"]["lanes"]["warm"]["total_requests"] == len(warm)
    # 3 documents at 40 rps exhaust a phase before its 0.15s budget: the phase
    # window closes when every lane is exhausted.
    assert 0 < s["phases"]["revisit"]["wall_time_s"] < 0.15 and s["phases"]["warm"]["wall_time_s"] < 0.15
    assert 0 < s["phases"]["revisit"]["active_s"] <= s["phases"]["revisit"]["wall_time_s"]
    assert s["phases"]["revisit"]["groups"]["cache_class"]["device_hit"]["share"] == 1.0
    assert s["phases"]["warm"]["workload_metrics"]["cached_ratio"]["mean"] == 0.0  # non-streaming stub: no ttft
    assert s["phases"]["revisit"]["workload_metrics"]["cached_ratio"]["mean"] > 0
    assert s["groups"]["cache_class"]["miss"]["share"] > 0 and s["groups"]["cache_class"]["device_hit"]["share"] > 0
    assert "warmup_excluded" not in s or s["warmup_excluded"] == 0


@pytest.mark.asyncio
async def test_phased_run_drains_between_phases_when_asked(cache_server):
    base, state = cache_server
    cfg = {
        "workload_type": {"type": "openai", "url": f"{base}/v1/chat/completions", "model": "m", "stream": False},
        "phases": [
            {"name": "p1", "duration": "0.05s", "drain": True,
             "lanes": [{"name": "a", "rate": "100", "workload": {"type": "static", "items": ["x"]}}]},
            {"name": "p2", "duration": "0.05s", "drain": True,
             "lanes": [{"name": "b", "rate": "100", "workload": {"type": "static", "items": ["y"]}}]},
        ],
    }
    runner = BenchRunner(build_config(cfg, dotenv_path=None))
    result = await runner.run()
    t0 = min(s.start_ts for s in result.samples)
    end_p1 = max(s.start_ts + s.latency_s for s in result.samples if s.meta["phase"] == "p1")
    start_p2 = min(s.start_ts for s in result.samples if s.meta["phase"] == "p2")
    # every p1 request completed before p2 started admitting
    assert end_p1 <= start_p2 + 1e-6
    windows = result.summary["phases"]
    assert windows["p1"]["wall_time_s"] >= 0.05 and windows["p2"]["wall_time_s"] >= 0.05
    # the phase window really covers the drain: p1 window ends after its last completion
    assert windows["p1"]["wall_time_s"] >= end_p1 - t0


@pytest.mark.asyncio
async def test_phased_run_without_drain_overlaps_and_warmup_tagging(cache_server):
    base, state = cache_server
    cfg = {
        "workload_type": {"type": "openai", "url": f"{base}/v1/chat/completions", "model": "m", "stream": False},
        "warmup_requests": 3,
        "phases": [
            {"name": "p1", "duration": "0.05s", "lanes": [{"name": "a", "rate": "100", "workload": {"type": "static", "items": ["x"]}}]},
            {"name": "p2", "duration": "0.05s", "lanes": [{"name": "a", "rate": "100", "workload": {"type": "static", "items": ["y"]}}]},
        ],
    }
    runner = BenchRunner(build_config(cfg, dotenv_path=None))
    result = await runner.run()
    tagged = [s for s in result.samples if s.meta.get("warmup")]
    assert len(tagged) == 3 and all(s.meta["phase"] == "p1" for s in tagged)
    s = result.summary
    assert s["warmup_excluded"] == 3 and s["total_requests"] == len(result.samples) - 3
    assert s["phases"]["p1"]["total_requests"] == len([x for x in result.samples if x.meta["phase"] == "p1" and not x.meta.get("warmup")])
    # same lane name in both phases: per-phase lane summaries keep them apart
    assert s["phases"]["p1"]["lanes"]["a"]["total_requests"] > 0 and s["phases"]["p2"]["lanes"]["a"]["total_requests"] > 0
    # without drain, the second phase admits while the first still has requests in flight
    end_p1 = max(x.start_ts + x.latency_s for x in result.samples if x.meta["phase"] == "p1")
    start_p2 = min(x.start_ts for x in result.samples if x.meta["phase"] == "p2")
    assert start_p2 < end_p1


@pytest.mark.asyncio
async def test_write_bundle_names_phases(cache_server, tmp_path):
    base, state = cache_server
    cfg = {
        "workload_type": {"type": "openai", "url": f"{base}/v1/chat/completions", "model": "m", "stream": False},
        "phases": [{"name": "only", "duration": "0.05s", "lanes": [{"name": "a", "rate": "50", "workload": {"type": "static", "items": ["x"]}}]}],
    }
    runner = BenchRunner(build_config(cfg, dotenv_path=None))
    await runner.run()
    runner.write_bundle(str(tmp_path), run_id="ph", source_config=cfg)
    meta = json.loads((tmp_path / "ph" / "meta.json").read_text())
    summary = json.loads((tmp_path / "ph" / "summary.json").read_text())
    assert meta["workload"] == "phases:only"
    assert "phases" in summary and "only" in summary["phases"]
    man = json.loads((tmp_path / "ph" / "manifest.json").read_text())
    assert man["prompt_set"]["name"] == "phases:only"


# --------------------------------------------------------------------------- #
# Warm-up exclusion, per-phase summary blocks, per-lane model overrides        #
# --------------------------------------------------------------------------- #


def _sample(extra: dict, series: list[float] | None = None, meta: dict | None = None) -> Sample:
    s = Sample(start_ts=0.0, latency_s=1.0, status=200, ok=True)
    s.extra.update(extra)
    if series is not None:
        s.series[ITL_SERIES] = series
    if meta:
        s.meta.update(meta)
    return s


def test_warmup_samples_recorded_but_excluded():
    agg = MetricsAggregator(group_by=["visit"])
    for i in range(3):
        agg.add(_sample({"ttft_s": 1.0}, [1.0], meta={"warmup": True, "visit": "warm"}))
    agg.add(_sample({"ttft_s": 0.1}, [2.0], meta={"visit": "warm"}))
    agg.add(_sample({"ttft_s": 0.2}, [3.0], meta={"visit": "revisit"}))
    agg.finalize()
    s = agg.summary()
    assert s["total_requests"] == 2 and s["warmup_excluded"] == 3
    assert s["workload_metrics"]["ttft_s"]["mean"] == pytest.approx(0.15)
    assert s["groups"]["visit"]["warm"]["share"] == 0.5
    assert s["itl_ms_pooled"]["count"] == 2
    assert len(agg.samples) == 5  # still on disk

def test_phase_windows_produce_per_phase_blocks():
    agg = MetricsAggregator()
    agg.phase_windows = {"hot-a": (0.0, 10.0), "hot-b": (10.0, 20.0)}
    agg.add(_sample({"ttft_s": 0.1}, [1.0], meta={"phase": "hot-a", "lane": "a"}))
    agg.add(_sample({"ttft_s": 0.1}, [1.0], meta={"phase": "hot-a", "lane": "a"}))
    agg.add(_sample({"ttft_s": 0.4}, [7.0], meta={"phase": "hot-b", "lane": "b"}))
    agg.finalize()
    s = agg.summary()
    assert s["phases"]["hot-a"]["total_requests"] == 2 and s["phases"]["hot-a"]["wall_time_s"] == 10.0
    assert s["phases"]["hot-b"]["workload_metrics"]["ttft_s"]["mean"] == pytest.approx(0.4)
    assert s["phases"]["hot-a"]["lanes"]["a"]["total_requests"] == 2
    assert s["phases"]["hot-b"]["itl_ms_pooled"]["max"] == 7.0

def test_apply_overrides_replaces_model():
    wt = OpenAIChatWorkloadType(url="http://x", model="a")
    req = wt.apply_overrides(Request(url="u", method="POST", json={"model": "a", "x": 1}), {"model": "b"})
    assert req.json["model"] == "b" and req.json["x"] == 1 and wt.model == "a"
