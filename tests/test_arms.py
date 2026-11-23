"""Multi-arm comparison runs (issue #3) and the compare helpers: TTFT mix/latency
decomposition (#2) and TTFT parity warnings (#4)."""

from __future__ import annotations

import asyncio
import json
import socket
import stat

import pytest
import pytest_asyncio
from aiohttp import web

from benchmaker.core.arms import (
    ArmsRunner, ArmsSettings, build_comparison, deep_merge, parse_arms_config,
    plan_order, runs_from_arm_dirs,
)
from benchmaker.core.compare import (
    RunRef, check_same_job, compare, decompose_ttft, metric_value, render_comparison,
    ttft_parity, ttft_parity_warnings,
)


# --------------------------------------------------------------------------- #
# Stub OpenAI-compatible streaming server                                      #
# --------------------------------------------------------------------------- #


async def _chat(request: web.Request) -> web.StreamResponse:
    body = await request.json()
    model = body.get("model", "m")
    # arm "slow" streams slower so the comparison has something to show.
    delay = 0.02 if "slow" in model else 0.002
    resp = web.StreamResponse(status=200, headers={"Content-Type": "text/event-stream"})
    await resp.prepare(request)
    for i in range(6):
        await asyncio.sleep(delay)
        delta = ({"reasoning_content": "think "} if (i == 0 and "reason" in model)
                 else {"content": f"tok{i} "})
        chunk = {"choices": [{"index": 0, "delta": delta, "finish_reason": None}]}
        await resp.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
    await resp.write(b"data: " + json.dumps({
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 6, "total_tokens": 26,
                  "prompt_tokens_details": {"cached_tokens": 0}},
    }).encode() + b"\n\n")
    await resp.write(b"data: [DONE]\n\n")
    await resp.write_eof()
    return resp


async def _server_info(request: web.Request) -> web.Response:
    return web.json_response({"version": "0.1", "fallbacks": [], "model": "stub"})


@pytest_asyncio.fixture
async def openai_stub() -> str:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    app = web.Application()
    app.router.add_post("/v1/chat/completions", _chat)
    app.router.add_get("/server_info", _server_info)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", port)
    await site.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        await runner.cleanup()


# --------------------------------------------------------------------------- #
# Planning                                                                     #
# --------------------------------------------------------------------------- #


def test_abba_plan_interleaves_and_pairs_reps():
    plan = plan_order(["A", "B"], reps=4, order="abba")
    assert plan == [(0, "A"), (0, "B"), (1, "B"), (1, "A"), (2, "A"), (2, "B"), (3, "B"), (3, "A")]
    # odd reps: the last block is just forward
    assert plan_order(["A", "B"], reps=3, order="abba") == [
        (0, "A"), (0, "B"), (1, "B"), (1, "A"), (2, "A"), (2, "B")]
    three = plan_order(["A", "B", "C"], reps=2, order="abba")
    assert sorted(three) == sorted([(r, a) for a in "ABC" for r in range(2)])
    assert three[:3] == [(0, "A"), (0, "B"), (0, "C")] and three[3:6] == [(1, "C"), (1, "B"), (1, "A")]


def test_rep_major_and_random_plans():
    assert plan_order(["A", "B"], reps=2, order="rep-major") == [(0, "A"), (1, "A"), (0, "B"), (1, "B")]
    p1 = plan_order(["A", "B"], reps=8, order="random", seed=7)
    p2 = plan_order(["A", "B"], reps=8, order="random", seed=7)
    assert p1 == p2 and p1 != plan_order(["A", "B"], reps=8, order="random", seed=8)
    assert sorted(p1) == sorted((r, a) for a in "AB" for r in range(8))
    with pytest.raises(ValueError):
        plan_order(["A"], 1, "nope")


def test_parse_arms_config_merges_base_and_validates():
    cfg = {
        "workload_type": {"type": "openai", "model": "m", "stream": True},
        "duration": "1s",
        "arms": {
            "reps": 3, "order": "rep-major", "warmup_reps": 1, "reference": "b",
            "arms": {
                "a": {"config": {"workload_type": {"base_url": "http://a"}}, "env": {"X": "1"}},
                "b": {"config": {"workload_type": {"base_url": "http://b", "model": "b-model"}}},
            },
        },
    }
    base, settings = parse_arms_config(cfg)
    assert "arms" not in base and settings.reps == 3 and settings.order == "rep-major"
    assert settings.warmup_reps == 1 and settings.reference == "b"
    assert [a.name for a in settings.arms] == ["a", "b"]
    merged = deep_merge(base, settings.arms[1].config)
    assert merged["workload_type"] == {"type": "openai", "model": "b-model", "stream": True,
                                       "base_url": "http://b"}
    with pytest.raises(ValueError):
        parse_arms_config({"arms": {"arms": {"a": {}}}, "reps": 0} | {"arms": {"arms": {"a": {}}, "reps": 0}})
    with pytest.raises(ValueError):
        parse_arms_config({"arms": {"arms": {"a": {}}, "order": "zigzag"}})
    with pytest.raises(ValueError):
        parse_arms_config({"arms": {"arms": {"a": {}}, "reference": "zzz"}})
    with pytest.raises(ValueError):
        parse_arms_config({"arms": {"arms": {"a": {}}, "warmup_reps": 1}})  # warmup == reps


# --------------------------------------------------------------------------- #
# Compare helpers (pure)                                                       #
# --------------------------------------------------------------------------- #


def _summary(ttft_mean: float, classes: dict | None = None, idx: float = 0.0):
    s = {"total_requests": 10, "throughput_rps": 10.0, "error_rate": 0.0,
         "itl_ms_pooled": {"p50": 10.0, "p99": 15.0, "p995": 20.0},
         "stalls": {"excess_ms": 12.0},
         "workload_metrics": {"ttft_s": {"mean": ttft_mean, "p50": ttft_mean, "p99": ttft_mean * 2},
                              "ttft_token_index": {"mean": idx}}}
    if classes:
        s["groups"] = {"cache_class": {
            c: {"share": share, "workload_metrics": {"ttft_s": {"mean": mean}}}
            for c, (share, mean) in classes.items()}}
    return s


def test_metric_value_aliases_and_pct():
    s = _summary(0.1)
    assert metric_value(s, "ttft_mean") == pytest.approx(0.1)
    assert metric_value(s, "itl_p99.5") == 20.0
    assert metric_value(s, "stall_excess_ms") == 12.0
    assert metric_value(s, "err_pct") == 0.0
    assert metric_value(s, "workload_metrics.ttft_s.p99") == pytest.approx(0.2)
    assert metric_value(s, "does.not.exist") is None


def test_compare_medians_ranges_and_ratios():
    runs = [RunRef("a", 0, "", _summary(0.10)), RunRef("a", 1, "", _summary(0.20)),
            RunRef("b", 0, "", _summary(0.05)), RunRef("b", 1, "", _summary(0.10))]
    res = compare(runs, ["ttft_mean", "itl_p99.5", "rps"])
    assert res["reference"] == "a"
    a, b = res["arms"]["a"]["metrics"]["ttft_mean"], res["arms"]["b"]["metrics"]["ttft_mean"]
    assert a["median"] == pytest.approx(0.15) and a["min"] == pytest.approx(0.10) and a["max"] == pytest.approx(0.20)
    assert b["ratio_vs_ref"] == pytest.approx(0.075 / 0.15) and b["better_than_ref"] is True
    assert "a" in res["arms"]["b"]["metrics"]["rps"]["path"] or res["arms"]["b"]["metrics"]["rps"]["ratio_vs_ref"] == 1.0
    md = render_comparison(res)
    assert "| **a** | 2 |" in md and "(×0.500 ✓)" in md
    with pytest.raises(ValueError):
        compare(runs, reference="zzz")


def test_compare_excludes_warmup_and_void():
    runs = [RunRef("a", 0, "", _summary(9.0), warmup=True), RunRef("a", 1, "", _summary(0.1)),
            RunRef("b", 0, "", _summary(0.2), meta={"void": {"void": True, "reason": "fallbacks"}}),
            RunRef("b", 1, "", _summary(0.3))]
    res = compare(runs, ["ttft_mean"])
    assert res["arms"]["a"]["metrics"]["ttft_mean"]["median"] == pytest.approx(0.1)
    assert res["arms"]["b"]["metrics"]["ttft_mean"]["median"] == pytest.approx(0.3)
    assert res["excluded"] == {"a": 1, "b": 1} and "b" in res["void"]
    assert "VOID" in render_comparison(res)


def test_same_job_enforcement():
    host = {"job_id": "42", "hostname": "n1"}
    runs = [RunRef("a", 0, "", {}, manifest={"host": host}), RunRef("b", 0, "", {}, manifest={"host": host})]
    assert check_same_job(runs) == []
    runs.append(RunRef("b", 1, "", {}, manifest={"host": {"job_id": "43", "hostname": "n1"}}))
    with pytest.raises(ValueError, match="different jobs/hosts"):
        check_same_job(runs)
    assert check_same_job(runs, allow_mixed=True)
    with pytest.raises(ValueError, match="without a manifest"):
        check_same_job([RunRef("a", 0, "", {})])

def test_decompose_ttft_is_exact_and_symmetric():
    # A: 50% hit @10ms, 50% miss @100ms -> 55ms.  B: 80% hit @8ms, 20% miss @90ms -> 24.4ms
    A = _summary(0, {"device_hit": (0.5, 0.010), "miss": (0.5, 0.100)})
    B = _summary(0, {"device_hit": (0.8, 0.008), "miss": (0.2, 0.090)})
    d = decompose_ttft(A, B)
    assert d["delta_ms"] == pytest.approx(24.4 - 55.0)
    assert d["mix_ms"] + d["latency_ms"] == pytest.approx(d["delta_ms"])
    assert d["mix_ms"] < 0 and d["latency_ms"] < 0
    assert d["mix_share"] == pytest.approx(d["mix_ms"] / d["delta_ms"])
    # swap: same magnitudes, opposite signs
    d2 = decompose_ttft(B, A)
    assert d2["delta_ms"] == pytest.approx(-d["delta_ms"]) and d2["mix_ms"] == pytest.approx(-d["mix_ms"])
    # class present on one side only lands entirely in the mix term
    C = _summary(0, {"device_hit": (0.5, 0.010), "miss": (0.5, 0.100), "restore": (0.0, 0.050)})
    d3 = decompose_ttft(A, C)
    assert abs(d3["latency_ms"]) < 1e-9 and d3["delta_ms"] == pytest.approx(0.0)

def test_decomposition_and_parity_exclude_void_reps():
    """Void reps are not in the table (compare excludes them), so they must not
    weigh on the TTFT decomposition or the parity medians either (#17)."""
    void = {"void": True, "violations": [{"rule": "ready_check"}]}
    a0 = RunRef("a", 0, "", _summary(0.10, {"device_hit": (0.5, 0.010), "miss": (0.5, 0.100)}))
    a1 = RunRef("a", 1, "", _summary(0.12, {"device_hit": (0.5, 0.010), "miss": (0.5, 0.100)}))
    # a void rep with a wildly different mix/latency: it would flip the decomposition
    a2 = RunRef("a", 2, "", _summary(9.0, {"restore": (1.0, 9.0)}), meta={"void": True, "void_violations": void["violations"]})
    b0 = RunRef("b", 0, "", _summary(0.05, {"device_hit": (0.8, 0.008), "miss": (0.2, 0.090)}, idx=0.0))
    b1 = RunRef("b", 1, "", _summary(0.05, {"device_hit": (0.8, 0.008), "miss": (0.2, 0.090)}, idx=0.0))
    b2 = RunRef("b", 2, "", _summary(0.05, {"device_hit": (0.8, 0.008), "miss": (0.2, 0.090)}, idx=7.0),
                meta={"void": True})
    res = build_comparison([a0, a1, a2, b0, b1, b2], ArmsSettings(metrics=["ttft_mean"], reps=3))
    clean = build_comparison([a0, a1, b0, b1], ArmsSettings(metrics=["ttft_mean"], reps=2))
    dec_b = next(d for d in res["decomposition"] if d["arm"] == "b")
    dec_c = next(d for d in clean["decomposition"] if d["arm"] == "b")
    assert dec_b == dec_c and dec_b["per_class"]
    assert "a" in res["void"] and "b" in res["void"]
    parities = {e["arm"]: e for e in res["ttft_parity"]}
    assert parities["b"]["index_median"] == 0.0 and parities["a"]["index_median"] == 0.0
    assert ttft_parity([a0, a1, a2, b0, b1, b2], exclude_void=False)[1]["index_values"] == [0.0, 0.0, 7.0]


def test_ttft_parity_detects_reasoning_parser_mismatch():
    a = RunRef("a", 0, "", _summary(0.1, idx=0.0), meta={"source_config": {"workload_type": {"ttft_token": "any"}}})
    b = RunRef("b", 0, "", _summary(0.1, idx=2.0), meta={"source_config": {"workload_type": {"ttft_token": "content"}}})
    entries = ttft_parity([a, b])
    assert [e["kind"] for e in entries] == ["any", "content"]
    warns = ttft_parity_warnings(entries)
    assert any("different token kinds" in w for w in warns) and any("first-token index" in w for w in warns)
    assert ttft_parity_warnings(ttft_parity([a, RunRef("c", 0, "", _summary(0.1))])) == []


# --------------------------------------------------------------------------- #
# End-to-end: ArmsRunner against the stub server                               #
# --------------------------------------------------------------------------- #


async def test_arms_runner_end_to_end(openai_stub: str, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    hook = tmp_path / "launch.sh"
    hook.write_text("#!/bin/sh\necho launched $BENCHMAKER_ARM rep=$BENCHMAKER_REP\n")
    hook.chmod(hook.stat().st_mode | stat.S_IEXEC)
    cfg = {
        "workload_type": {"type": "openai", "url": f"{openai_stub}/v1/chat/completions",
                          "model": "fast", "stream": True, "include_usage": True},
        "workload": {"type": "static", "items": ["hello world"]},
        "load": 20, "duration": "0.3s",
        "capability": {"url": f"{openai_stub}/server_info"},
        "group_by": ["cache_class"],
        "arms": {
            "reps": 2, "order": "abba", "warmup_reps": 1,
            "arms": {
                "fast": {"launch": str(hook), "labels": {"tier": "gpu"}},
                "slow": {"config": {"workload_type": {"model": "slow-reason"}},
                         "env": {"ARM_FLAG": "1"}},
            },
        },
    }
    runner = ArmsRunner(cfg, out_dir=str(tmp_path / "out"), run_id="r1", dotenv_path=None, progress=False)
    result = await runner.run_async()

    root = tmp_path / "out" / "r1"
    assert (root / "plan.json").is_file() and (root / "comparison.md").is_file()
    for arm in ("fast", "slow"):
        for rep in range(2):
            d = root / "arms" / arm / f"rep-{rep}"
            assert (d / "summary.json").is_file() and (d / "manifest.json").is_file()
            assert (d / "capability.start.json").is_file() and (d / "capability.end.json").is_file()
    assert result["reference"] == "fast" and result["reps"] == 2 and result["warmup_reps"] == 1
    assert result["arms"]["fast"]["reps"] == 1 and result["arms"]["slow"]["reps"] == 1  # warm-up excluded
    assert result["arms"]["slow"]["metrics"]["ttft_p50"]["ratio_vs_ref"] > 1.0
    assert result["arms"]["slow"]["metrics"]["ttft_p50"]["better_than_ref"] is False
    assert result["same_job_violations"] == []
    # launch hook ran once for "fast" (relaunch: once), with the arm env exported
    log = (root / "arms" / "fast" / "launch-rep-0.log").read_text()
    assert "launched fast rep=0" in log and not (root / "arms" / "fast" / "launch-rep-1.log").exists()
    # arm env recorded in the manifest, labels carry arm/rep/warmup
    man = json.load(open(root / "arms" / "slow" / "rep-0" / "manifest.json"))
    assert man["env"]["ARM_FLAG"] == "1" and man["host"]["hostname"]
    meta = json.load(open(root / "arms" / "slow" / "rep-0" / "meta.json"))
    assert meta["labels"]["arm"] == "slow" and meta["labels"]["rep"] == 0 and meta["labels"]["warmup"] == "true"
    assert man["model"]["id"] == "slow-reason"

    # re-compare from disk with the same result, and refuse mixed jobs
    from benchmaker.core.arms import discover_arm_dirs
    runs = runs_from_arm_dirs(discover_arm_dirs(str(root)), warmup_reps=1)
    again = build_comparison(runs, ArmsSettings(metrics=result["metrics"], warmup_reps=1, reps=2))
    assert again["arms"]["slow"]["metrics"]["ttft_p50"]["values"] == result["arms"]["slow"]["metrics"]["ttft_p50"]["values"]
    mpath = root / "arms" / "slow" / "rep-1" / "manifest.json"
    m = json.load(open(mpath))
    m["host"]["job_id"] = "other-job"
    json.dump(m, open(mpath, "w"))
    with pytest.raises(ValueError):
        check_same_job(runs_from_arm_dirs(discover_arm_dirs(str(root))))
    assert check_same_job(runs_from_arm_dirs(discover_arm_dirs(str(root))), allow_mixed=True)


async def test_arms_cli_dry_run_and_compare(openai_stub, tmp_path, monkeypatch):
    from click.testing import CliRunner
    from benchmaker.cli import main

    monkeypatch.chdir(tmp_path)
    cfg = {
        "workload_type": {"type": "openai", "url": f"{openai_stub}/v1/chat/completions", "model": "m"},
        "load": 50, "duration": "0.2s",
        "arms": {"reps": 2, "arms": {"a": {}, "b": {"config": {"workload_type": {"model": "slow"}}}}},
    }
    cfg_path = tmp_path / "arms.yaml"
    cfg_path.write_text(json.dumps(cfg))
    out = tmp_path / "out"
    cli = CliRunner()
    r = cli.invoke(main, ["arms", str(cfg_path), "--out-dir", str(out), "--run-id", "dry", "--dry-run"])
    assert r.exit_code == 0, r.output
    # no hooks on either arm: nothing to start/stop, so no markers or overlap note
    assert r.output.splitlines() == ["a rep-0", "b rep-0", "b rep-1", "a rep-1"]

    # With hooks and relaunch: once, both servers are up during each other's
    # reps (ABBA) — the dry run says so, and marks where the stop hooks run (#14).
    cfg2 = json.loads(json.dumps(cfg))
    cfg2["order"] = "abba"
    cfg2["arms"]["arms"] = {"a": {"launch": "true", "stop": "true"},
                            "b": {"launch": "true", "stop": "true",
                                  "config": {"workload_type": {"model": "slow"}}}}
    cfg_path.write_text(json.dumps(cfg2))
    r = cli.invoke(main, ["arms", str(cfg_path), "--out-dir", str(out), "--run-id", "dry2", "--dry-run"])
    assert r.exit_code == 0, r.output
    assert r.output.splitlines() == [
        "a rep-0", "b rep-0", "b rep-1  + stop", "a rep-1  + stop",
        "⚠ a and b are both live at some point (relaunch once/once); use relaunch: visit or "
        "every_rep if they share GPUs",
    ]
    cfg2["arms"]["arms"]["a"]["relaunch"] = "visit"
    cfg_path.write_text(json.dumps(cfg2))
    r = cli.invoke(main, ["arms", str(cfg_path), "--out-dir", str(out), "--run-id", "dry3", "--dry-run"])
    assert r.output.splitlines() == ["a rep-0  + stop", "b rep-0", "b rep-1  + stop", "a rep-1  + stop"]
    cfg_path.write_text(json.dumps(cfg))

    # the CLI owns its event loop; run it in a thread so the stub keeps serving
    r = await asyncio.to_thread(cli.invoke, main, ["arms", str(cfg_path), "--out-dir", str(out),
                                                   "--run-id", "e2e", "--quiet", "--label", "env=ci"])
    assert r.exit_code == 0, r.output
    assert "| **a** | 2 |" in r.output and "| b | 2 |" in r.output
    assert "✓" not in r.output or "✗" in r.output  # slow arm can't be better on ttft

    r = cli.invoke(main, ["compare", str(out), "--metric", "ttft_mean"])
    assert r.exit_code == 0, r.output
    assert "| b | 2 |" in r.output

    # mixed jobs -> exit 2 unless allowed
    mpath = out / "e2e" / "arms" / "b" / "rep-1" / "manifest.json"
    m = json.load(open(mpath))
    m["host"]["hostname"] = "elsewhere"
    json.dump(m, open(mpath, "w"))
    r = cli.invoke(main, ["compare", str(out / "e2e")])
    assert r.exit_code == 2 and "different jobs/hosts" in r.output
    r = cli.invoke(main, ["compare", str(out / "e2e"), "--allow-mixed-jobs", "--format", "json"])
    assert r.exit_code == 0 and json.loads(r.output)["same_job_violations"]
