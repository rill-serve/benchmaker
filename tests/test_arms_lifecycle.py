"""Arm lifecycle hooks (issue #9): start/stop, relaunch modes, ready_check and
post_stop_check, and how their failures void the affected reps."""

from __future__ import annotations

import json
import socket
import stat

import pytest
import pytest_asyncio
from aiohttp import web

from benchmaker.core.arms import ArmSpec, ArmsRunner, evaluate_check, normalize_check, run_hook, stops_after
from benchmaker.io.bundle import read_bundle


async def _chat(request: web.Request) -> web.StreamResponse:
    await request.json()
    resp = web.StreamResponse(status=200, headers={"Content-Type": "text/event-stream"})
    await resp.prepare(request)
    chunk = {"choices": [{"index": 0, "delta": {"content": "ok "}, "finish_reason": "stop"}],
             "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
    await resp.write(b"data: " + json.dumps(chunk).encode() + b"\n\ndata: [DONE]\n\n")
    await resp.write_eof()
    return resp


@pytest_asyncio.fixture
async def openai_stub() -> str:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    app = web.Application()
    app.router.add_post("/v1/chat/completions", _chat)
    app.router.add_get("/health", lambda r: web.json_response({"ok": True}))
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", port)
    await site.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        await runner.cleanup()


def _script(path, body: str) -> str:
    path.write_text(f"#!/bin/sh\n{body}\n")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return str(path)


def _cfg(openai_stub, arms_block: dict) -> dict:
    """``arms_block`` is the whole ``arms:`` mapping (reps/order/arms/...)."""
    return {
        "workload_type": {"type": "openai", "url": f"{openai_stub}/v1/chat/completions",
                          "model": "m", "stream": True, "include_usage": True},
        "workload": {"type": "static", "items": ["hello"]},
        "load": 10, "duration": "0.1s",
        "arms": arms_block,
    }


def _meta(root, arm, rep):
    return json.load(open(root / "arms" / arm / f"rep-{rep}" / "meta.json"))


def _hook_events(root, arm, kind):
    d = root / "arms" / arm
    return sorted(p.name for p in d.glob(f"{kind}-*.log"))


# --------------------------------------------------------------------------- #
# Unit level                                                                    #
# --------------------------------------------------------------------------- #


def test_arm_spec_accepts_start_alias_and_relaunch_modes(tmp_path):
    a = ArmSpec.from_dict("x", {"launch": "./l.sh", "relaunch": "every-rep", "stop": "./s.sh",
                                "ready_check": {"url": "http://h/health"}})
    assert a.start == "./l.sh" and a.relaunch == "every_rep" and a.stop == "./s.sh"
    assert a.ready_check == {"url": "http://h/health"}
    with pytest.raises(ValueError, match="relaunch"):
        ArmSpec.from_dict("x", {"relaunch": "sometimes"})
    with pytest.raises(TypeError):
        ArmSpec.from_dict("x", {"ready_check": "http://h/health"})


def test_stops_after_follows_relaunch_mode():
    arm_once = ArmSpec(name="a", relaunch="once")
    arm_visit = ArmSpec(name="a", relaunch="visit")
    arm_rep = ArmSpec(name="a", relaunch="every_rep")
    plan = [(0, "a"), (0, "b"), (1, "b"), (1, "a")]  # abba
    assert [stops_after(arm_visit, plan, i) for i in range(4)] == [True, False, True, True]
    assert not any(stops_after(arm_once, plan, i) for i in range(4))
    assert all(stops_after(arm_rep, plan, i) for i in range(4))


def test_run_hook_records_exit_output_and_timeouts(tmp_path):
    ok = run_hook("echo hi $BENCHMAKER_ARM", arm="a", rep=1, env={}, log_path=str(tmp_path / "ok.log"),
                  timeout_s=10)
    assert ok["exit_code"] == 0 and ok["timed_out"] is False and "hi a" in ok["output_tail"]
    bad = run_hook("exit 3", arm="a", rep=None, env={}, log_path=str(tmp_path / "bad.log"), timeout_s=10)
    assert bad["exit_code"] == 3
    slow = run_hook("sleep 30", arm="a", rep=None, env={}, log_path=str(tmp_path / "slow.log"), timeout_s=0.3)
    assert slow["timed_out"] is True and slow["exit_code"] is None and "timed out" in slow["error"]
    assert slow["duration_s"] < 15


def test_normalize_check_and_evaluate_polling():
    spec = normalize_check({"url": "http://127.0.0.1:1/health", "timeout_s": 0.05, "interval_s": 0.01},
                           default_timeout_s=5)
    assert spec["status"] == 200 and spec["timeout_s"] == 0.05
    assert normalize_check({"urls": "http://x/health"}, default_timeout_s=5)["urls"] == ["http://x/health"]
    with pytest.raises(ValueError):
        normalize_check({"status": 200}, default_timeout_s=5)
    rec = evaluate_check(spec, kind="ready_check", arm="a", rep=0, env={}, log_path="/dev/null",
                         timeout_s=0.1, poll=True)
    assert rec["ok"] is False and rec["attempts"] >= 1 and "not ready" in rec["error"]
    rec = evaluate_check(normalize_check({"run": "true"}, default_timeout_s=5), kind="post_stop_check",
                         arm="a", rep=0, env={}, log_path="/tmp/does-not-matter.log", timeout_s=5, poll=False)
    assert rec["ok"] is True and rec["run"]["exit_code"] == 0
    rec = evaluate_check(normalize_check({"run": "false", "expect_exit": 1}, default_timeout_s=5),
                         kind="post_stop_check", arm="a", rep=0, env={}, log_path="/dev/null",
                         timeout_s=5, poll=False)
    assert rec["ok"] is True
    rec = evaluate_check(normalize_check({"run": "true", "expect_exit": 1}, default_timeout_s=5),
                         kind="post_stop_check", arm="a", rep=0, env={}, log_path="/dev/null",
                         timeout_s=5, poll=False)
    assert rec["ok"] is False and "exit 0" in rec["error"]
    # the spec's own timeout_s is the polling budget (not the settings-level fallback):
    # a dead URL with timeout_s 0.1 must give up after ~0.1 s even if the caller says 5
    import time
    t0 = time.monotonic()
    rec = evaluate_check(normalize_check({"url": "http://127.0.0.1:1/health", "timeout_s": 0.1,
                                          "interval_s": 0.02}, default_timeout_s=5),
                         kind="ready_check", arm="a", rep=0, env={}, log_path="/dev/null",
                         timeout_s=5, poll=True)
    assert rec["ok"] is False and time.monotonic() - t0 < 2 and rec["timeout_s"] == 0.1
    assert rec["attempts"] >= 3 and "not ready after" in rec["error"]


# --------------------------------------------------------------------------- #
# Runner level                                                                  #
# --------------------------------------------------------------------------- #


async def test_visit_relaunch_wraps_each_visit_with_start_stop(openai_stub, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    start = _script(tmp_path / "start.sh", "echo start $BENCHMAKER_ARM rep=$BENCHMAKER_REP >> events.txt")
    stop = _script(tmp_path / "stop.sh", "echo stop $BENCHMAKER_ARM >> events.txt")
    cfg = _cfg(openai_stub, {
        "reps": 2, "order": "abba",
        "arms": {"a": {"start": start, "stop": stop, "relaunch": "visit",
                       "ready_check": {"urls": [f"{openai_stub}/health"], "timeout_s": 2}},
                 "b": {}},
    })
    runner = ArmsRunner(cfg, out_dir=str(tmp_path / "out"), run_id="r", dotenv_path=None, progress=False)
    result = await runner.run_async()
    # plan: a0 b0 b1 a1 -> visits: a(rep0), b(rep0-1), a(rep1)
    events = (tmp_path / "events.txt").read_text().split("\n")
    events = [e for e in events if e]
    assert events == ["start a rep=0", "stop a", "start a rep=1", "stop a"]
    root = tmp_path / "out" / "r"
    assert _hook_events(root, "a", "launch") == ["launch-rep-0.log", "launch-rep-1.log"]
    assert _hook_events(root, "a", "stop") == ["stop-rep-0.log", "stop-rep-1.log"]
    assert not (root / "arms" / "b" / "launch-rep-0.log").exists()
    for rep in (0, 1):
        hooks = _meta(root, "a", rep)["hooks"]
        assert hooks["start"]["exit_code"] == 0 and hooks["start"]["command"] == start
        assert hooks["ready_check"]["ok"] is True and hooks["stop"]["exit_code"] == 0
        assert hooks["stop"]["output_tail"] == ""  # the script logs to events.txt, not stdout
    assert "hooks" not in _meta(root, "b", 0)
    assert result["void"] == {}


async def test_once_relaunch_stops_at_end_and_records_hooks(openai_stub, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    start = _script(tmp_path / "start.sh", "echo start")
    stop = _script(tmp_path / "stop.sh", "echo stop")
    cfg = _cfg(openai_stub, {
        "reps": 2, "order": "rep-major",
        "arms": {"a": {"launch": start, "stop": stop, "post_stop_check": {"run": "true"}}},
    })
    runner = ArmsRunner(cfg, out_dir=str(tmp_path / "out"), run_id="r", dotenv_path=None, progress=False)
    await runner.run_async()
    root = tmp_path / "out" / "r"
    assert _hook_events(root, "a", "launch") == ["launch-rep-0.log"]
    assert _hook_events(root, "a", "stop") == ["stop-final.log"]
    for rep in (0, 1):
        hooks = _meta(root, "a", rep)["hooks"]
        assert hooks["start"]["exit_code"] == 0 and hooks["stop"]["exit_code"] == 0
        assert hooks["post_stop_check"]["ok"] is True
    assert read_bundle(str(root / "arms" / "a" / "rep-1"))["meta"]["void"] is None  # no rules, no violations


async def test_failed_ready_check_voids_the_rep(openai_stub, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    stop = _script(tmp_path / "stop.sh", "echo stopped")
    cfg = _cfg(openai_stub, {
        "reps": 1,
        "arms": {"a": {"stop": stop, "ready_check": {"urls": ["http://127.0.0.1:1/health"],
                                                       "timeout_s": 0.1, "interval_s": 0.02}}},
        "ready_timeout_s": 1,
    })
    runner = ArmsRunner(cfg, out_dir=str(tmp_path / "out"), run_id="r", dotenv_path=None, progress=False)
    result = await runner.run_async()
    root = tmp_path / "out" / "r"
    meta = _meta(root, "a", 0)
    assert meta["void"] is True and meta["void_violations"][0]["rule"] == "ready_check"
    assert meta["hooks"]["ready_check"]["ok"] is False and meta["hooks"]["ready_check"]["attempts"] > 1
    assert meta["hooks"]["stop"]["exit_code"] == 0  # torn down regardless
    assert read_bundle(str(root / "arms" / "a" / "rep-0"))["summary"]["void"]["void"] is True
    assert result["void"]["a"]["void"] is True
    assert result["arms"]["a"]["reps"] == 0  # nothing left to compare


async def test_failed_post_stop_check_voids_the_launch(openai_stub, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = _cfg(openai_stub, {
        "reps": 2, "order": "rep-major",
        "arms": {"a": {"stop": _script(tmp_path / "stop.sh", "exit 2"),
                        "post_stop_check": {"run": "pgrep -x definitely-not-running-xyz"}}},
    })
    runner = ArmsRunner(cfg, out_dir=str(tmp_path / "out"), run_id="r", dotenv_path=None, progress=False)
    result = await runner.run_async()
    root = tmp_path / "out" / "r"
    for rep in (0, 1):
        b = read_bundle(str(root / "arms" / "a" / f"rep-{rep}"))
        assert b["meta"]["void"] is True
        assert [v["rule"] for v in b["meta"]["void_violations"]] == ["post_stop_check"]
        assert b["summary"]["void"]["void"] is True
        assert b["meta"]["hooks"]["stop"]["exit_code"] == 2 and b["meta"]["hooks"]["stop"]["ok"] is False
        assert b["meta"]["hooks"]["post_stop_check"]["ok"] is False
        assert "definitely-not-running-xyz" in b["meta"]["hooks"]["post_stop_check"]["error"]
    assert result["void"]["a"]["void"] is True


async def test_failed_start_hook_runs_stop_and_aborts(openai_stub, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = _cfg(openai_stub, {
        "reps": 1,
        "arms": {"a": {"start": _script(tmp_path / "start.sh", "echo boom >&2; exit 3"),
                        "stop": _script(tmp_path / "stop.sh", "echo stopped"),
                        "ready_check": {"urls": [f"{openai_stub}/health"]}}},
    })
    runner = ArmsRunner(cfg, out_dir=str(tmp_path / "out"), run_id="r", dotenv_path=None, progress=False)
    with pytest.raises(RuntimeError, match="start hook for arm 'a' failed"):
        await runner.run_async()
    root = tmp_path / "out" / "r"
    assert (root / "arms" / "a" / "stop-rep-0.log").read_text().strip() == "stopped"
    assert "boom" in (root / "arms" / "a" / "launch-rep-0.log").read_text()
    assert not (root / "arms" / "a" / "rep-0").exists()


async def test_stop_runs_when_a_rep_fails(openai_stub, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    stop = _script(tmp_path / "stop.sh", "echo stopped $BENCHMAKER_REP")
    cfg = _cfg(openai_stub, {"reps": 2, "order": "rep-major", "arms": {"a": {"start": "true", "stop": stop}}})
    from benchmaker.core.runner import BenchRunner

    real_run = BenchRunner.run
    calls = {"n": 0}

    async def flaky(self):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("server died mid-run")
        return await real_run(self)

    monkeypatch.setattr(BenchRunner, "run", flaky)
    runner = ArmsRunner(cfg, out_dir=str(tmp_path / "out"), run_id="r", dotenv_path=None, progress=False)
    with pytest.raises(RuntimeError, match="server died mid-run"):
        await runner.run_async()
    root = tmp_path / "out" / "r"
    assert (root / "arms" / "a" / "stop-final.log").read_text().strip() == "stopped 1"
    assert "hooks" in _meta(root, "a", 0)


async def test_arm_endpoints_merge_into_config(openai_stub, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = _cfg(openai_stub, {"reps": 1, "arms": {"a": {"endpoints": {"m": openai_stub}}}})
    runner = ArmsRunner(cfg, out_dir=str(tmp_path / "out"), run_id="r", dotenv_path=None, progress=False)
    await runner.run_async()
    man = json.load(open(tmp_path / "out" / "r" / "arms" / "a" / "rep-0" / "manifest.json"))
    # the arm's base URL is routed through the workload-type's path
    assert man["endpoints"]["m"]["url"] == f"{openai_stub}/v1/chat/completions"
    assert man["endpoints"]["m"]["configured_url"] == openai_stub
