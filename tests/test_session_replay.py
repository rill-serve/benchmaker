"""Session replay (issue #10): file integrity, the session scheduler, think time, warm-up tagging, template check."""
from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from benchmaker.workloads.session_replay import (
    ReplayWorkload,
    group_sessions,
    load_replay,
    prompt_set_sha256,
    template_check,
    think_seconds,
    to_item,
)


def _rows(sessions=3, turns=2, *, model="a", at_step=0.0, warmup_first=True):
    rows = []
    for s in range(sessions):
        for t in range(turns):
            row = {"id": f"s{s}.t{t}", "session": f"s{s}", "turn": t, "model": model,
                   "messages": [{"role": "user", "content": f"s{s} t{t}"}], "max_tokens": 5,
                   "tag": "warmup" if warmup_first and s == 0 else "measure", "expect": {"input_tokens": 10 + t}}
            if t:
                row["think_z"] = 0.0
            else:
                row["at_s"] = s * at_step
            rows.append(row)
    return rows


def _write(path, rows, **manifest):
    head = {"kind": "manifest", "prompt_set_sha256": prompt_set_sha256(rows), **manifest}
    path.write_text("\n".join(json.dumps(r) for r in [head, *rows]) + "\n")
    return str(path)


class _Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


async def _drain(w, *, complete_after=None):
    """Pull every item; complete each turn right away (or via complete_after(item))."""
    out = []
    while True:
        try:
            item = await asyncio.wait_for(w.next_item(), timeout=2)
        except StopAsyncIteration:
            return out
        out.append(item)
        (complete_after or (lambda it: w.notify_turn_complete(it["meta"]["conversation_id"])))(item)


# --- file integrity ---------------------------------------------------------------------------------------------

def test_hash_round_trip_and_tamper(tmp_path):
    rows = _rows()
    path = _write(tmp_path / "r.jsonl", rows)
    manifest, got = load_replay(path)
    assert got == rows and manifest["prompt_set_sha256"] == prompt_set_sha256(rows)
    lines = (tmp_path / "r.jsonl").read_text().splitlines()
    lines[2] = lines[2].replace("s0 t1", "s0 tX")
    (tmp_path / "r.jsonl").write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="hashes to"):
        load_replay(path)
    load_replay(path, verify=False)


def test_first_line_must_be_manifest(tmp_path):
    (tmp_path / "r.jsonl").write_text(json.dumps(_rows()[0]) + "\n")
    with pytest.raises(ValueError, match="manifest"):
        load_replay(str(tmp_path / "r.jsonl"))


def test_sessions_keep_turn_order():
    rows = _rows(2, 3)
    assert [[r["id"] for r in g] for g in group_sessions(rows)] == [
        ["s0.t0", "s0.t1", "s0.t2"], ["s1.t0", "s1.t1", "s1.t2"]]
    with pytest.raises(ValueError, match="out of order"):
        group_sessions([rows[1], rows[0]])


# --- items ------------------------------------------------------------------------------------------------------

def test_item_pins_output_and_tags_warmup():
    rows = _rows()
    item = to_item(rows[0])
    assert item["max_tokens"] == item["min_tokens"] == 5 and item["ignore_eos"] is True
    assert item["meta"]["warmup"] is True and item["meta"]["conversation_id"] == "s0"
    assert item["meta"]["expected_input_tokens"] == 10
    assert "warmup" not in to_item(rows[2])["meta"]
    with pytest.raises(ValueError, match="no messages"):
        to_item({"id": "x", "session": "s", "prompt_ids": [1], "max_tokens": 1})


def test_think_seconds():
    assert think_seconds({"think_z": 1.0}, 10.0, 0.5) == pytest.approx(10.0 * 2.718281828 ** 0.5)
    assert think_seconds({"after": {"think_s": 3.0}, "think_z": 9}, 10.0, 0.5) == 3.0
    assert think_seconds({}, 10.0, 0.5) == 0.0
    assert think_seconds({"think_z": 1.0}, 0.0, 0.5) == 0.0


# --- scheduling -------------------------------------------------------------------------------------------------

def test_closed_loop_gates_each_turn_on_completion(tmp_path):
    path = _write(tmp_path / "r.jsonl", _rows(4, 3), think={"median_s": 0, "sigma": 0})
    w = ReplayWorkload(path, concurrent_sessions=2)
    in_flight, peak, seen = set(), [0], []

    async def run():
        pending = []
        while True:
            try:
                item = await asyncio.wait_for(w.next_item(), timeout=0.05)
            except asyncio.TimeoutError:  # every active session is waiting on its in-flight turn
                sid = pending.pop(0)
                in_flight.discard(sid)
                w.notify_turn_complete(sid)
                continue
            except StopAsyncIteration:
                return
            sid = item["meta"]["conversation_id"]
            assert sid not in in_flight, "a session's next turn went out before its previous one completed"
            in_flight.add(sid)
            peak[0] = max(peak[0], len(in_flight))
            seen.append(item["meta"]["replay_id"])
            pending.append(sid)

    asyncio.run(run())
    assert len(seen) == 12 and peak[0] == 2
    for s in range(4):
        assert [i for i in seen if i.startswith(f"s{s}.")] == [f"s{s}.t0", f"s{s}.t1", f"s{s}.t2"]


def test_think_time_delays_the_next_turn(tmp_path):
    rows = _rows(1, 2)
    path = _write(tmp_path / "r.jsonl", rows, think={"median_s": 5.0, "sigma": 0.0})
    clock = _Clock()
    w = ReplayWorkload(path, concurrent_sessions=1, clock=clock)
    assert w.think_median_s == 5.0

    async def run():
        first = await w.next_item()
        w.notify_turn_complete(first["meta"]["conversation_id"])
        task = asyncio.ensure_future(w.next_item())
        await asyncio.sleep(0.05)
        assert not task.done()  # 5 s of think time pending on the fake clock
        clock.t += 5.0
        w._wakeup.set()
        return await asyncio.wait_for(task, timeout=2)

    assert asyncio.run(run())["meta"]["replay_id"] == "s0.t1"
    assert ReplayWorkload(path, concurrent_sessions=1, think_median_s=0.0).think_median_s == 0.0


def test_trace_scale_orders_session_starts(tmp_path):
    path = _write(tmp_path / "r.jsonl", _rows(3, 1, at_step=10.0))
    clock = _Clock()
    w = ReplayWorkload(path, trace_scale=0.5, clock=clock)

    async def run():
        got = [(await w.next_item())["meta"]["conversation_id"]]
        nxt = asyncio.ensure_future(w.next_item())
        await asyncio.sleep(0.05)
        assert not nxt.done()  # s1 starts at 10 s x 0.5
        clock.t += 5.0
        w._wakeup.set()
        got.append((await asyncio.wait_for(nxt, timeout=2))["meta"]["conversation_id"])
        return got

    assert asyncio.run(run()) == ["s0", "s1"]


def test_open_loop_is_seeded_and_completes(tmp_path):
    path = _write(tmp_path / "r.jsonl", _rows(5, 2), think={"median_s": 0, "sigma": 0})
    a = ReplayWorkload(path, session_rate=1000.0, seed=3)
    b = ReplayWorkload(path, session_rate=1000.0, seed=3)
    a._t0 = b._t0 = 0.0
    a._plan_starts()
    b._plan_starts()
    assert a._starts == b._starts and a._starts == sorted(a._starts)
    w = ReplayWorkload(path, session_rate=1000.0, seed=3)
    assert len(asyncio.run(_drain(w))) == 10


def test_model_label_filters_sessions(tmp_path):
    rows = _rows(2, 1, model="a") + [dict(r, session=f"b{r['session']}", id=f"b{r['id']}", model="b")
                                     for r in _rows(3, 1)]
    path = _write(tmp_path / "r.jsonl", rows)
    assert len(asyncio.run(_drain(ReplayWorkload(path, concurrent_sessions=4, model_label="b")))) == 3
    with pytest.raises(ValueError, match="no sessions"):
        ReplayWorkload(path, concurrent_sessions=1, model_label="c")


def test_exactly_one_mode(tmp_path):
    path = _write(tmp_path / "r.jsonl", _rows())
    for kwargs in ({}, {"concurrent_sessions": 2, "session_rate": 1.0}, {"concurrent_sessions": 0}):
        with pytest.raises(ValueError):
            ReplayWorkload(path, **kwargs)
    w = ReplayWorkload(path, concurrent_sessions=1)
    assert w.item_count() == 6 and w.content_hash() == prompt_set_sha256(_rows())


def test_completion_hook_advances_the_session(tmp_path):
    path = _write(tmp_path / "r.jsonl", _rows(1, 2), think={"median_s": 0, "sigma": 0})
    w = ReplayWorkload(path, concurrent_sessions=1)

    class Req:
        meta = {"conversation_id": "s0"}

    async def run():
        await w.next_item()
        assert w.completion_hook()(Req(), None, "sample") == "sample"
        return await asyncio.wait_for(w.next_item(), timeout=2)

    assert asyncio.run(run())["meta"]["replay_id"] == "s0.t1"


# --- template check ---------------------------------------------------------------------------------------------

class _Stub(BaseHTTPRequestHandler):
    tokens = 7

    def log_message(self, *a):
        pass

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        body = json.dumps({"usage": {"prompt_tokens": _Stub.tokens}}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def test_template_check_reads_server_prompt_tokens():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Stub)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{srv.server_address[1]}/v1/chat/completions"
        probe = {"messages": [{"role": "user", "content": "hi"}], "prompt_tokens": 7}
        assert template_check(url, "m", probe) == (7, 7)
        _Stub.tokens = 9
        assert template_check(url, "m", probe) == (9, 7)
    finally:
        srv.shutdown()


def test_yaml_and_recipe_registration(tmp_path):
    from benchmaker import recipes
    from benchmaker.config import build_workload

    path = _write(tmp_path / "r.jsonl", _rows())
    w = build_workload({"type": "session-replay", "path": path, "concurrent_sessions": 2})
    assert isinstance(w, ReplayWorkload)
    assert recipes.get("session-replay").name == "session-replay"
