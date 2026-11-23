"""Session replay — run a prepared session file with per-session turn dependencies (issue #10).

A replay file is JSONL: one manifest row, then one row per request, grouped by session in send order::

    {"kind": "manifest", "prompt_set_sha256": "...", "think": {"median_s": 10, "sigma": 1.4}, "probe": {...}, ...}
    {"id": "chat-00012.t03", "session": "chat-00012", "turn": 3, "model": "a",
     "messages": [...], "max_tokens": 180, "think_z": 0.41, "tag": "measure", "expect": {"input_tokens": 5321}}

Each row is one request whose ``messages`` already hold the recorded history (never the server's output), so every
arm sees byte-identical prompts. ``max_tokens`` is the recorded output length; the item pins it with
``min_tokens = max_tokens`` and ``ignore_eos``.

Scheduling. Turn k+1 of a session is eligible only after turn k completed plus a think time:

* the row's ``after.think_s`` (seconds), or
* ``median * exp(sigma * think_z)`` (a lognormal) with ``median``/``sigma`` from the manifest's ``think`` block,
  ``think_median_s`` overriding the median at run time so one prompt set can sweep think time.

Session starts:

* closed loop (``concurrent_sessions=N``): N sessions active, the next starts when one finishes;
* open loop (``session_rate=R``): session starts are Poisson at R per second (seeded);
* trace (``trace_scale=S``): each session starts at its first row's ``at_s`` times S.

``model_label`` keeps the sessions whose rows carry that ``model`` label (a file prepared for several models,
one endpoint per model).

Integrity. The prompt set is re-hashed on load (sha256 over the canonical JSON of the request rows, one per line)
and a mismatch with the manifest's ``prompt_set_sha256`` is refused. Rows tagged ``warmup`` carry
``meta.warmup`` and are excluded from summaries. :func:`template_check` compares a server's prompt-token count for
the manifest's probe conversation with the count prepared offline.

Pair with ``OpenAIChatWorkloadType(passthrough_meta=True)`` and a closed-loop load with enough slots for every
session that can be in flight (``closed:N`` for N concurrent sessions).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import random
import time
from typing import Any, Callable, Optional

from benchmaker.workloads.datasets import Workload


def canonical(row: dict) -> bytes:
    return json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def prompt_set_sha256(rows: list[dict]) -> str:
    h = hashlib.sha256()
    for row in rows:
        h.update(canonical(row))
        h.update(b"\n")
    return h.hexdigest()


def load_replay(path: str, *, verify: bool = True) -> tuple[dict, list[dict]]:
    """The manifest and request rows of a replay file; refuses a prompt set that no longer matches its hash."""
    with open(path) as f:
        manifest = json.loads(f.readline())
        rows = [json.loads(line) for line in f if line.strip()]
    if manifest.get("kind") != "manifest":
        raise ValueError(f"{path}: first line is not a manifest row")
    if verify:
        got = prompt_set_sha256(rows)
        want = manifest.get("prompt_set_sha256")
        if got != want:
            raise ValueError(f"{path}: prompt set hashes to {got}, manifest says {want}")
    return manifest, rows


def group_sessions(rows: list[dict]) -> list[list[dict]]:
    """Rows by session, in first-appearance order; each session's turns must be in order."""
    by: dict[Any, list[dict]] = {}
    for row in rows:
        by.setdefault(row["session"], []).append(row)
    out = list(by.values())
    for turns in out:
        order = [r.get("turn", i) for i, r in enumerate(turns)]
        if order != sorted(order):
            raise ValueError(f"session {turns[0]['session']!r}: turns out of order")
    return out


def think_seconds(row: dict, median_s: float, sigma: float) -> float:
    after = row.get("after") or {}
    if after.get("think_s") is not None:
        return float(after["think_s"])
    z = row.get("think_z")
    if z is None or median_s <= 0:
        return 0.0
    return median_s * math.exp(sigma * float(z))


def to_item(row: dict) -> dict:
    """The chat item for one row: recorded history, pinned output length, replay identifiers in meta."""
    if "messages" not in row:
        raise ValueError(f"row {row.get('id')!r} has no messages (token-id rows need "
                                 f"workload_type: {{type: openai-completions}}, #11)")
    meta = {
        "conversation_id": row["session"],
        "turn_index": row.get("turn"),
        "replay_id": row.get("id"),
        "model_label": row.get("model"),
        "expected_input_tokens": (row.get("expect") or {}).get("input_tokens"),
    }
    if row.get("tag") == "warmup":
        meta["warmup"] = True
    n = int(row["max_tokens"])
    return {"messages": row["messages"], "max_tokens": n, "min_tokens": n, "ignore_eos": True, "meta": meta}


class _Session:
    __slots__ = ("sid", "rows", "idx", "in_flight", "ready_at")

    def __init__(self, sid: Any, rows: list[dict], start: float):
        self.sid = sid
        self.rows = rows
        self.idx = 0
        self.in_flight = False
        self.ready_at = start

    @property
    def done(self) -> bool:
        return self.idx >= len(self.rows) and not self.in_flight


class ReplayWorkload(Workload):
    """Replay a prepared session file; see the module docstring for the format and the scheduling modes."""

    name = "session-replay"

    def __init__(self, path: str, *, concurrent_sessions: Optional[int] = None, session_rate: Optional[float] = None,
                 trace_scale: Optional[float] = None, think_median_s: Optional[float] = None, seed: int = 0,
                 model_label: Optional[str] = None, verify: bool = True,
                 clock: Callable[[], float] = time.monotonic):
        modes = [m for m in (concurrent_sessions, session_rate, trace_scale) if m is not None]
        if len(modes) != 1:
            raise ValueError("set exactly one of concurrent_sessions, session_rate, trace_scale")
        if concurrent_sessions is not None and concurrent_sessions < 1:
            raise ValueError("concurrent_sessions must be >= 1")
        if session_rate is not None and session_rate <= 0:
            raise ValueError("session_rate must be > 0")
        self.path = path
        self.manifest, rows = load_replay(path, verify=verify)
        self._groups = group_sessions(rows)
        if model_label is not None:  # one model's sessions of a multi-model file
            self._groups = [g for g in self._groups if g[0].get("model") == model_label]
            if not self._groups:
                raise ValueError(f"{path}: no sessions with model label {model_label!r}")
        think = self.manifest.get("think") or {}
        self.think_median_s = float(think_median_s if think_median_s is not None else think.get("median_s") or 0.0)
        self.think_sigma = float(think.get("sigma") or 0.0)
        self._concurrency = concurrent_sessions
        self._rate = session_rate
        self._trace_scale = trace_scale
        self._seed = seed
        self._clock = clock
        self._t0: Optional[float] = None
        self._starts: list[float] = []
        self._next = 0
        self._active: list[_Session] = []
        self._by_sid: dict[Any, _Session] = {}
        self._wakeup: Optional[asyncio.Event] = None

    def item_count(self) -> Optional[int]:
        return sum(len(g) for g in self._groups)

    def content_hash(self) -> Optional[str]:
        return self.manifest.get("prompt_set_sha256")

    def _plan_starts(self) -> None:
        if self._rate is not None:
            rng, at = random.Random(f"{self._seed}/replay/arrivals"), 0.0
            for _ in self._groups:
                at += rng.expovariate(self._rate)
                self._starts.append(at)
        elif self._trace_scale is not None:
            self._starts = [float(g[0].get("at_s", 0.0)) * self._trace_scale for g in self._groups]
        else:
            self._starts = [0.0] * len(self._groups)

    def _admit(self, now: float) -> None:
        while self._next < len(self._groups):
            if self._concurrency is not None and len(self._active) >= self._concurrency:
                return
            start = self._t0 + self._starts[self._next]
            if start > now:
                return
            rows = self._groups[self._next]
            sess = _Session(rows[0]["session"], rows, start)
            self._active.append(sess)
            self._by_sid[sess.sid] = sess
            self._next += 1

    def _reap(self) -> None:
        kept = []
        for s in self._active:
            if s.done:
                self._by_sid.pop(s.sid, None)
            else:
                kept.append(s)
        self._active = kept

    def _soonest(self, now: float) -> Optional[float]:
        waits = [s.ready_at - now for s in self._active if not s.in_flight and s.idx < len(s.rows)]
        if self._next < len(self._groups) and (self._concurrency is None or len(self._active) < self._concurrency):
            waits.append(self._t0 + self._starts[self._next] - now)
        return max(0.0, min(waits)) if waits else None

    async def next_item(self) -> Any:
        if self._wakeup is None:
            self._wakeup = asyncio.Event()
        if self._t0 is None:
            self._t0 = self._clock()
            self._plan_starts()
        while True:
            now = self._clock()
            self._reap()
            self._admit(now)
            ready = [s for s in self._active if not s.in_flight and s.idx < len(s.rows) and s.ready_at <= now]
            if ready:
                sess = min(ready, key=lambda s: s.ready_at)
                row = sess.rows[sess.idx]
                sess.idx += 1
                sess.in_flight = True
                return to_item(row)
            if not self._active and self._next >= len(self._groups):
                raise StopAsyncIteration
            self._wakeup.clear()
            try:
                await asyncio.wait_for(self._wakeup.wait(), timeout=self._soonest(now))
            except asyncio.TimeoutError:
                pass

    def notify_turn_complete(self, session_id: Any) -> None:
        """A session's in-flight turn finished: its next turn becomes eligible after the think time."""
        sess = self._by_sid.get(session_id)
        if sess is None or not sess.in_flight:
            return
        sess.in_flight = False
        if sess.idx < len(sess.rows):
            sess.ready_at = self._clock() + think_seconds(sess.rows[sess.idx], self.think_median_s, self.think_sigma)
        if self._wakeup is not None:
            self._wakeup.set()

    def note_complete_hook(self, request: Any, response: Any, sample: Any) -> Any:
        sid = (getattr(request, "meta", None) or {}).get("conversation_id")
        if sid is not None:
            self.notify_turn_complete(sid)
        return sample

    def completion_hook(self) -> Optional[Callable]:
        return self.note_complete_hook


def template_check(url: str, model: str, probe: dict, *, headers: Optional[dict] = None,
                   timeout_s: float = 120.0) -> tuple[Optional[int], int]:
    """(server prompt_tokens, prepared prompt_tokens) for the manifest's probe conversation.

    ``url`` is the chat-completions endpoint. A mismatch means the server renders a different chat template (or
    thinking mode) than the one the replay file was prepared with, so its prompts are not the prepared ones.
    """
    import urllib.request

    body = json.dumps({"model": model, "messages": probe["messages"], "max_tokens": 1, "temperature": 0}).encode()
    req = urllib.request.Request(url, body, {"Content-Type": "application/json", **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        usage = json.load(resp).get("usage") or {}
    return usage.get("prompt_tokens"), int(probe["prompt_tokens"])
