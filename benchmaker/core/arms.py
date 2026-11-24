"""Multi-arm comparison runs: ABBA interleaving, per-rep bundles, same-job
enforcement (issue #3), per-model endpoint maps (#8) and start/stop lifecycle
hooks with readiness and teardown checks (#9).

A *multi-arm run* executes the same benchmark config against several servers
("arms") several times each ("reps"), interleaved so that node drift affects
every arm equally. The YAML shape::

    arms:
      reps: 4
      order: abba            # abba | rep-major | random
      seed: 0                # for order: random
      warmup_reps: 1         # first rep(s) of every arm excluded from the table
      reference: rill        # ratios are against this arm (default: first)
      allow_mixed_jobs: false
      metrics: [ttft_p99, itl_p99.5, ...]
      launch_timeout_s: 600  # start hook timeout
      ready_timeout_s: 600   # ready_check polling budget
      stop_timeout_s: 120    # stop hook timeout
      stop_failure_voids: true  # a failed stop hook voids every rep of its launch (issue #14)
      arms:
        rill:
          config: {workload_type: {url: "http://rill:8000/v1/chat/completions"}}
          env: {RILL_KV_CACHE_TIER: host}
          start: "./scripts/start-rill.sh"    # run before the arm's first rep ("launch" is an alias)
          stop: "./scripts/stop-rill.sh"      # run after the arm's last rep, also on failure/interrupt
          ready_check: {urls: ["http://rill:8000/health"], status: 200, timeout_s: 300}
          post_stop_check: {run: "! nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q ."}
          relaunch: visit                   # once | every_rep | visit
          endpoints: {model_a: "http://rill:8401", model_b: "http://rill:8402"}   # #8
          labels: {image: "rill:0.9.1"}
        vllm:
          config: {workload_type: {url: "http://vllm:8000/v1/chat/completions"}}
          env: {VLLM_USE_V1: "1"}
    workload_type: {type: openai, stream: true, model: m, ...}   # shared base
    workload: {...}
    load: poisson:8
    duration: 60s

Each ``(arm, rep)`` writes its own bundle under
``<out-dir>/<run-id>/arms/<arm>/rep-<k>/`` with its own manifest (host/job id,
capability snapshots, void verdict) and the hook records that surrounded it
(``meta.hooks``, #9); the comparison table, TTFT parity check and TTFT
mix/latency decomposition go next to them in ``comparison.json`` /
``comparison.md``.

Lifecycle (issue #9): ``relaunch`` decides when a launch starts and stops —
``once`` starts before the arm's first rep and stops after its last (or at
the end of the run); ``every_rep`` wraps every rep; ``visit`` wraps every
*contiguous run* of the arm's reps in the plan, so in ABBA order the GPUs are
free again before the other arm starts. ``stop`` is always run for a started
launch — on failure or interrupt too — with ``stop_timeout_s``. A failed
``ready_check`` voids the reps that ran under that launch; a failed
``post_stop_check`` voids the launch's reps as well; both are recorded as
void reasons in the bundles and the comparison.
"""

from __future__ import annotations

import copy
import logging
import os
import random
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Optional

from benchmaker.core.capability import hook_violation
from benchmaker.core.compare import (
    DEFAULT_METRICS,
    RunRef,
    check_same_job,
    compare,
    decompose_ttft,
    render_comparison,
    render_decomposition,
    ttft_parity,
    ttft_parity_warnings,
)
from benchmaker.core.endpoints import normalize_endpoints

_logger = logging.getLogger(__name__)

ORDERS = ("abba", "rep-major", "random")
RELAUNCH_MODES = ("once", "every_rep", "visit")
DEFAULT_INTERVAL_S = 2.0
OUTPUT_TAIL_CHARS = 2048
HOOK_POLL_S = 0.25          # how often a running hook is checked for cancellation
HOOK_CANCEL_GRACE_S = 30.0  # how long teardown waits for an interrupted start hook to die


@dataclass
class ArmSpec:
    name: str
    config: dict = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)
    # ``start`` (alias ``launch``): shell command run before the arm's first
    # rep under the current launch; a non-zero exit aborts the run (#9).
    start: Optional[str] = None
    # ``stop``: shell command run after the launch's last rep, and on failure
    # or interrupt — so the GPUs are free before the next arm starts (#9).
    stop: Optional[str] = None
    relaunch: str = "once"
    # ``ready_check``: ``{urls: [...], status: 200, run: "...", timeout_s, interval_s}``
    # polled after the start hook returns; a failure voids the launch's reps.
    ready_check: dict = field(default_factory=dict)
    # ``post_stop_check``: same shape, run once after the stop hook.
    post_stop_check: dict = field(default_factory=dict)
    # ``endpoints``: per-model server map (#8), merged into the arm's config.
    endpoints: dict = field(default_factory=dict)
    labels: dict = field(default_factory=dict)

    @property
    def managed(self) -> bool:
        """Whether the runner starts/stops this arm's server itself (it has a
        start or stop hook). An unmanaged arm is served from outside."""
        return bool(self.start or self.stop)

    @classmethod
    def from_dict(cls, name: str, d: Any) -> "ArmSpec":
        if not isinstance(d, dict):
            raise TypeError(f"arms.{name} must be a mapping")
        env = {str(k): str(v) for k, v in (d.get("env") or {}).items()}
        relaunch = str(d.get("relaunch", "once") or "once").lower().replace("-", "_")
        if relaunch not in RELAUNCH_MODES:
            raise ValueError(f"arms.{name}.relaunch must be one of {RELAUNCH_MODES}, got {relaunch!r}")
        start = d.get("start") or d.get("launch")
        for key in ("ready_check", "post_stop_check"):
            if d.get(key) is not None and not isinstance(d[key], dict):
                raise TypeError(f"arms.{name}.{key} must be a mapping")
        return cls(name=name, config=dict(d.get("config") or {}), env=env,
                   start=start, stop=d.get("stop"), relaunch=relaunch,
                   ready_check=dict(d.get("ready_check") or {}),
                   post_stop_check=dict(d.get("post_stop_check") or {}),
                   endpoints=normalize_endpoints(d.get("endpoints")),
                   labels=dict(d.get("labels") or {}))


@dataclass
class ArmsSettings:
    reps: int = 1
    order: str = "abba"
    seed: int = 0
    warmup_reps: int = 0
    reference: Optional[str] = None
    allow_mixed_jobs: bool = False
    metrics: list[str] = field(default_factory=lambda: list(DEFAULT_METRICS))
    launch_timeout_s: float = 600.0
    ready_timeout_s: float = 600.0
    stop_timeout_s: float = 120.0
    check_timeout_s: float = 60.0
    stop_failure_voids: bool = True
    arms: list[ArmSpec] = field(default_factory=list)


def deep_merge(base: Any, over: Any) -> Any:
    """Dicts merge recursively; everything else (lists included) is replaced."""
    if isinstance(base, dict) and isinstance(over, dict):
        out = dict(base)
        for k, v in over.items():
            out[k] = deep_merge(base[k], v) if k in base else copy.deepcopy(v)
        return out
    return copy.deepcopy(over)


def parse_arms_config(cfg: dict) -> tuple[dict, ArmsSettings]:
    """Split a config into the shared base and the ``arms:`` settings."""
    spec = cfg.get("arms")
    if not isinstance(spec, dict) or not spec.get("arms"):
        raise ValueError("multi-arm config needs an 'arms:' block with named arms")
    arms_spec = spec["arms"]
    if not isinstance(arms_spec, dict) or not arms_spec:
        raise ValueError("'arms.arms' must map arm name -> {config, env, start, stop, ...}")
    settings = ArmsSettings(
        reps=int(spec.get("reps", 1)),
        order=str(spec.get("order", "abba")).lower(),
        seed=int(spec.get("seed", 0)),
        warmup_reps=int(spec.get("warmup_reps", 0)),
        reference=spec.get("reference"),
        allow_mixed_jobs=bool(spec.get("allow_mixed_jobs", False)),
        metrics=[str(m) for m in (spec.get("metrics") or DEFAULT_METRICS)],
        launch_timeout_s=float(spec.get("launch_timeout_s", 600)),
        ready_timeout_s=float(spec.get("ready_timeout_s", 600)),
        stop_timeout_s=float(spec.get("stop_timeout_s", 120)),
        check_timeout_s=float(spec.get("check_timeout_s", 60)),
        stop_failure_voids=bool(spec.get("stop_failure_voids", True)),
        arms=[ArmSpec.from_dict(name, d) for name, d in arms_spec.items()],
    )
    if settings.reps < 1:
        raise ValueError("arms.reps must be >= 1")
    if settings.order not in ORDERS:
        raise ValueError(f"arms.order must be one of {ORDERS}, got {settings.order!r}")
    if not (0 <= settings.warmup_reps < settings.reps):
        raise ValueError("arms.warmup_reps must be in [0, reps)")
    if settings.reference is not None and settings.reference not in arms_spec:
        raise ValueError(f"arms.reference {settings.reference!r} is not an arm")
    base = {k: v for k, v in cfg.items() if k != "arms"}
    return base, settings


def plan_order(arm_names: list[str], reps: int, order: str = "abba",
               seed: int = 0) -> list[tuple[int, str]]:
    """``[(rep, arm), ...]`` — the execution plan.

    * ``abba``: forward then backward over the arms, repeating — A B B A A B B A …
      — so consecutive reps of an arm are split across both orders.
    * ``rep-major``: every rep of arm 1, then every rep of arm 2, …
    * ``random``: seeded uniform shuffle of all ``(rep, arm)`` pairs.
    """
    if not arm_names:
        return []
    plan: list[tuple[int, str]] = []
    if order == "rep-major":
        for a in arm_names:
            plan.extend((r, a) for r in range(reps))
        return plan
    if order == "random":
        plan = [(r, a) for a in arm_names for r in range(reps)]
        random.Random(seed).shuffle(plan)
        return plan
    if order != "abba":
        raise ValueError(f"unknown order {order!r}")
    fwd = list(arm_names)
    bwd = list(reversed(arm_names))
    counts = {a: 0 for a in arm_names}
    while any(counts[a] < reps for a in arm_names):
        for a in fwd + bwd:
            if counts[a] < reps:
                plan.append((counts[a], a))
                counts[a] += 1
    return plan


# --------------------------------------------------------------------------- #
# Hooks and checks (issue #9)                                                  #
# --------------------------------------------------------------------------- #


def _kill_tree(proc: subprocess.Popen) -> None:
    """SIGTERM then SIGKILL the hook's whole process group (it got its own
    session), so a hung launcher cannot leave the run waiting."""
    try:
        pgid = os.getpgid(proc.pid)
    except Exception:  # noqa: BLE001 - already gone
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pgid, sig)
        except Exception:  # noqa: BLE001
            break
        try:
            proc.wait(timeout=10)
            return
        except subprocess.TimeoutExpired:
            continue


def _tail(path: str, limit: int = OUTPUT_TAIL_CHARS) -> str:
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - limit))
            return f.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def run_hook(command: str, *, arm: str, rep: Optional[int], env: dict[str, str],
             log_path: str, timeout_s: float,
             cancel: Optional[threading.Event] = None) -> dict:
    """Run a shell hook synchronously and return its record (issue #9):
    ``{command, exit_code, timed_out, cancelled, error, duration_s, log, output_tail}``.
    Never raises for a failing hook — the caller decides what a non-zero exit
    means. ``BENCHMAKER_ARM`` / ``BENCHMAKER_REP`` are exported to the hook
    along with the arm's ``env``.

    ``cancel`` (issue #14): the process is reaped in short waits so a set
    event — the run was interrupted while the hook ran — kills its whole
    process group promptly instead of after ``timeout_s``."""
    full_env = {**os.environ, **env, "BENCHMAKER_ARM": arm}
    if rep is not None:
        full_env["BENCHMAKER_REP"] = str(rep)
    rec: dict[str, Any] = {"command": command, "exit_code": None, "timed_out": False,
                           "cancelled": False, "error": None, "duration_s": None, "log": log_path}
    t0 = time.monotonic()
    try:
        with open(log_path, "w") as log:
            proc = subprocess.Popen(  # noqa: S602 - user-provided lab script
                command, shell=True, env=full_env, stdout=log,
                stderr=subprocess.STDOUT, cwd=os.getcwd(), start_new_session=True)
            try:
                deadline = t0 + timeout_s
                while True:
                    slice_s = max(deadline - time.monotonic(), 0.01)
                    try:
                        proc.wait(timeout=min(HOOK_POLL_S, slice_s))
                        rec["exit_code"] = proc.returncode
                        break
                    except subprocess.TimeoutExpired:
                        if cancel is not None and cancel.is_set():
                            rec["cancelled"] = True
                            rec["error"] = "cancelled (run interrupted)"
                            _kill_tree(proc)
                            break
                        if time.monotonic() >= deadline:
                            rec["timed_out"] = True
                            rec["error"] = f"timed out after {timeout_s:.0f}s"
                            _kill_tree(proc)
                            break
            except BaseException:
                _kill_tree(proc)
                raise
    except OSError as e:
        rec["error"] = f"{type(e).__name__}: {e}"
    rec["duration_s"] = time.monotonic() - t0
    rec["output_tail"] = _tail(log_path)
    return rec


def http_status(url: str, timeout_s: float) -> int:
    """Status code of ``GET url`` (0 when it could not connect)."""
    try:
        with urllib.request.urlopen(urllib.request.Request(url), timeout=timeout_s) as resp:  # noqa: S310
            return int(resp.status)
    except urllib.error.HTTPError as e:
        return int(e.code)
    except Exception:  # noqa: BLE001
        return 0


def normalize_check(spec: dict, *, default_timeout_s: float) -> dict:
    """``{url|urls, status, run, expect_exit, timeout_s, interval_s}`` with
    defaults filled in (``urls`` may be a single string); a check needs
    ``urls`` or ``run``."""
    if not isinstance(spec, dict):
        raise TypeError("check must be a mapping")
    urls = spec.get("urls") or ([spec["url"]] if spec.get("url") else [])
    if isinstance(urls, str):
        urls = [urls]
    urls = [str(u) for u in urls]
    run = spec.get("run")
    if not urls and not run:
        raise ValueError("check needs 'urls' (or 'url') and/or 'run'")
    timeout_s = spec.get("timeout_s", default_timeout_s)
    return {
        "urls": urls, "status": int(spec.get("status", 200)),
        "run": run, "expect_exit": int(spec.get("expect_exit", 0)),
        "timeout_s": float(timeout_s), "interval_s": float(spec.get("interval_s", DEFAULT_INTERVAL_S)),
    }


def _check_once(spec: dict, *, arm: str, rep: Optional[int], env: dict[str, str],
                log_path: str, timeout_s: float) -> dict:
    rec: dict[str, Any] = {"urls": [], "run": None, "ok": True, "error": None}
    for u in spec["urls"]:
        status = http_status(u, min(timeout_s, 30.0))
        ok = status == spec["status"]
        rec["urls"].append({"url": u, "status": status, "expected": spec["status"], "ok": ok})
        if not ok:
            rec["ok"] = False
    if spec["run"]:
        run_rec = run_hook(spec["run"], arm=arm, rep=rep, env=env, log_path=log_path,
                           timeout_s=timeout_s)
        rec["run"] = run_rec
        if run_rec["exit_code"] != spec["expect_exit"] or run_rec["error"]:
            rec["ok"] = False
    if not rec["ok"]:
        bad = [f"{x['url']} -> {x['status']} (expected {x['expected']})" for x in rec["urls"] if not x["ok"]]
        if rec["run"] and not rec["run"].get("exit_code") == spec["expect_exit"]:
            bad.append(f"{spec['run']!r} -> exit {rec['run'].get('exit_code')} {rec['run'].get('error') or ''}".rstrip())
        rec["error"] = "; ".join(bad) or "check failed"
    return rec


def evaluate_check(spec: dict, *, kind: str, arm: str, rep: Optional[int], env: dict[str, str],
                   log_path: str, timeout_s: float, poll: bool) -> dict:
    """Run a readiness / teardown check. With ``poll`` (readiness) retry
    every ``interval_s`` until it passes or the check's ``timeout_s``
    (``spec``'s own, else the ``timeout_s`` fallback) elapses; otherwise run
    once. The record carries every URL's status, the command's exit code and
    log tail, attempts and elapsed time."""
    budget = float(spec.get("timeout_s", timeout_s))
    t0 = time.monotonic()
    deadline = t0 + budget
    attempts = 0
    while True:
        attempts += 1
        rec = _check_once(spec, arm=arm, rep=rep, env=env, log_path=log_path,
                          timeout_s=max(deadline - time.monotonic(), 0.05))
        if rec["ok"] or not poll or time.monotonic() + spec["interval_s"] > deadline:
            break
        time.sleep(spec["interval_s"])
    rec.update({"kind": kind, "attempts": attempts, "elapsed_s": time.monotonic() - t0,
                "timeout_s": budget, "poll": poll})
    if not rec["ok"] and poll:
        rec["error"] = f"not ready after {attempts} attempt(s) / {rec['elapsed_s']:.0f}s: {rec['error']}"
    return rec


def stops_after(arm: ArmSpec, plan: list[tuple[int, str]], index: int) -> bool:
    """Whether the launch must be torn down after ``plan[index]`` under the
    arm's ``relaunch`` mode (starting is positional too — before every entry
    of an arm that is not live):

    * ``every_rep``: after every rep.
    * ``visit``: when the next entry is another arm (or there is none), so in
      ABBA order the GPUs are free before the other arm starts.
    * ``once``: after the arm's last entry in the plan, so the stop hook
      runs in plan order rather than only in the runner's ``finally`` (and
      a failing one voids that launch, issue #14); before then never — the
      launch stays up across the other arm's reps.
    """
    name = plan[index][1]
    if arm.relaunch == "every_rep":
        return True
    if arm.relaunch == "visit":
        nxt = plan[index + 1][1] if index + 1 < len(plan) else None
        return nxt != name
    return name not in {n for _, n in plan[index + 1:]}


def live_overlaps(plan: list[tuple[int, str]], arms: dict[str, ArmSpec]) -> list[tuple[str, str]]:
    """Pairs of arms whose launches are both up at the same time under the
    plan and the arms' ``relaunch`` modes (issue #14): with ``once`` (or no
    hooks at all) A's launch is still up while B's reps run, so two servers
    share the node's GPUs and each arm's numbers include the other's load.
    Repeated pairs are reported once; order is first-seen."""
    live: dict[str, int] = {}
    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, str]] = []
    for i, (rep, name) in enumerate(plan):
        arm = arms.get(name) or ArmSpec(name=name)
        if not arm.managed:
            continue  # no hooks: managed outside the runner, never "live" here
        for other in live:
            if other != name:
                key = (other, name) if other < name else (name, other)
                if key not in seen:
                    seen.add(key)
                    out.append(key)
        live[name] = rep
        if stops_after(arm, plan, i):
            live.pop(name, None)
    return out


# --------------------------------------------------------------------------- #
# Execution                                                                    #
# --------------------------------------------------------------------------- #


class ArmsRunner:
    """Run the plan, write one bundle per (arm, rep), then compare."""

    def __init__(self, cfg: dict, *, out_dir: str, run_id: Optional[str] = None,
                 labels: Optional[dict] = None, notes: str = "",
                 dotenv_path: Optional[str] = ".env", quiet: bool = False,
                 allow_mixed_jobs: Optional[bool] = None, progress: bool = True):
        self.base_cfg, self.settings = parse_arms_config(cfg)
        self.out_dir = out_dir
        self.run_id = run_id or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        self.labels = dict(labels or {})
        self.notes = notes
        self.dotenv_path = dotenv_path
        self.quiet = quiet
        self.progress = progress
        if allow_mixed_jobs is not None:
            self.settings.allow_mixed_jobs = bool(allow_mixed_jobs)
        self.root = os.path.join(out_dir, self.run_id)
        self.runs: list[RunRef] = []
        # Lifecycle bookkeeping (#9): arms whose launch is up (-> the latest
        # rep it served), the hook records of the current launch, and the
        # bundles written under it.
        self.live: dict[str, int] = {}
        self.launched: set[str] = set()          # arms that were ever started (compat)
        self.launch_hooks: dict[str, dict] = {}
        self.launch_bundles: dict[str, list[str]] = {}
        self.launch_ready_violation: dict[str, Optional[dict]] = {}  # per launch (#14)
        self._hooks_in_flight: dict[str, tuple[threading.Event, threading.Event]] = {}  # (cancel, done)
        self.log_lines: list[str] = []

    # -- helpers ----------------------------------------------------------
    def _say(self, msg: str) -> None:
        self.log_lines.append(msg)
        if self.progress:
            _logger.info(msg)

    def _warn(self, msg: str) -> None:
        self.log_lines.append(msg)
        _logger.warning(msg)

    def arm_dir(self, arm: str) -> str:
        return os.path.join(self.root, "arms", arm)

    def rep_dir(self, arm: str, rep: int) -> str:
        return os.path.join(self.arm_dir(arm), f"rep-{rep}")

    def arm_config(self, arm: ArmSpec) -> dict:
        cfg = deep_merge(self.base_cfg, arm.config)
        if arm.endpoints:
            cfg["endpoints"] = deep_merge(cfg.get("endpoints") or {}, arm.endpoints)
        return cfg

    def _hook_log(self, arm: ArmSpec, kind: str, rep: Optional[int]) -> str:
        os.makedirs(self.arm_dir(arm.name), exist_ok=True)
        suffix = f"rep-{rep}" if rep is not None else "final"
        return os.path.join(self.arm_dir(arm.name), f"{kind}-{suffix}.log")

    # -- lifecycle (#9) -----------------------------------------------------
    async def _start(self, arm: ArmSpec, rep: int) -> list[dict]:
        """Bring the arm up for ``rep`` if it is not live. Returns void
        violations: a failed ``ready_check`` voids *every* rep of this launch
        (issue #14), so it is returned again for each later rep that reuses
        the launch. A start hook that exits non-zero is fatal: the launch is
        torn down and the run aborted. Hooks and checks run in a worker
        thread so the event loop stays alive."""
        import asyncio

        if arm.name in self.live:
            v = self.launch_ready_violation.get(arm.name)
            return [{**v, "rep": rep}] if v else []
        hooks: dict[str, Any] = {}
        s = self.settings
        if arm.start:
            # The launch counts as live *before* the hook runs (issue #14):
            # Ctrl-C while it runs lands in run_async's finally, which
            # cancels the hook (_hooks_in_flight) and runs the stop hook.
            self.live[arm.name] = rep
            self.launch_bundles[arm.name] = []
            cancel, done = threading.Event(), threading.Event()
            self._hooks_in_flight[arm.name] = (cancel, done)
            log_path = self._hook_log(arm, "launch", rep)

            def _run() -> dict:
                try:
                    return run_hook(arm.start, arm=arm.name, rep=rep, env=arm.env, log_path=log_path,
                                    timeout_s=s.launch_timeout_s, cancel=cancel)
                finally:
                    done.set()

            try:
                rec = await asyncio.to_thread(_run)
            except BaseException as e:
                # On CancelledError the thread keeps running: leave the entry
                # so teardown can cancel it; otherwise it has finished.
                if not isinstance(e, asyncio.CancelledError):
                    self._hooks_in_flight.pop(arm.name, None)
                raise
            self._hooks_in_flight.pop(arm.name, None)
            hooks["start"] = rec
            self._say(f"[arms] starting {arm.name}: {arm.start} -> exit {rec['exit_code']} "
                      f"in {rec['duration_s']:.1f}s ({rec['log']})")
            if rec["exit_code"] != 0:
                self._say(f"[arms] start hook for {arm.name} failed; running stop hook before aborting")
                self.launch_hooks[arm.name] = hooks
                await self._stop(arm, rep)
                raise RuntimeError(f"start hook for arm {arm.name!r} failed (exit {rec['exit_code']}); "
                                   f"see {rec['log']}")
        self.live[arm.name] = rep
        self.launched.add(arm.name)
        self.launch_bundles.setdefault(arm.name, [])
        violations: list[dict] = []
        self.launch_ready_violation[arm.name] = None
        if arm.ready_check:
            spec = normalize_check(arm.ready_check, default_timeout_s=s.ready_timeout_s)
            rec = await asyncio.to_thread(
                evaluate_check, spec, kind="ready_check", arm=arm.name, rep=rep, env=arm.env,
                log_path=self._hook_log(arm, "ready-check", rep),
                timeout_s=s.ready_timeout_s, poll=True)
            hooks["ready_check"] = rec
            if rec["ok"]:
                self._say(f"[arms] {arm.name} ready after {rec['elapsed_s']:.1f}s")
            else:
                self._warn(f"[arms] {arm.name} NOT ready: {rec['error']} -> every rep of this "
                           f"launch is void (issue #14)")
                v = hook_violation("ready_check", rec["error"], arm=arm.name, rep=rep, launch_rep=rep)
                violations.append(v)
                self.launch_ready_violation[arm.name] = v
        self.launch_hooks[arm.name] = hooks
        return violations

    async def _cancel_in_flight(self, name: str) -> None:
        """Ask a still-running start hook to stop and wait (bounded) for its
        thread, so the stop hook does not race the launcher it precludes."""
        import asyncio

        entry = self._hooks_in_flight.get(name)
        if entry is None:
            return
        cancel, done = entry
        if not done.is_set():
            cancel.set()
            self._warn(f"[arms] interrupting {name}'s start hook (still running; issue #14)")
            await asyncio.to_thread(done.wait, HOOK_CANCEL_GRACE_S)
        if done.is_set():
            self._hooks_in_flight.pop(name, None)
        else:
            self._warn(f"[arms] start hook for {name} did not stop within {HOOK_CANCEL_GRACE_S:.0f}s")

    async def _stop(self, arm: ArmSpec, rep: int, *, final: bool = False) -> list[dict]:
        """Tear the launch down: stop hook (with ``stop_timeout_s``), then
        ``post_stop_check``. Returns void violations, which are applied to
        every bundle written under this launch. A stop hook that fails (or
        times out) is one of them unless ``arms.stop_failure_voids`` is off
        (issue #14)."""
        import asyncio

        if arm.name not in self.live:
            return []
        s = self.settings
        await self._cancel_in_flight(arm.name)
        hooks: dict[str, Any] = {}
        violations: list[dict] = []
        if arm.stop:
            rec = await asyncio.to_thread(
                run_hook, arm.stop, arm=arm.name, rep=rep, env=arm.env,
                log_path=self._hook_log(arm, "stop", None if final else rep),
                timeout_s=s.stop_timeout_s)
            hooks["stop"] = rec
            self._say(f"[arms] stopping {arm.name}: {arm.stop} -> exit {rec['exit_code']} "
                      f"in {rec['duration_s']:.1f}s ({rec['log']})")
            if rec["exit_code"] != 0:
                self._warn(f"[arms] stop hook for {arm.name} failed: exit {rec['exit_code']} "
                           f"{rec['error'] or ''} (see {rec['log']})")
                hooks["stop"]["ok"] = False
                if s.stop_failure_voids:
                    self._warn(f"[arms] voiding {len(self.launch_bundles.get(arm.name) or [])} rep(s) "
                               f"of {arm.name}'s launch (arms.stop_failure_voids)")
                    violations.append(hook_violation(
                        "stop", f"exit {rec['exit_code']} {rec['error'] or ''}".strip(),
                        arm=arm.name, rep=rep))
        if arm.post_stop_check:
            spec = normalize_check(arm.post_stop_check, default_timeout_s=s.check_timeout_s)
            rec = await asyncio.to_thread(
                evaluate_check, spec, kind="post_stop_check", arm=arm.name, rep=rep, env=arm.env,
                log_path=self._hook_log(arm, "post-stop-check", None if final else rep),
                timeout_s=s.check_timeout_s, poll=False)
            hooks["post_stop_check"] = rec
            if rec["ok"]:
                self._say(f"[arms] {arm.name} torn down cleanly ({rec['elapsed_s']:.1f}s)")
            else:
                self._warn(f"[arms] {arm.name} post_stop_check failed: {rec['error']} -> "
                           f"voiding {len(self.launch_bundles.get(arm.name) or [])} rep(s)")
                violations.append(hook_violation("post_stop_check", rec["error"], arm=arm.name, rep=rep))
        self._apply_to_launch(arm, hooks, violations)
        self.live.pop(arm.name, None)
        self.launch_hooks.pop(arm.name, None)
        self.launch_bundles.pop(arm.name, None)
        self.launch_ready_violation.pop(arm.name, None)
        return violations

    def _apply_to_launch(self, arm: ArmSpec, hooks: dict, violations: list[dict]) -> None:
        """Merge stop-side hook records and void reasons into the bundles the
        launch produced (they were written before the teardown ran)."""
        from benchmaker.io.bundle import patch_bundle

        for run_dir in self.launch_bundles.get(arm.name) or []:
            patch_bundle(run_dir, hooks=hooks or None, void_violations=violations or None)
        # The RunRefs were built before the teardown ran: reload them so the
        # comparison sees the amended void verdict.
        for i, ref in enumerate(self.runs):
            if ref.bundle_dir in (self.launch_bundles.get(arm.name) or []):
                self.runs[i] = RunRef.from_bundle(ref.arm, ref.rep, ref.bundle_dir, warmup=ref.warmup)

    async def _stop_all(self) -> None:
        """Tear every live launch down — the ``finally`` of the run, so
        failures and interrupts never leave a server behind. Start hooks
        still running are cancelled first (issue #14) so they cannot restart
        a server after its stop hook ran."""
        arms = {a.name: a for a in self.settings.arms}
        for cancel, done in list(self._hooks_in_flight.values()):
            if not done.is_set():
                cancel.set()
        for name in list(self.live):
            arm = arms.get(name)
            if arm is None:
                continue
            try:
                await self._stop(arm, self.live[name], final=True)
            except Exception as e:  # noqa: BLE001 - teardown must not mask the original error
                self._warn(f"[arms] stop hook for {name} raised {type(e).__name__}: {e}")

    # -- main -------------------------------------------------------------
    def run(self) -> dict:
        """Sync entry point (own event loop). Use :meth:`run_async` inside one."""
        import asyncio
        return asyncio.run(self.run_async())

    async def run_async(self) -> dict:
        import json

        from benchmaker.config import build_config
        from benchmaker.core.runner import BenchRunner

        s = self.settings
        names = [a.name for a in s.arms]
        plan = plan_order(names, s.reps, s.order, s.seed)
        arms_by_name = {a.name: a for a in s.arms}
        overlaps = live_overlaps(plan, arms_by_name)
        for a, b in overlaps:
            self._warn(f"[arms] arms {a!r} and {b!r} will both be live at the same time "
                       f"(relaunch {arms_by_name[a].relaunch}/{arms_by_name[b].relaunch}): if they share "
                       f"GPUs, use relaunch: visit or every_rep so each is torn down before the "
                       f"other starts (issue #14)")
        os.makedirs(self.root, exist_ok=True)
        with open(os.path.join(self.root, "plan.json"), "w") as f:
            json.dump({"order": s.order, "reps": s.reps, "warmup_reps": s.warmup_reps,
                       "seed": s.seed, "plan": [{"arm": a, "rep": r} for r, a in plan],
                       "stops_after": [{"arm": a, "rep": r, "stop": stops_after(arms_by_name[a], plan, i)}
                                       for i, (r, a) in enumerate(plan)],
                       "overlaps": [{"arm": a, "other": b} for a, b in overlaps]},
                      f, indent=2)
        self._say(f"[arms] {len(names)} arms x {s.reps} reps, order={s.order}, "
                  f"warmup_reps={s.warmup_reps} -> {len(plan)} runs")

        t_all = time.monotonic()
        try:
            for i, (rep, name) in enumerate(plan, 1):
                arm = arms_by_name[name]
                violations = await self._start(arm, rep)
                self.live[name] = rep  # the rep a forced teardown will be attributed to
                cfg_dict = self.arm_config(arm)
                warm = rep < s.warmup_reps
                labels = {**self.labels, **arm.labels, "arm": name, "rep": rep}
                if warm:
                    labels["warmup"] = "true"
                t0 = time.monotonic()
                self._say(f"[arms] ({i}/{len(plan)}) arm={name} rep={rep}{' warmup' if warm else ''}")
                bench_cfg = build_config(cfg_dict, dotenv_path=self.dotenv_path)
                if self.quiet:
                    bench_cfg.progress_every_s = 0.0
                bench_cfg.extra_void.extend(violations)
                runner = BenchRunner(bench_cfg)
                await runner.run()
                source_config = {**cfg_dict, "arm": name, "rep": rep, "warmup": warm}
                run_dir = runner.write_bundle(
                    self.arm_dir(name), run_id=f"rep-{rep}",
                    source_config=source_config, labels=labels, notes=self.notes,
                    env_overrides=dict(arm.env) or None,
                    hooks=dict(self.launch_hooks.get(name) or {}) or None,
                )
                self.launch_bundles.setdefault(name, []).append(run_dir)
                self.runs.append(RunRef.from_bundle(name, rep, run_dir, warmup=warm))
                m = runner.metrics
                self._say(f"[arms] done {name}/rep-{rep}: {m.summary().get('total_requests')} requests "
                          f"in {time.monotonic() - t0:.1f}s -> {run_dir}")
                if stops_after(arm, plan, i - 1):
                    await self._stop(arm, rep)
        finally:
            await self._stop_all()

        result = self.compare()
        write_comparison(result, self.root)
        counts = result["signoff"]
        self._say(f"[arms] {counts['passed']} passed, {counts['void']} void, "
                  f"{counts['failed']} failed, {counts['warmup']} warm-up "
                  f"in {time.monotonic() - t_all:.1f}s -> {self.root}")
        with open(os.path.join(self.root, "arms.log"), "w") as f:
            f.write("\n".join(self.log_lines) + "\n")
        return result

    def compare(self) -> dict:
        return build_comparison(self.runs, self.settings)

    def exit_code(self, result: dict) -> int:
        if result.get("same_job_violations") and not self.settings.allow_mixed_jobs:
            return 2
        if not result["signoff"]["passed"]:
            return 1
        return 0


def build_comparison(runs: list[RunRef], settings: ArmsSettings) -> dict:
    """Table + same-job check + TTFT parity + decomposition for finished runs."""
    result = compare(runs, settings.metrics, settings.reference)
    result["order"] = settings.order
    result["reps"] = settings.reps
    result["warmup_reps"] = settings.warmup_reps
    counts = {"passed": 0, "void": 0, "failed": 0, "warmup": 0}
    for r in runs:
        if r.warmup:
            counts["warmup"] += 1
        elif r.is_void():
            counts["void"] += 1
        elif r.summary.get("success") == 0:
            counts["failed"] += 1
        else:
            counts["passed"] += 1
    result["signoff"] = counts
    result["same_job_violations"] = check_same_job(runs, allow_mixed=True)
    parity = ttft_parity(runs)
    result["ttft_parity"] = parity
    result["ttft_parity_warnings"] = ttft_parity_warnings(parity)
    # TTFT mix/latency decomposition vs the reference (#2), only when the
    # runs carry per-class groups. Void reps are excluded like warm-up ones
    # (issue #17): a rep voided for a failed readiness check would otherwise
    # weigh on the medians the decomposition attributes to the arm.
    ref_runs = [r for r in runs if r.arm == result["reference"] and not r.warmup and not r.is_void()]
    decs = []
    for arm in result["arms"]:
        if arm == result["reference"]:
            continue
        a_runs = [r for r in runs if r.arm == arm and not r.warmup and not r.is_void()]
        if not a_runs or not ref_runs:
            continue
        try:
            ref_summary = _median_summary([r.summary for r in ref_runs])
            arm_summary = _median_summary([r.summary for r in a_runs])
            dec = decompose_ttft(ref_summary, arm_summary)
        except Exception as e:  # noqa: BLE001
            _logger.debug("decomposition %s vs %s skipped: %s", result["reference"], arm, e)
            continue
        if not dec["per_class"]:
            continue
        dec["arm"] = arm
        decs.append(dec)
    result["decomposition"] = decs
    return result


def write_comparison(result: dict, root: str) -> None:
    """comparison.json / comparison.md / decomposition-<arm>.md under ``root``."""
    import json
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, "comparison.json"), "w") as f:
        json.dump(result, f, indent=2, default=str)
    with open(os.path.join(root, "comparison.md"), "w") as f:
        f.write(render_comparison(result) + "\n")
    for dec in result.get("decomposition", []):
        with open(os.path.join(root, f"decomposition-{dec['arm']}.md"), "w") as f:
            f.write(render_decomposition(dec, result["reference"], dec["arm"]) + "\n")


def discover_arm_dirs(root: str) -> dict[str, list[str]]:
    """``{arm: [bundle_dir, ...]}`` from an ``arms/`` layout produced by
    :class:`ArmsRunner` (``<root>/arms/<arm>/rep-<k>/``)."""
    from benchmaker.io.bundle import is_bundle_dir
    arms_root = os.path.join(root, "arms")
    out: dict[str, list[str]] = {}
    if not os.path.isdir(arms_root):
        return out
    for arm in sorted(os.listdir(arms_root)):
        d = os.path.join(arms_root, arm)
        if not os.path.isdir(d):
            continue
        reps = sorted((os.path.join(d, r) for r in os.listdir(d)
                       if r.startswith("rep-") and is_bundle_dir(os.path.join(d, r))),
                      key=lambda p: _rep_index(os.path.basename(p)))
        if reps:
            out[arm] = reps
    return out


def runs_from_arm_dirs(arm_dirs: dict[str, list[str]], warmup_reps: int = 0) -> list[RunRef]:
    runs = []
    for arm, dirs in arm_dirs.items():
        for d in sorted(dirs, key=lambda p: _rep_index(os.path.basename(p))):
            rep = _rep_index(os.path.basename(d))
            runs.append(RunRef.from_bundle(arm, rep, d, warmup=rep < warmup_reps))
    return runs


def _rep_index(name: str, default: int = 0) -> int:
    """``rep-<k>`` -> ``k``."""
    try:
        return int(name.split("-", 1)[1])
    except (IndexError, ValueError):
        return default


def _median_summary(summaries: list[dict], metric: str = "workload_metrics.ttft_s.mean",
                    key: str = "cache_class") -> dict:
    """A synthetic summary whose class shares/means are the per-rep medians."""
    if len(summaries) == 1:
        return summaries[0]
    shares: dict[str, list[float]] = {}
    means: dict[str, list[float]] = {}
    for s in summaries:
        for cls, sub in ((s.get("groups") or {}).get(key) or {}).items():
            shares.setdefault(cls, []).append(float(sub.get("share", 0.0)))
            m: Any = sub
            for part in metric.split("."):
                m = m.get(part) if isinstance(m, dict) else None
            if m is not None:
                means.setdefault(cls, []).append(float(m))
    groups = {}
    parts = metric.split(".")
    for cls in shares:
        med = sorted(means[cls])[len(means[cls]) // 2] if cls in means else 0.0
        # nest the metric path (``workload_metrics.ttft_s.mean``) so that
        # ``dotted_get`` finds it — a literal dotted key was never found, and
        # the decomposition was silently skipped whenever an arm had >= 2 reps
        leaf: dict = {parts[-1]: med}
        for part in reversed(parts[:-1]):
            leaf = {part: leaf}
        groups[cls] = {"share": sorted(shares[cls])[len(shares[cls]) // 2], **leaf}
    return {"groups": {key: groups}}
