"""Multi-arm comparison runs: ABBA interleaving, per-rep bundles, same-job
enforcement (issue #3).

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
      launch_timeout_s: 600
      arms:
        rill:
          config: {workload_type: {url: "http://rill:8000/v1/chat/completions"}}
          env: {RILL_KV_CACHE_TIER: host}
          launch: "./scripts/launch-rill.sh"   # run before the arm's first rep
          relaunch: once                       # once | every_rep
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
capability snapshots, void verdict); the comparison table, TTFT parity check
and TTFT mix/latency decomposition go next to them in ``comparison.json`` /
``comparison.md``.
"""

from __future__ import annotations

import copy
import logging
import os
import random
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Optional

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

_logger = logging.getLogger(__name__)

ORDERS = ("abba", "rep-major", "random")


@dataclass
class ArmSpec:
    name: str
    config: dict = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)
    launch: Optional[str] = None
    relaunch: str = "once"
    labels: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, name: str, d: Any) -> "ArmSpec":
        if not isinstance(d, dict):
            raise TypeError(f"arms.{name} must be a mapping")
        env = {str(k): str(v) for k, v in (d.get("env") or {}).items()}
        relaunch = str(d.get("relaunch", "once") or "once").lower()
        if relaunch not in ("once", "every_rep", "every-rep"):
            raise ValueError(f"arms.{name}.relaunch must be once or every_rep, got {relaunch!r}")
        return cls(name=name, config=dict(d.get("config") or {}), env=env,
                   launch=d.get("launch"), relaunch=relaunch,
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
        raise ValueError("'arms.arms' must map arm name -> {config, env, launch, ...}")
    settings = ArmsSettings(
        reps=int(spec.get("reps", 1)),
        order=str(spec.get("order", "abba")).lower(),
        seed=int(spec.get("seed", 0)),
        warmup_reps=int(spec.get("warmup_reps", 0)),
        reference=spec.get("reference"),
        allow_mixed_jobs=bool(spec.get("allow_mixed_jobs", False)),
        metrics=[str(m) for m in (spec.get("metrics") or DEFAULT_METRICS)],
        launch_timeout_s=float(spec.get("launch_timeout_s", 600)),
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
        self.launched: set[str] = set()
        self.log_lines: list[str] = []

    # -- helpers ----------------------------------------------------------
    def _say(self, msg: str) -> None:
        self.log_lines.append(msg)
        if self.progress:
            _logger.info(msg)

    def arm_dir(self, arm: str) -> str:
        return os.path.join(self.root, "arms", arm)

    def rep_dir(self, arm: str, rep: int) -> str:
        return os.path.join(self.arm_dir(arm), f"rep-{rep}")

    def arm_config(self, arm: ArmSpec) -> dict:
        return deep_merge(self.base_cfg, arm.config)

    def _launch(self, arm: ArmSpec, rep: int) -> None:
        if not arm.launch:
            return
        if arm.relaunch == "once" and arm.name in self.launched:
            return
        env = {**os.environ, **arm.env, "BENCHMAKER_ARM": arm.name, "BENCHMAKER_REP": str(rep)}
        os.makedirs(self.arm_dir(arm.name), exist_ok=True)
        log_path = os.path.join(self.arm_dir(arm.name), f"launch-rep-{rep}.log")
        self._say(f"[arms] launching {arm.name}: {arm.launch}")
        with open(log_path, "w") as log:
            proc = subprocess.run(arm.launch, shell=True, env=env, stdout=log,
                                  stderr=subprocess.STDOUT, cwd=os.getcwd(),
                                  timeout=self.settings.launch_timeout_s)
        if proc.returncode != 0:
            raise RuntimeError(f"launch hook for arm {arm.name!r} failed (exit {proc.returncode}); see {log_path}")
        self.launched.add(arm.name)

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
        os.makedirs(self.root, exist_ok=True)
        with open(os.path.join(self.root, "plan.json"), "w") as f:
            json.dump({"order": s.order, "reps": s.reps, "warmup_reps": s.warmup_reps,
                       "seed": s.seed, "plan": [{"arm": a, "rep": r} for r, a in plan]},
                      f, indent=2)
        self._say(f"[arms] {len(names)} arms x {s.reps} reps, order={s.order}, "
                  f"warmup_reps={s.warmup_reps} -> {len(plan)} runs")

        arms_by_name = {a.name: a for a in s.arms}
        t_all = time.monotonic()
        for i, (rep, name) in enumerate(plan, 1):
            arm = arms_by_name[name]
            self._launch(arm, rep)
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
            runner = BenchRunner(bench_cfg)
            await runner.run()
            source_config = {**cfg_dict, "arm": name, "rep": rep, "warmup": warm}
            run_dir = runner.write_bundle(
                self.arm_dir(name), run_id=f"rep-{rep}",
                source_config=source_config, labels=labels, notes=self.notes,
                env_overrides=dict(arm.env) or None,
            )
            self.runs.append(RunRef.from_bundle(name, rep, run_dir, warmup=warm))
            m = runner.metrics
            self._say(f"[arms] done {name}/rep-{rep}: {m.summary().get('total_requests')} requests "
                      f"in {time.monotonic() - t0:.1f}s -> {run_dir}")

        result = self.compare()
        write_comparison(result, self.root)
        with open(os.path.join(self.root, "arms.log"), "w") as f:
            f.write("\n".join(self.log_lines) + "\n")
        self._say(f"[arms] all done in {time.monotonic() - t_all:.1f}s -> {self.root}")
        return result

    def compare(self) -> dict:
        return build_comparison(self.runs, self.settings)


    def exit_code(self, result: dict) -> int:
        if result.get("same_job_violations") and not self.settings.allow_mixed_jobs:
            return 2
        return 0


def build_comparison(runs: list[RunRef], settings: ArmsSettings) -> dict:
    """Table + same-job check + TTFT parity + decomposition for finished runs."""
    result = compare(runs, settings.metrics, settings.reference)
    result["order"] = settings.order
    result["reps"] = settings.reps
    result["warmup_reps"] = settings.warmup_reps
    result["same_job_violations"] = check_same_job(runs, allow_mixed=True)
    parity = ttft_parity(runs)
    result["ttft_parity"] = parity
    result["ttft_parity_warnings"] = ttft_parity_warnings(parity)
    # TTFT mix/latency decomposition vs the reference (#2), only when the
    # runs carry per-class groups.
    ref_runs = [r for r in runs if r.arm == result["reference"] and not r.warmup]
    decs = []
    for arm in result["arms"]:
        if arm == result["reference"]:
            continue
        a_runs = [r for r in runs if r.arm == arm and not r.warmup]
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
    parent, leaf = metric.rsplit(".", 1)
    for cls in shares:
        groups[cls] = {
            "share": sorted(shares[cls])[len(shares[cls]) // 2],
            parent: {leaf: (sorted(means[cls])[len(means[cls]) // 2] if cls in means else 0.0)},
        }
    return {"groups": {key: groups}}
