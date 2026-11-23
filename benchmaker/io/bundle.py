"""Per-run output bundle.

Each benchmark run writes a directory:

    <out_dir>/<run_id>/
        meta.json              # run identifiers, timestamps, resolved config
        manifest.json          # what was benchmarked: config/prompt-set hashes,
                               # model+revision, host/job identity, env (#4)
        summary.json           # aggregated metrics
        samples.jsonl          # one record per request
        monitors.jsonl         # one record per monitor tick
        capability.start.json  # server self-description before the run (#4, optional)
        capability.end.json    # ... and after it (optional)
        capability.<model>.start.json / .end.json
                               # per-endpoint snapshots when an arm fans out to
                               # several servers (#8, optional)
        meta.json#hooks        # start/stop hook exit codes and log paths (#9, optional)

The bundle is the unit of result interchange. `benchmaker collect` walks
one or more such directories and pivots the summaries into a table.
"""

from __future__ import annotations

import json
import os
import platform
import socket
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from benchmaker.core.metrics import MetricsAggregator, _safe_meta


BUNDLE_VERSION = 2
META_FILENAME = "meta.json"
MANIFEST_FILENAME = "manifest.json"
SUMMARY_FILENAME = "summary.json"
SAMPLES_FILENAME = "samples.jsonl"
MONITORS_FILENAME = "monitors.jsonl"
CAPABILITY_START_FILENAME = "capability.start.json"
CAPABILITY_END_FILENAME = "capability.end.json"
_CAPABILITY_PREFIX = "capability."


def _safe_segment(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_.," else "_" for c in name) or "endpoint"


def capability_filenames(name: str) -> tuple[str, str]:
    """``(start, end)`` file names for endpoint ``name`` (#8)."""
    safe = _safe_segment(name)
    return f"{_CAPABILITY_PREFIX}{safe}.start.json", f"{_CAPABILITY_PREFIX}{safe}.end.json"


def default_run_id(prefix: Optional[str] = None) -> str:
    """Filesystem-safe UTC timestamp, e.g. `20260526T142233Z` or `prefix-20260526T142233Z`."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}-{ts}" if prefix else ts


@dataclass
class RunMeta:
    """Identifiers and provenance written next to the results."""

    run_id: str
    started_at: str                            # ISO-8601 UTC
    ended_at: str                              # ISO-8601 UTC
    wall_time_s: float
    workload_type: str                         # e.g. "http", "openai-chat"
    workload: str                              # e.g. "static", "jsonl"
    hostname: str = field(default_factory=socket.gethostname)
    python_version: str = field(default_factory=platform.python_version)
    bench_maker_version: str = ""              # filled in by write_bundle
    bundle_version: int = BUNDLE_VERSION
    source_config: dict = field(default_factory=dict)
    labels: dict = field(default_factory=dict)
    notes: str = ""
    # Void verdict (issue #4): None when no void rules were configured.
    void: Optional[bool] = None
    void_violations: list = field(default_factory=list)
    # Arm lifecycle hooks that ran around this rep (issue #9): exit codes,
    # durations, log paths and readiness/teardown check results.
    hooks: Optional[dict] = None

    def to_dict(self) -> dict:
        d = {
            "run_id": self.run_id,
            "bundle_version": self.bundle_version,
            "bench_maker_version": self.bench_maker_version,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "wall_time_s": self.wall_time_s,
            "hostname": self.hostname,
            "python_version": self.python_version,
            "workload_type": self.workload_type,
            "workload": self.workload,
            "labels": self.labels,
            "notes": self.notes,
            "void": self.void,
            "void_violations": self.void_violations,
            "source_config": _safe_meta(self.source_config),
        }
        if self.hooks is not None:
            d["hooks"] = _safe_meta(self.hooks)
        return d


def write_bundle(
    out_dir: str,
    metrics: MetricsAggregator,
    *,
    workload_type_name: str,
    workload_name: str,
    run_id: Optional[str] = None,
    source_config: Optional[dict] = None,
    labels: Optional[dict] = None,
    notes: str = "",
    started_wall: Optional[float] = None,
    ended_wall: Optional[float] = None,
    manifest: Optional[dict] = None,
    capability: Optional[dict[str, Any]] = None,
    summary: Optional[dict] = None,
    hooks: Optional[dict[str, Any]] = None,
) -> str:
    """Write the bundle for a finished run. Returns the absolute run directory path.

    ``summary`` overrides ``metrics.summary()`` — the runner passes the
    summary it already enriched (capability, void verdict, phase windows).

    `out_dir` is treated as the parent: the actual run lives in `<out_dir>/<run_id>/`.
    If `out_dir` itself already looks like a run dir (already has `<run_id>` baked in),
    pass `run_id=""` to skip the extra nesting.

    ``manifest`` (see :mod:`benchmaker.io.manifest`) and ``capability``
    (``{"start": ..., "end": ...}`` snapshots, or ``{"endpoints": {model:
    {"start", "end"}}}`` with an endpoint map, #8) are written as separate
    files when given (#4). The void verdict, if any, is copied into
    ``meta.json`` so `collect` can flag void runs without opening
    summary.json. ``hooks`` (#9) lands in ``meta.json`` as ``hooks``.
    """
    from benchmaker import __version__ as bm_version

    if metrics.end_time is None:
        metrics.finalize()

    run_id = run_id if run_id is not None else default_run_id()
    target = os.path.join(out_dir, run_id) if run_id else out_dir
    os.makedirs(target, exist_ok=True)

    duration = metrics.end_time - metrics.start_time if metrics.end_time else 0.0
    end_wall = ended_wall if ended_wall is not None else (metrics.end_wall or time.time())
    start_wall = started_wall if started_wall is not None else (metrics.start_wall or (end_wall - duration))

    summary = summary if summary is not None else metrics.summary()
    void = summary.get("void")

    meta = RunMeta(
        run_id=run_id or os.path.basename(os.path.abspath(target)),
        started_at=_iso(start_wall),
        ended_at=_iso(end_wall),
        wall_time_s=duration,
        workload_type=workload_type_name,
        workload=workload_name,
        bench_maker_version=bm_version,
        source_config=source_config or {},
        labels=labels or {},
        notes=notes,
        void=bool(void["void"]) if void else None,
        void_violations=list(void.get("violations", [])) if void else [],
        hooks=hooks,
    )

    with open(os.path.join(target, META_FILENAME), "w") as f:
        json.dump(meta.to_dict(), f, indent=2, default=_json_default)

    if manifest is not None:
        if not manifest.get("run_id"):
            manifest["run_id"] = meta.run_id
        if not manifest.get("started_at"):
            manifest["started_at"] = meta.started_at
        with open(os.path.join(target, MANIFEST_FILENAME), "w") as f:
            json.dump(manifest, f, indent=2, default=_json_default)

    with open(os.path.join(target, SUMMARY_FILENAME), "w") as f:
        json.dump(summary, f, indent=2, default=_json_default)

    for at, name in (("start", CAPABILITY_START_FILENAME), ("end", CAPABILITY_END_FILENAME)):
        snap = (capability or {}).get(at)
        if snap is not None:
            with open(os.path.join(target, name), "w") as f:
                json.dump(snap, f, indent=2, default=_json_default)
    for ep_name, snaps in ((capability or {}).get("endpoints") or {}).items():
        start_name, end_name = capability_filenames(ep_name)
        for at, name in (("start", start_name), ("end", end_name)):
            snap = (snaps or {}).get(at)
            if snap is not None:
                with open(os.path.join(target, name), "w") as f:
                    json.dump(snap, f, indent=2, default=_json_default)

    metrics.dump_samples_jsonl(os.path.join(target, SAMPLES_FILENAME))
    if metrics.monitor_samples:
        metrics.dump_monitor_jsonl(os.path.join(target, MONITORS_FILENAME))

    return os.path.abspath(target)


def patch_bundle(run_dir: str, *, hooks: Optional[dict[str, Any]] = None,
                 void_violations: Optional[list] = None) -> None:
    """Amend a written bundle in place (issue #9).

    ``hooks`` is merged into ``meta.hooks`` (key-wise, so a rep's ``stop``
    record joins its ``start`` record). ``void_violations`` are appended to
    both ``summary.void`` and ``meta.void/void_violations`` — that is how a
    failed readiness or teardown check makes a rep void after the fact, since
    `collect`/`compare` read the verdict from ``meta.json``.
    """
    from benchmaker.core.capability import merge_void_verdicts

    meta_path = os.path.join(run_dir, META_FILENAME)
    summary_path = os.path.join(run_dir, SUMMARY_FILENAME)
    with open(meta_path) as f:
        meta = json.load(f)
    with open(summary_path) as f:
        summary = json.load(f)
    if hooks:
        merged = dict(meta.get("hooks") or {})
        for k, v in hooks.items():
            if isinstance(v, dict) and isinstance(merged.get(k), dict):
                merged[k] = {**merged[k], **v}
            else:
                merged[k] = v
        meta["hooks"] = merged
    if void_violations:
        prev = summary.get("void") or {"void": False, "violations": [], "checks": []}
        summary["void"] = merge_void_verdicts([prev, {"void": True, "violations": list(void_violations), "checks": []}])
        meta["void"] = bool(summary["void"]["void"])
        meta["void_violations"] = list(summary["void"]["violations"])
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=_json_default)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=_json_default)


def read_bundle(run_dir: str) -> dict:
    """Read a run directory and return
    ``{meta, summary, manifest, capability, samples_path, monitors_path, dir}``.

    Samples and monitor ticks are returned as file paths rather than loaded, so
    a `collect` over many runs stays cheap. Use `iter_jsonl` to stream rows.
    ``manifest`` and ``capability`` are ``None`` for bundles written before #4.
    """
    meta_path = os.path.join(run_dir, META_FILENAME)
    summary_path = os.path.join(run_dir, SUMMARY_FILENAME)
    if not (os.path.isfile(meta_path) and os.path.isfile(summary_path)):
        raise FileNotFoundError(
            f"{run_dir!r} is not a run bundle (missing meta.json or summary.json)"
        )
    with open(meta_path) as f:
        meta = json.load(f)
    with open(summary_path) as f:
        summary = json.load(f)
    samples = os.path.join(run_dir, SAMPLES_FILENAME)
    monitors = os.path.join(run_dir, MONITORS_FILENAME)
    capability = {
        at: _read_json_if_exists(os.path.join(run_dir, name))
        for at, name in (("start", CAPABILITY_START_FILENAME), ("end", CAPABILITY_END_FILENAME))
    }
    # Per-endpoint snapshots (#8): capability.<model>.{start,end}.json.
    per: dict[str, dict] = {}
    if os.path.isdir(run_dir):
        for fn in sorted(os.listdir(run_dir)):
            if not fn.startswith(_CAPABILITY_PREFIX) or not fn.endswith(".json"):
                continue
            stem = fn[len(_CAPABILITY_PREFIX):-len(".json")]
            ep_name, _, at = stem.rpartition(".")
            if at not in ("start", "end") or not ep_name:
                continue
            per.setdefault(ep_name, {"start": None, "end": None})[at] = _read_json_if_exists(
                os.path.join(run_dir, fn))
    if per:
        capability["endpoints"] = per
    return {
        "meta": meta,
        "summary": summary,
        "manifest": _read_json_if_exists(os.path.join(run_dir, MANIFEST_FILENAME)),
        "capability": capability,
        "samples_path": samples if os.path.isfile(samples) else None,
        "monitors_path": monitors if os.path.isfile(monitors) else None,
        "dir": os.path.abspath(run_dir),
    }


def is_bundle_dir(path: str) -> bool:
    return (
        os.path.isdir(path)
        and os.path.isfile(os.path.join(path, META_FILENAME))
        and os.path.isfile(os.path.join(path, SUMMARY_FILENAME))
    )


def iter_jsonl(path: str):
    """Yield decoded JSON objects from a JSONL file. Skips blank lines."""
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _read_json_if_exists(path: str) -> Optional[Any]:
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def _iso(wall_seconds: float) -> str:
    return datetime.fromtimestamp(wall_seconds, tz=timezone.utc).isoformat()


def _json_default(obj: Any) -> Any:
    if isinstance(obj, set):
        return sorted(obj)
    if hasattr(obj, "to_dict"):
        try:
            return obj.to_dict()
        except Exception:
            pass
    return repr(obj)
