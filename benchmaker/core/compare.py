"""Compare finished runs: multi-arm table, same-job enforcement, TTFT parity,
and the TTFT mix/latency decomposition (issues #3 and #2).

Everything here is pure: it takes bundles already on disk (``read_bundle``
dicts) or summary dicts, so the comparison can be re-run and audited after
the fact — ``benchmaker arms`` and ``benchmaker collect`` both call into it.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Iterable, Optional

# ---------------------------------------------------------------------------
# Metric paths
# ---------------------------------------------------------------------------

# Friendly name -> dotted path in summary.json. ``lower`` says which direction
# is better (for the "vs ref" column).
METRIC_ALIASES: dict[str, tuple[str, bool]] = {
    "ttft_p50": ("workload_metrics.ttft_s.p50", True),
    "ttft_p90": ("workload_metrics.ttft_s.p90", True),
    "ttft_p99": ("workload_metrics.ttft_s.p99", True),
    "ttft_mean": ("workload_metrics.ttft_s.mean", True),
    "lat_p50": ("latency_s.p50", True),
    "lat_p99": ("latency_s.p99", True),
    "itl_p50": ("itl_ms_pooled.p50", True),
    "itl_p99": ("itl_ms_pooled.p99", True),
    "itl_p99.5": ("itl_ms_pooled.p995", True),
    "itl_p99.9": ("itl_ms_pooled.p999", True),
    "itl_max": ("itl_ms_pooled.max", True),
    "stalls_per_req": ("stalls.per_request.mean", True),
    "stalls_per_req_p99": ("stalls.per_request.p99", True),
    "stall_total_ms": ("stalls.total_time_ms", True),
    "stall_excess_ms": ("stalls.excess_ms", True),
    "rps": ("throughput_rps", False),
    "goodput_rps": ("goodput_rps", False),
    "err_pct": ("error_rate", True),
    "req_fail_pct": ("request_failure_rate", True),
    "tok_out_per_s": ("workload_metrics.tokens_per_s.mean", False),
    "cached_pct": ("workload_metrics.cached_ratio.mean", False),
    "prompt_tokens": ("workload_metrics.prompt_tokens.mean", None),
    "tokens_out": ("workload_metrics.tokens_out.mean", None),
}

DEFAULT_METRICS = [
    "ttft_p50", "ttft_p99", "itl_p99", "itl_p99.5", "stalls_per_req",
    "stall_excess_ms", "rps", "err_pct",
]

_ALLOCATION_COLUMNS = {
    "device_kv_tokens": "device KV tokens",
    "host_tier_bytes": "host bytes",
    "host_tier_tokens": "host KV tokens",
    "bytes_per_token": "host bytes/token",
    "sharing": "host sharing",
}
FLEET_HOST_TOLERANCE = 0.05

# Metrics reported as percentages (value 0..1 -> 0..100).
_PCT_METRICS = {"err_pct", "req_fail_pct", "cached_pct"}


def resolve_metric(name: str) -> tuple[str, Optional[bool]]:
    """``(dotted_path, lower_is_better)`` for an alias or a raw dotted path."""
    if name in METRIC_ALIASES:
        return METRIC_ALIASES[name]
    path = name
    lower = None
    for alias, (p, lo) in METRIC_ALIASES.items():
        if p == path:
            lower = lo
            name = alias
            break
    return path, lower


def dotted_get(d: Any, path: str, default: Any = None) -> Any:
    """``d["a"]["b"]["c"]`` for ``"a.b.c"``; returns ``default`` if missing."""
    cur = d
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


def metric_value(summary: dict, name: str) -> Optional[float]:
    path, _ = resolve_metric(name)
    v = dotted_get(summary, path)
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    v = float(v)
    if name in _PCT_METRICS:
        v *= 100.0
    return v


def _fmt(name: str, v: Optional[float]) -> str:
    if v is None:
        return "—"
    if name in _PCT_METRICS:
        return f"{v:.2f}%"
    if name.startswith("itl_") or name.startswith("stall"):
        return f"{v:.2f}"
    if "ttft" in name or name.startswith("lat"):
        return f"{v * 1e3:.1f}" if v < 10 else f"{v:.2f}"
    return f"{v:.2f}"


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------


@dataclass
class RunRef:
    """One finished run: its bundle location plus the pieces we compare on."""

    arm: str
    rep: int
    bundle_dir: str
    summary: dict
    manifest: Optional[dict] = None
    meta: Optional[dict] = None
    warmup: bool = False
    extra: dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.arm}/rep{self.rep}"

    @classmethod
    def from_bundle(cls, arm: str, rep: int, bundle_dir: str, *,
                    warmup: bool = False) -> "RunRef":
        from benchmaker.io.bundle import read_bundle
        b = read_bundle(bundle_dir)
        return cls(arm=arm, rep=rep, bundle_dir=bundle_dir, summary=b["summary"],
                   manifest=b.get("manifest"), meta=b.get("meta"), warmup=warmup)

    def is_void(self) -> Optional[dict]:
        """The void verdict (issue #4) if this run was marked void.

        ``meta.void`` is a bool in real bundles (``write_bundle`` stores the
        verdict flag and lists the reasons in ``meta.void_violations``); a
        dict ``{"void": True, ...}`` is accepted too. Returns
        ``{"void": True, "violations": [...]}`` so callers can show why."""
        if not self.meta:
            return None
        v = self.meta.get("void")
        if isinstance(v, dict):
            return v if v.get("void") else None
        if v is True:
            return {"void": True, "violations": list(self.meta.get("void_violations") or [])}
        return None


def load_runs(arm_dirs: dict[str, Iterable[str]]) -> list[RunRef]:
    """``{arm: [bundle_dir, ...]}`` → RunRefs (rep index = position)."""
    runs = []
    for arm, dirs in arm_dirs.items():
        for rep, d in enumerate(sorted(dirs)):
            runs.append(RunRef.from_bundle(arm, rep, d))
    return runs


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------


@dataclass
class ArmStats:
    arm: str
    reps: int
    metrics: dict[str, dict] = field(default_factory=dict)  # metric -> stats

    def to_dict(self) -> dict:
        return {"arm": self.arm, "reps": self.reps, "metrics": self.metrics}


def _quantile(vals: list[float], q: float) -> float:
    s = sorted(vals)
    if not s:
        return float("nan")
    if len(s) == 1:
        return s[0]
    pos = q * (len(s) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def compare(runs: list[RunRef], metrics: Optional[list[str]] = None,
            reference: Optional[str] = None, *,
            exclude_warmup: bool = True) -> dict:
    """Per-arm, per-metric median and min–max across reps, with ratios to
    ``reference`` (default: the first arm). Void runs are excluded."""
    metrics = list(metrics or DEFAULT_METRICS)
    arms = sorted({r.arm for r in runs})
    if not arms:
        raise ValueError("no runs to compare")
    ref = reference or arms[0]
    if ref not in arms:
        raise ValueError(f"reference arm {ref!r} not in runs {arms}")

    by_arm: dict[str, list[RunRef]] = {a: [] for a in arms}
    excluded: dict[str, int] = {}
    for r in runs:
        if exclude_warmup and r.warmup:
            excluded[r.arm] = excluded.get(r.arm, 0) + 1
            continue
        if r.is_void():
            excluded[r.arm] = excluded.get(r.arm, 0) + 1
            continue
        by_arm[r.arm].append(r)

    table: dict[str, ArmStats] = {}
    for arm in arms:
        stats: dict[str, dict] = {}
        for m in metrics:
            path, lower = resolve_metric(m)
            vals = [metric_value(r.summary, m) for r in by_arm[arm]]
            vals = [v for v in vals if v is not None]
            if not vals:
                stats[m] = {"path": path, "lower_is_better": lower, "reps": 0}
                continue
            stats[m] = {
                "path": path, "lower_is_better": lower, "reps": len(vals),
                "median": statistics.median(vals), "mean": statistics.mean(vals),
                "min": min(vals), "max": max(vals),
                "values": vals,
            }
        table[arm] = ArmStats(arm=arm, reps=len(by_arm[arm]), metrics=stats)

    loop_bound = {
        arm: [r.summary["closed_loop"] for r in by_arm[arm]
              if (r.summary.get("closed_loop") or {}).get("loop_bound")]
        for arm in arms
    }
    loop_bound = {arm: reps for arm, reps in loop_bound.items() if reps}
    for arm in arms:
        for m, st in table[arm].metrics.items():
            ref_st = table[ref].metrics.get(m) or {}
            ratio = None
            if "median" in st and "median" in ref_st and ref_st["median"] != 0:
                ratio = st["median"] / ref_st["median"]
            if st["path"] in {"throughput_rps", "goodput_rps"} and (arm in loop_bound or ref in loop_bound):
                ratio = None
            st["ratio_vs_ref"] = ratio
            st["better_than_ref"] = _better(st, ref_st, ratio)

    allocations = {arm: [_run_allocation(run) for run in by_arm[arm]] for arm in arms}
    return {
        "metrics": metrics,
        "reference": ref,
        "arms": {a: {**table[a].to_dict(), "allocations": allocations[a]} for a in arms},
        "allocation_warnings": _allocation_warnings(allocations),
        "cache_restores": {
            arm: [{"rep": r.rep, "sources": r.summary.get("cache_restores", {})} for r in by_arm[arm]]
            for arm in arms
        },
        "excluded": excluded,
        "loop_bound": loop_bound,
        "void": {r.arm: r.is_void() for r in runs if r.is_void()},
    }


def _run_allocation(run: RunRef) -> dict:
    manifest = run.manifest or {}
    return {
        "rep": run.rep,
        "arm": manifest.get("allocation") or {},
        "endpoints": {
            name: endpoint.get("allocation") or {}
            for name, endpoint in (manifest.get("endpoints") or {}).items()
        },
    }


def _allocation_warnings(allocations: dict[str, list[dict]]) -> list[str]:
    totals = {
        arm: sorted({
            rep["arm"]["fleet_host_total_bytes"]
            for rep in reps if rep["arm"].get("fleet_host_total_bytes") is not None
        })
        for arm, reps in allocations.items()
    }
    totals = {arm: values for arm, values in totals.items() if values}
    for left, right in combinations(totals.values(), 2):
        if (right[-1] - left[0] > FLEET_HOST_TOLERANCE * right[-1]
                or left[-1] - right[0] > FLEET_HOST_TOLERANCE * left[-1]):
            detail = ", ".join(f"{name}={_format_alloc(vals)}" for name, vals in totals.items())
            return [f"Fleet host KV budgets differ by more than {FLEET_HOST_TOLERANCE:.0%} ({detail} bytes)."]
    return []


def _format_alloc(values: Iterable[Any]) -> str:
    return " / ".join(dict.fromkeys(
        "—" if v is None else f"{v:,}" if isinstance(v, int) else f"{v:,.2f}"
        if isinstance(v, float) else str(v).replace("|", "\\|")
        for v in values
    )) or "—"


def _allocation_cells(reps: list[dict]) -> list[str]:
    names = sorted({name for rep in reps for name in rep["endpoints"]})
    cells = []
    for key in _ALLOCATION_COLUMNS:
        if names:
            cell = "<br>".join(
                _format_alloc([name]) + " "
                + _format_alloc(rep["endpoints"].get(name, {}).get(key) for rep in reps)
                for name in names
            )
        else:
            cell = _format_alloc(rep["arm"].get(key) for rep in reps)
        cells.append(cell)
    cells.append(_format_alloc(rep["arm"].get("fleet_host_total_bytes") for rep in reps))
    return cells


def _better(st: dict, ref_st: dict, ratio: Optional[float]) -> Optional[bool]:
    if ratio is None or st.get("lower_is_better") is None:
        return None
    lower = st["lower_is_better"]
    if ratio == 1.0:
        return None
    return (ratio < 1.0) if lower else (ratio > 1.0)


def format_value(name: str, v: Optional[float]) -> str:
    return _fmt(name, v)


def _render_cache_restores(reps_by_arm: dict, order: list[str]) -> list[str]:
    if not any(rep["sources"] for reps in reps_by_arm.values() for rep in reps):
        return []
    lines = [
        "", "### Host-tier KV restores", "",
        "| arm | rep | monitor | engine | restored share of prompt tokens | restored bytes | attribution |",
        "|---|---:|---|---|---:|---:|---|",
    ]
    for arm in order:
        for rep in reps_by_arm.get(arm) or [{"rep": "—", "sources": {}}]:
            sources = rep["sources"] or {"—": {"reason": "no restore telemetry"}}
            for source, report in sources.items():
                share = report.get("restored_token_share")
                restored_bytes = report.get("restored_bytes")
                status = report.get("attribution", "unavailable")
                if report.get("reason"):
                    status += f" ({report['reason']})"
                if report.get("scope"):
                    status += f", {report['scope']}"
                byte_display = f"{restored_bytes:,.0f}" if restored_bytes is not None else "unavailable"
                if report.get("restored_bytes_source") == "estimated from tokens":
                    byte_display += " (estimate)"
                share_display = f"{share:.2%}" if share is not None else "unavailable"
                if share is not None and report.get("restored_token_estimate"):
                    share_display += f" ({report['restored_token_estimate']})"
                cells = [
                    arm, str(rep["rep"]), source, report.get("engine", "—"),
                    share_display, byte_display, status,
                ]
                lines.append("| " + " | ".join(cell.replace("|", "\\|") for cell in cells) + " |")
    return lines


def render_comparison(result: dict, *, ref: Optional[str] = None) -> str:
    """Markdown table: one row per arm, ``median [min–max] (×ref)`` per metric."""
    ref = ref or result["reference"]
    metrics = result["metrics"]
    arms = result["arms"]
    show_allocation = any(rep["arm"] for arm in arms.values() for rep in arm.get("allocations", []))
    head = ["arm", "reps"] + metrics
    if show_allocation:
        head += [*_ALLOCATION_COLUMNS.values(), "fleet host bytes"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    order = [ref] + [a for a in arms if a != ref]
    for arm in order:
        st = arms[arm]
        cells = [f"**{arm}**" if arm == ref else arm, str(st["reps"])]
        for m in metrics:
            s = st["metrics"].get(m) or {}
            if "median" not in s:
                cells.append("—")
                continue
            cell = f"{_fmt(m, s['median'])} [{_fmt(m, s['min'])}–{_fmt(m, s['max'])}]"
            if arm in result.get("loop_bound", {}) and s.get("path") in {"throughput_rps", "goodput_rps"}:
                cell += " (loop_bound)"
            if arm != ref and s.get("ratio_vs_ref") is not None:
                mark = ""
                if s.get("better_than_ref") is True:
                    mark = " ✓"
                elif s.get("better_than_ref") is False:
                    mark = " ✗"
                cell += f" (×{s['ratio_vs_ref']:.3f}{mark})"
            cells.append(cell)
        if show_allocation:
            cells.extend(_allocation_cells(st.get("allocations", [])))
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend(_render_cache_restores(result.get("cache_restores", {}), order))
    for warning in result.get("allocation_warnings", []):
        lines.append(f"\n⚠ {warning}")
    if result.get("signoff"):
        counts = result["signoff"]
        lines.extend(["", f"_Reps: {counts['passed']} passed, {counts['void']} void, "
                      f"{counts['failed']} failed, {counts['warmup']} warm-up._"])
    if result.get("loop_bound"):
        lines.extend(["", "_Loop-bound throughput reflects session pacing. "
                      "Use cache hit rate and TTFT by cache class to compare these runs._"])
        lines.extend(["", "| arm | loop-bound reps | cache hit rate | TTFT p50 by cache class (ms) |",
                      "|---|---:|---:|---|"])
        for arm, reps in result["loop_bound"].items():
            hit_rates = [r["cache_hit_rate"] for r in reps if "cache_hit_rate" in r]
            hit = f"{statistics.median(hit_rates):.1%}" if hit_rates else "unavailable"
            ttfts: dict[str, list[float]] = {}
            for rep in reps:
                for cls, value in rep.get("cache_class_ttft_p50_s", {}).items():
                    if value is not None:
                        ttfts.setdefault(cls, []).append(value)
            ttft = ", ".join(f"{cls} {statistics.median(values) * 1000:.1f}"
                             for cls, values in sorted(ttfts.items())) or "unavailable"
            lines.append(f"| {arm} | {len(reps)} | {hit} | {ttft} |")
    if result.get("excluded"):
        ex = ", ".join(f"{a}: {n}" for a, n in sorted(result["excluded"].items()))
        lines.append("")
        lines.append(f"_Excluded reps: {ex}_")
    if result.get("void"):
        for arm, v in result["void"].items():
            reasons = [entry.get("reason") for entry in v.get("violations", [])]
            detail = ", ".join(reason for reason in reasons if reason) or v.get("reason", "void rule violated")
            lines.append(f"_**{arm} VOID**: {detail}_")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Same-job / same-host enforcement (issue #3)
# ---------------------------------------------------------------------------


def check_same_job(runs: list[RunRef], *, allow_mixed: bool = False,
                   exclude_warmup: bool = True) -> list[str]:
    """Violations of "all arms ran in the same job on the same host".

    Every non-warm-up run must carry a manifest with ``host.job_id`` and
    ``host.hostname``; all must agree. With ``allow_mixed`` the check only
    reports (so the caller can print a warning) instead of raising.
    """
    ident: dict[tuple[str, str], list[str]] = {}
    missing: list[str] = []
    for r in runs:
        if exclude_warmup and r.warmup:
            continue
        if not r.manifest:
            missing.append(r.key)
            continue
        host = r.manifest.get("host") or {}
        ident.setdefault((str(host.get("job_id")), str(host.get("hostname"))), []).append(r.key)
    problems = []
    if missing:
        problems.append("runs without a manifest (cannot verify job/host): " + ", ".join(missing))
    if len(ident) > 1:
        parts = "; ".join(f"job={j} host={h}: " + ", ".join(keys) for (j, h), keys in sorted(ident.items()))
        problems.append("results come from different jobs/hosts: " + parts)
    if problems and not allow_mixed:
        raise ValueError("refusing to merge results across jobs/hosts (pass --allow-mixed-jobs "
                         "or set arms.allow_mixed_jobs: true): " + " | ".join(problems))
    return problems


# ---------------------------------------------------------------------------
# TTFT token parity (issue #4)
# ---------------------------------------------------------------------------


def ttft_parity(runs: list[RunRef], *, exclude_warmup: bool = True,
               exclude_void: bool = True) -> list[dict]:
    """Which token each arm's TTFT was stamped on.

    Returns one entry per arm: ``{"arm", "kind", "index_median", "source"}``;
    ``kind`` comes from the configured ``ttft_token`` knob when present,
    otherwise ``"any"``. A reasoning parser on one arm and not the other
    shifts TTFT by a decode step; the caller warns when ``index_median``
    or ``kind`` differ across arms. Void runs are left out like warm-up
    reps (issue #17): they are not in the table, so they must not steer
    its warnings either.
    """
    per_arm: dict[str, list[RunRef]] = {}
    for r in runs:
        if exclude_warmup and r.warmup:
            continue
        if exclude_void and r.is_void():
            continue
        per_arm.setdefault(r.arm, []).append(r)
    out = []
    for arm in sorted(per_arm):
        idx = [metric_value(r.summary, "workload_metrics.ttft_token_index.mean") for r in per_arm[arm]]
        idx = [v for v in idx if v is not None]
        kind = None
        for r in per_arm[arm]:
            src = (r.meta or {}).get("source_config") or {}
            wt = src.get("workload_type") or {}
            if isinstance(wt, dict) and wt.get("ttft_token"):
                kind = wt["ttft_token"]
                break
        out.append({
            "arm": arm,
            "kind": kind or "any",
            "index_median": statistics.median(idx) if idx else None,
            "index_values": idx,
        })
    return out


def ttft_parity_warnings(entries: list[dict]) -> list[str]:
    if len(entries) < 2:
        return []
    warns = []
    kinds = {e["kind"] for e in entries}
    if len(kinds) > 1:
        warns.append("arms stamp TTFT on different token kinds: "
                     + ", ".join(f"{e['arm']}={e['kind']}" for e in entries)
                     + " (a reasoning parser on one side shifts TTFT by a decode step)")
    idx = {e["arm"]: e["index_median"] for e in entries if e["index_median"] is not None}
    if idx and max(idx.values()) - min(idx.values()) >= 1.0:
        warns.append("arms' first-token index differs: "
                     + ", ".join(f"{a}={v:g}" for a, v in sorted(idx.items())))
    return warns


# ---------------------------------------------------------------------------
# TTFT mix/latency decomposition (issue #2)
# ---------------------------------------------------------------------------


def class_shares(summary: dict, key: str = "cache_class", metric: str = "workload_metrics.ttft_s.mean",
                 include_unclassified: bool = True) -> dict[str, tuple[float, float]]:
    """``{class: (share, mean_latency)}`` from ``summary["groups"][key]``."""
    groups = (summary.get("groups") or {}).get(key) or {}
    out: dict[str, tuple[float, float]] = {}
    for cls, sub in groups.items():
        if not include_unclassified and cls in ("(none)", "(ungrouped)", "unknown"):
            continue
        mean = dotted_get(sub, metric)
        if mean is None:
            continue
        out[cls] = (float(sub.get("share", 0.0)), float(mean))
    return out


def decompose_ttft(summary_a: dict, summary_b: dict, *, key: str = "cache_class",
                   metric: str = "workload_metrics.ttft_s.mean") -> dict:
    """Split ``Δ mean TTFT = mean_b − mean_a`` into a *mix* and a *latency* part.

    Symmetric (Shapley/midpoint) decomposition, exact and order-independent:

        Δ       = Σ_c (s_b − s_a)·(μ_a + μ_b)/2  +  Σ_c (s_a + s_b)/2·(μ_b − μ_a)
                  |------------ mix -------------|    |-------- latency --------|

    where ``s`` is the class share and ``μ`` the per-class mean. The mix term
    is what B gained from a better hit mix alone; the latency term is what it
    gained from serving each class faster.
    """
    A = class_shares(summary_a, key, metric)
    B = class_shares(summary_b, key, metric)
    classes = sorted(set(A) | set(B))
    per_class = {}
    mix = lat = 0.0
    for c in classes:
        s_a, mu_a = A.get(c, (0.0, 0.0))
        s_b, mu_b = B.get(c, (0.0, 0.0))
        # A class missing on one side: use the other side's mean for both so the
        # whole difference lands in the mix term (no latency claim without data).
        if c not in A:
            mu_a = mu_b
        if c not in B:
            mu_b = mu_a
        m = (s_b - s_a) * (mu_a + mu_b) / 2.0
        lat_c = (s_a + s_b) / 2.0 * (mu_b - mu_a)
        mix += m
        lat += lat_c
        per_class[c] = {
            "share_a": s_a, "share_b": s_b, "mean_a": mu_a, "mean_b": mu_b,
            "mix_ms": m * 1e3, "latency_ms": lat_c * 1e3,
        }
    mean_a = sum(s * mu for s, mu in A.values())
    mean_b = sum(s * mu for s, mu in B.values())
    delta = mean_b - mean_a
    return {
        "group_key": key,
        "mean_a_s": mean_a, "mean_b_s": mean_b,
        "delta_ms": delta * 1e3,
        "mix_ms": mix * 1e3, "latency_ms": lat * 1e3,
        "mix_share": (mix / delta) if delta else None,
        "latency_share": (lat / delta) if delta else None,
        "per_class": per_class,
    }


def render_decomposition(dec: dict, name_a: str = "A", name_b: str = "B") -> str:
    def ms(v: float) -> str:
        return f"{v:+.2f}"

    lines = [
        f"Δ mean TTFT ({name_b} − {name_a}) = {ms(dec['delta_ms'])} ms = "
        f"mix {ms(dec['mix_ms'])} ms + latency {ms(dec['latency_ms'])} ms"
    ]
    if dec.get("mix_share") is not None:
        lines.append(f"  mix {dec['mix_share'] * 100:+.0f}%  ·  latency {dec['latency_share'] * 100:+.0f}%")
    lines.append("")
    lines.append("| class | share " + name_a + " | share " + name_b + " | mean " + name_a
                 + " ms | mean " + name_b + " ms | mix ms | latency ms |")
    lines.append("|---|---|---|---|---|---|---|")
    for c, v in sorted(dec["per_class"].items(), key=lambda kv: -abs(kv[1]["mix_ms"] + kv[1]["latency_ms"])):
        lines.append(f"| {c} | {v['share_a'] * 100:.1f}% | {v['share_b'] * 100:.1f}% | "
                     f"{v['mean_a'] * 1e3:.1f} | {v['mean_b'] * 1e3:.1f} | "
                     f"{v['mix_ms']:+.2f} | {v['latency_ms']:+.2f} |")
    return "\n".join(lines)
