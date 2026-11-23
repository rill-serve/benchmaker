"""Aggregation + reporting of Samples."""

from __future__ import annotations

import json
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional, TextIO

from benchmaker.core.types import Sample

# Series key the LLM workload-types use for per-token inter-token gaps (ms).
# The summary pools this series across requests and derives the decode-stall
# block from it (#1).
ITL_SERIES = "itl_ms"
DEFAULT_STALL_MS = 12.0

# Warm-up requests carry ``meta["warmup"] = True`` (set by a workload item or
# by ``BenchConfig.warmup_requests`` / ``warmup_s``) and are left out of every
# summary block (#5).
WARMUP_META_KEY = "warmup"

# Bucket label used for samples that lack a value for a ``group_by`` key.
UNGROUPED = "(none)"

# Upper edges (ms) of the log2 gap histogram. The last bin is open-ended.
_HISTOGRAM_EDGES_MS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)


def _pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = (len(xs) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return xs[f]
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def _stat_block(xs: list[float], pct_keys=(50, 90, 99)) -> dict:
    if not xs:
        return {}
    out: dict[str, float] = {
        "mean": statistics.mean(xs),
        "min": min(xs),
        "max": max(xs),
    }
    for p in pct_keys:
        out[f"p{str(p).replace('.', '')}"] = _pct(xs, p)
    return out


@dataclass
class MetricsAggregator:
    """Accumulates samples and produces a summary.

    ``stall_ms`` is the inter-token gap above which a decode step counts as a
    stall (issue #1). ``None`` disables the ``stalls`` block. ``group_by`` lists
    ``Sample.meta`` keys to break the summary down by (e.g. ``["cache_class"]``
    for per-cache-class TTFT, issue #2). ``dump_series`` writes each sample's
    raw ``series`` (per-token gaps) into samples.jsonl. ``phase_windows`` is
    filled by the runner for phased runs (issue #5): phase name -> (start, end)
    on the monotonic clock.
    """

    samples: list[Sample] = field(default_factory=list)
    start_time: float = field(default_factory=time.monotonic)
    end_time: Optional[float] = None
    # Wall-clock (time.time) markers, only used when writing the run bundle.
    start_wall: float = field(default_factory=time.time)
    end_wall: Optional[float] = None
    # name -> list of (elapsed_s, {metric_name: value})
    monitor_samples: dict[str, list[tuple[float, dict[str, float]]]] = field(default_factory=dict)
    stall_ms: Optional[float] = DEFAULT_STALL_MS
    group_by: list[str] = field(default_factory=list)
    dump_series: bool = False
    phase_windows: dict[str, tuple[float, float]] = field(default_factory=dict)

    def add(self, sample: Sample) -> None:
        self.samples.append(sample)

    def monitor_buffer(self, name: str) -> list[tuple[float, dict[str, float]]]:
        if name not in self.monitor_samples:
            self.monitor_samples[name] = []
        return self.monitor_samples[name]

    def finalize(self) -> None:
        self.end_time = time.monotonic()
        self.end_wall = time.time()

    # ------------------------------------------------------------- summary

    def summary(self) -> dict:
        end = self.end_time or time.monotonic()
        wall_s = max(end - self.start_time, 1e-9)

        # Warm-up requests are recorded (samples.jsonl keeps them, tagged) but
        # excluded from every summary block (#5).
        warm = [s for s in self.samples if _is_warmup(s)]
        samples = [s for s in self.samples if not _is_warmup(s)]
        # Throughput is measured over the window the measured requests
        # actually span — first measured start to last measured completion —
        # not the whole run: ramp-up, warm-up and a drained queue are not the
        # server's steady state (issue #17). Latency and the other metrics
        # are per request and need no window.
        window_s, w_start, w_end = measured_window(samples, self.start_time, end)
        out = _summary_for_samples(samples, window_s, stall_ms=self.stall_ms)
        out["wall_time_s"] = wall_s
        out["measured_window_s"] = window_s
        out["measured_window"] = {"start_s": w_start - self.start_time, "end_s": w_end - self.start_time}
        if warm:
            out["warmup_excluded"] = len(warm)

        # A mixed benchmark needs each lane's SLO signal independently.  Use
        # the same window as the aggregate so lane throughput is directly
        # comparable to the total, while latency and workload metrics remain
        # scoped to that lane's samples.
        out["lanes"] = _grouped(
            samples, "lane", window_s, self.stall_ms, nested=("phase",))
        _apply_group_by(out, samples, window_s, self.group_by, self.stall_ms)
        if self.phase_windows:
            phases: dict[str, dict] = {}
            for name, (p_start, p_end) in self.phase_windows.items():
                bucket = [s for s in samples if s.meta.get("phase") == name]
                if not bucket:
                    continue
                phase_wall = max(p_end - p_start, 1e-9)
                sub = _summary_for_samples(bucket, phase_wall, stall_ms=self.stall_ms)
                sub["lanes"] = _grouped(bucket, "lane", phase_wall, self.stall_ms)
                # same breakdowns as the top level, so a per-phase cache-class
                # table (capacity warm vs revisit) falls out of group_by (#2, #5)
                _apply_group_by(sub, bucket, phase_wall, self.group_by, self.stall_ms)
                # How long the phase was actually producing: a lane that
                # exhausts its items closes the window early.
                sub["active_s"] = max(s.start_ts + s.latency_s for s in bucket) - p_start
                sub["start_s"] = p_start - self.start_time
                sub["end_s"] = p_end - self.start_time
                phases[name] = sub
            if phases:
                out["phases"] = phases

        # Monitor time-series: summarize each metric per monitor.
        if self.monitor_samples:
            monitors_summary: dict[str, dict] = {}
            for mon_name, ticks in self.monitor_samples.items():
                if not ticks:
                    continue
                by_metric: dict[str, list[float]] = defaultdict(list)
                for _t, values in ticks:
                    for k, v in values.items():
                        if isinstance(v, (int, float)):
                            by_metric[k].append(float(v))
                per_metric = {}
                for k, vals in by_metric.items():
                    per_metric[k] = {
                        "mean": statistics.mean(vals),
                        "min": min(vals),
                        "max": max(vals),
                        "p50": _pct(vals, 50),
                        "p90": _pct(vals, 90),
                        "p99": _pct(vals, 99),
                        "first": vals[0],
                        "last": vals[-1],
                    }
                monitors_summary[mon_name] = {
                    "tick_count": len(ticks),
                    "metrics": per_metric,
                }
            if monitors_summary:
                out["monitors"] = monitors_summary
        return out

    # ------------------------------------------------------------ rendering

    def render(self, out: TextIO) -> None:
        s = self.summary()
        lines: list[str] = []
        lines.append("=" * 60)
        lines.append(f"[benchmaker] results  ({s['total_requests']} requests, "
                     f"{s['wall_time_s']:.2f}s wall)")
        if s.get("warmup_excluded"):
            lines.append(f"  (excluding {s['warmup_excluded']} warm-up requests)")
        lines.append("=" * 60)
        window = s.get("measured_window_s")
        if window and abs(window - s["wall_time_s"]) > 0.5:
            lines.append(f"  throughput     : {s['throughput_rps']:>10.2f} req/s  "
                         f"(over the measured {window:.2f}s of {s['wall_time_s']:.2f}s wall)")
        else:
            lines.append(f"  throughput     : {s['throughput_rps']:>10.2f} req/s")
        lines.append(f"  goodput        : {s['goodput_rps']:>10.2f} req/s")
        lines.append(f"  success        : {s['success']}")
        lines.append(f"  failed         : {s['failed']}  ({s['error_rate']*100:.2f}%)")
        lines.append(
            f"    request failed : {s['request_failed']}  "
            f"({s['request_failure_rate']*100:.2f}%)"
        )
        lines.append(f"    wrong output   : {s['wrong_output']}")
        if s.get("latency_s"):
            lat = s["latency_s"]
            lines.append("")
            lines.append("  latency (s)")
            for k in ("mean", "p50", "p90", "p95", "p99", "p999", "max"):
                lines.append(f"    {k:<6}: {lat[k]:.4f}")
        if s["status_codes"]:
            lines.append("")
            lines.append("  status codes")
            for code, n in sorted(s["status_codes"].items()):
                lines.append(f"    {code:<4} : {n}")
        if s["errors"]:
            lines.append("")
            lines.append("  errors")
            for err, n in sorted(s["errors"].items(), key=lambda kv: -kv[1])[:10]:
                lines.append(f"    {n:<4} x {err}")
        if s.get("workload_metrics"):
            lines.append("")
            lines.append("  workload metrics")
            for k, v in s["workload_metrics"].items():
                lines.append(f"    {k}")
                for kk in ("mean", "p50", "p90", "p99", "max"):
                    lines.append(f"      {kk:<6}: {v[kk]:.4f}")
        lines.extend(_render_itl(s))
        lines.extend(_render_groups(s))
        if s.get("phases"):
            lines.append("")
            lines.append("  phases")
            for name, ph in s["phases"].items():
                lines.append(
                    f"    {name}: [{ph['start_s']:.1f}s .. {ph['end_s']:.1f}s] "
                    f"{ph['total_requests']} requests, "
                    f"{ph['throughput_rps']:.2f} req/s, "
                    f"{ph['success']} success"
                )
                lines.extend(_render_lane_lines(ph, indent=6))
        if s.get("lanes"):
            lines.append("")
            lines.append("  lanes")
            lines.extend(_render_lane_lines(s, indent=4))
        if s.get("monitors"):
            for mon_name, mon in s["monitors"].items():
                lines.append("")
                lines.append(f"  monitor: {mon_name}  ({mon['tick_count']} ticks)")
                for k, v in mon["metrics"].items():
                    lines.append(f"    {k}")
                    for kk in ("mean", "min", "max", "p50", "p99", "last"):
                        lines.append(f"      {kk:<6}: {v[kk]:.4f}")
        lines.append("=" * 60)
        out.write("\n".join(lines) + "\n")
        out.flush()

    # -------------------------------------------------------------- dumping

    def dump_samples_jsonl(self, path: str) -> None:
        """Write per-request records for offline analysis.

        Raw ``series`` (per-token gaps) are included only when ``dump_series``
        is on — they dominate the file size otherwise (#1).
        """
        with open(path, "w") as f:
            for s in self.samples:
                rec = {
                    "start_ts": s.start_ts,
                    "latency_s": s.latency_s,
                    "status": s.status,
                    "ok": s.ok,
                    "request_ok": s.request_ok,
                    "bytes_sent": s.bytes_sent,
                    "bytes_recv": s.bytes_recv,
                    "error": s.error,
                    "workload": s.workload,
                    "meta": _safe_meta(s.meta),
                    "extra": s.extra,
                }
                if self.dump_series and s.series:
                    rec["series"] = {k: list(v) for k, v in s.series.items()}
                f.write(json.dumps(rec) + "\n")

    def dump_monitor_jsonl(self, path: str) -> None:
        """Write monitor time-series ticks as JSONL for plotting/analysis."""
        with open(path, "w") as f:
            for mon_name, ticks in self.monitor_samples.items():
                for t, values in ticks:
                    f.write(json.dumps({
                        "monitor": mon_name,
                        "elapsed_s": t,
                        "values": values,
                    }) + "\n")


# --------------------------------------------------------------------------- #
# Summaries                                                                    #
# --------------------------------------------------------------------------- #


def _summary_for_samples(samples: list[Sample], wall_s: float,
                         stall_ms: Optional[float] = DEFAULT_STALL_MS) -> dict:
    """Summarize a sample subset over a benchmark wall-clock interval."""
    ok = [s for s in samples if s.ok]
    fail = [s for s in samples if not s.ok]
    # Split fail into transport failures vs. delivered-but-graded-wrong.
    wrong = [s for s in fail if s.request_ok]
    request_failed = [s for s in fail if not s.request_ok]
    latencies = [s.latency_s for s in ok]

    status_counts = Counter(s.status for s in samples)
    error_counts = Counter(s.error for s in fail if s.error)

    out: dict = {
        "wall_time_s": wall_s,
        "total_requests": len(samples),
        "success": len(ok),
        "failed": len(fail),
        "request_failed": len(request_failed),
        "wrong_output": len(wrong),
        "error_rate": (len(fail) / len(samples)) if samples else 0.0,
        "request_failure_rate": (
            (len(request_failed) / len(samples)) if samples else 0.0
        ),
        "throughput_rps": len(samples) / wall_s,
        "goodput_rps": len(ok) / wall_s,
        "bytes_sent": sum(s.bytes_sent for s in samples),
        "bytes_recv": sum(s.bytes_recv for s in samples),
        "status_codes": dict(status_counts),
        "errors": dict(error_counts),
    }
    if latencies:
        out["latency_s"] = {
            "mean": statistics.mean(latencies),
            "min": min(latencies),
            "max": max(latencies),
            "p50": _pct(latencies, 50),
            "p90": _pct(latencies, 90),
            "p95": _pct(latencies, 95),
            "p99": _pct(latencies, 99),
            "p999": _pct(latencies, 99.9),
        }

    # Aggregate workload-specific `extra` metrics generically: mean + percentiles.
    extras: dict[str, list[float]] = defaultdict(list)
    for s in ok:
        for k, v in s.extra.items():
            if isinstance(v, (int, float)):
                extras[k].append(float(v))
    if extras:
        ext_summary = {}
        for k, vals in extras.items():
            ext_summary[k] = {
                "mean": statistics.mean(vals),
                "p50": _pct(vals, 50),
                "p90": _pct(vals, 90),
                "p99": _pct(vals, 99),
                "min": min(vals),
                "max": max(vals),
            }
        out["workload_metrics"] = ext_summary

    # Raw per-event series are pooled across requests: every observation of
    # every request contributes, so a tail that per-request percentiles hide
    # (a few long decode stalls spread over many requests) is visible (#1).
    series: dict[str, list[list[float]]] = defaultdict(list)
    for s in ok:
        for k, vals in s.series.items():
            if vals:
                series[k].append(vals)
    for key, per_request in series.items():
        pooled = [v for vals in per_request for v in vals]
        out[f"{key}_pooled"] = pooled_series_summary(pooled)
        if key == ITL_SERIES and stall_ms is not None:
            out["stalls"] = stall_summary(per_request, stall_ms)
    # How the itl_ms gaps were counted (issue #17): per token (exact or
    # estimated from the server's token counts) or per streamed event, with
    # the coalesced-read and tokens-per-event figures that explain the two.
    itl_samples = [s for s in ok if s.series.get(ITL_SERIES)]
    if itl_samples:
        out["itl_modes"] = {
            "modes": dict(Counter(str(s.meta.get("itl_mode") or "per_event") for s in itl_samples)),
            "usage_mismatch_requests": sum(1 for s in itl_samples if s.meta.get("itl_usage_mismatch")),
            "coalesced_events_mean": statistics.mean(
                float(s.extra.get("itl_coalesced_events", 0.0)) for s in itl_samples),
            "tokens_per_event_mean": statistics.mean(
                float(s.extra.get("itl_tokens_per_event_mean", 1.0)) for s in itl_samples),
        }

    return out


def measured_window(samples: list[Sample], start_time: float, end_time: float) -> tuple[float, float, float]:
    """``(window_s, window_start, window_end)`` spanned by the measured
    requests: the first one's start to the last one's completion. The whole
    ``[start_time, end_time]`` when there is nothing to measure (issue #17)."""
    if not samples:
        return max(end_time - start_time, 1e-9), start_time, end_time
    w_start = min(s.start_ts for s in samples)
    w_end = max(s.start_ts + s.latency_s for s in samples)
    return max(w_end - w_start, 1e-9), w_start, w_end


def _apply_group_by(out: dict, samples: list[Sample], wall_s: float,
                    group_by: list[str], stall_ms: Optional[float]) -> None:
    """Break ``out`` down by arbitrary meta keys (e.g. cache_class) (#2)."""
    for key in group_by:
        grouped = _grouped(samples, key, wall_s, stall_ms)
        if grouped:
            out.setdefault("groups", {})[key] = grouped


def _grouped(samples: list[Sample], key: str, wall_s: float,
             stall_ms: Optional[float], nested: tuple[str, ...] = ()) -> dict:
    """Bucket samples by ``meta[key]`` and summarize each bucket.

    Each bucket gets a ``share`` (fraction of requests) next to its own summary,
    so a per-class latency table and the class mix live side by side (#2).
    """
    buckets: dict[Any, list[Sample]] = defaultdict(list)
    for s in samples:
        buckets[_group_value(s, key)].append(s)
    if not buckets:
        return {}
    total = len(samples)
    out: dict[str, dict] = {}
    for value, bucket in buckets.items():
        sub = _summary_for_samples(bucket, wall_s, stall_ms=stall_ms)
        sub["share"] = len(bucket) / total if total else 0.0
        for inner in nested:
            inner_groups = _grouped(bucket, inner, wall_s, stall_ms)
            if inner_groups:
                sub[inner if inner != "lane" else "lanes"] = inner_groups
        out[str(value)] = sub
    return dict(sorted(out.items()))


def _group_value(sample: Sample, key: str) -> Any:
    v = sample.meta.get(key)
    if v is None or v == "":
        return UNGROUPED
    return v


def _is_warmup(sample: Sample) -> bool:
    return bool(sample.meta.get(WARMUP_META_KEY))


# --------------------------------------------------------------------------- #
# Pooled series + decode stalls (issue #1)                                     #
# --------------------------------------------------------------------------- #


def pooled_series_summary(values: list[float]) -> dict:
    """Token-level percentiles over every gap of every request.

    Includes p99.5 and p99.9: a change that shortens every stall can *raise*
    p99 (the shorter stalls move up to p99's rank) while p99.5, max and total
    stall time fall — the tail beyond p99 is what decides ITL comparisons.
    """
    return {
        "count": len(values),
        "mean": statistics.mean(values) if values else 0.0,
        "min": min(values) if values else 0.0,
        "max": max(values) if values else 0.0,
        "p50": _pct(values, 50),
        "p90": _pct(values, 90),
        "p99": _pct(values, 99),
        "p995": _pct(values, 99.5),
        "p999": _pct(values, 99.9),
    }


def stall_summary(per_request_gaps: list[list[float]], threshold_ms: float,
                  edges: tuple = _HISTOGRAM_EDGES_MS) -> dict:
    """Decode-stall metrics at a fixed gap threshold.

    A *stall* is one inter-token gap longer than ``threshold_ms``. Reports who
    stalls how often (count, per-request mean/p99) vs. how long (total stall
    time, excess over the threshold) — the two facts p99 alone conflates.
    """
    counts = [sum(1 for g in vals if g > threshold_ms) for vals in per_request_gaps]
    stalled = [g for vals in per_request_gaps for g in vals if g > threshold_ms]
    all_gaps = [g for vals in per_request_gaps for g in vals]
    out: dict[str, Any] = {
        "threshold_ms": threshold_ms,
        "requests": len(per_request_gaps),
        "count": len(stalled),
        "per_request": _stat_block(counts, (50, 90, 99)),
        "total_time_ms": float(sum(stalled)),
        "excess_ms": float(sum(g - threshold_ms for g in stalled)),
        "time_per_request_ms": _stat_block(
            [sum(g for g in vals if g > threshold_ms) for vals in per_request_gaps],
            (50, 99)),
        "stall_ms": _stat_block(stalled, (50, 90, 99)),
        "histogram": log_histogram(all_gaps, edges),
    }
    return out


def log_histogram(values: list[float], edges: tuple = _HISTOGRAM_EDGES_MS) -> list[dict]:
    """Histogram with log2 bins: ``[0, e0), [e0, e1), …, [e_last, inf)``.

    Empty leading/trailing bins are dropped so the table starts at the first
    populated bin. ``share`` is relative to all values.
    """
    if not values:
        return []
    bounds: list[tuple[float, Optional[float]]] = [(0.0, float(edges[0]))]
    bounds += [(float(a), float(b)) for a, b in zip(edges, edges[1:])]
    bounds.append((float(edges[-1]), None))
    counts = [0] * len(bounds)
    for v in values:
        for i, (lo, hi) in enumerate(bounds):
            if v >= lo and (hi is None or v < hi):
                counts[i] += 1
                break
    first = next((i for i, c in enumerate(counts) if c), 0)
    last = max((i for i, c in enumerate(counts) if c), default=first)
    n = len(values)
    return [
        {
            "lo_ms": bounds[i][0],
            "hi_ms": bounds[i][1],
            "count": counts[i],
            "share": counts[i] / n,
        }
        for i in range(first, last + 1)
    ]


# --------------------------------------------------------------------------- #
# Rendering helpers                                                            #
# --------------------------------------------------------------------------- #


def _render_itl(s: dict) -> list[str]:
    lines: list[str] = []
    pooled = s.get(f"{ITL_SERIES}_pooled")
    if pooled:
        lines.append("")
        lines.append(f"  token-level ITL (pooled over {pooled['count']} gaps, ms)")
        for k in ("mean", "p50", "p90", "p99", "p995", "p999", "max"):
            label = "p99.5" if k == "p995" else "p99.9" if k == "p999" else k
            lines.append(f"    {label:<6}: {pooled[k]:.3f}")
        modes = s.get("itl_modes")
        if modes:
            mix = "  ".join(f"{m}={n}" for m, n in sorted(modes["modes"].items()))
            lines.append(f"    gaps per token : {mix}; {modes['coalesced_events_mean']:.2f} coalesced "
                         f"events/request, {modes['tokens_per_event_mean']:.2f} tokens/event"
                         + (f", {modes['usage_mismatch_requests']} requests with usage < events" if modes["usage_mismatch_requests"] else ""))
    stalls = s.get("stalls")
    if stalls and stalls["requests"]:
        lines.append(f"  decode stalls (gap > {stalls['threshold_ms']:g} ms)")
        pr = stalls["per_request"]
        lines.append(
            f"    count          : {stalls['count']}  "
            f"per request mean={pr.get('mean', 0.0):.3f} p99={pr.get('p99', 0.0):.1f}"
        )
        lines.append(
            f"    total time     : {stalls['total_time_ms'] / 1000.0:.3f} s   "
            f"excess over threshold: {stalls['excess_ms'] / 1000.0:.3f} s"
        )
        for b in stalls["histogram"]:
            hi = f"{b['hi_ms']:g}" if b["hi_ms"] is not None else "inf"
            lines.append(
                f"    [{b['lo_ms']:>5g}, {hi:>5}) ms : {b['count']:>8}  "
                f"({b['share'] * 100:5.1f}%)"
            )
    return lines


def _render_groups(s: dict) -> list[str]:
    lines: list[str] = []
    for key, groups in (s.get("groups") or {}).items():
        lines.append("")
        lines.append(f"  by {key}")
        header = f"    {'value':<14} {'share':>6} {'n':>6}  ttft p50/p90/p99"
        lines.append(header)
        for value, g in groups.items():
            ttft = (g.get("workload_metrics") or {}).get("ttft_s")
            if ttft:
                tt = f"{ttft['p50']:.3f} / {ttft['p90']:.3f} / {ttft['p99']:.3f}"
            else:
                tt = "-"
            lines.append(
                f"    {value:<14} {g['share'] * 100:5.1f}% {g['total_requests']:>6}  {tt}"
            )
    return lines


def _render_lane_lines(s: dict, indent: int) -> list[str]:
    lines: list[str] = []
    pad = " " * indent
    for name, lane in (s.get("lanes") or {}).items():
        lines.append(
            f"{pad}{name}: {lane['total_requests']} requests, "
            f"{lane['throughput_rps']:.2f} req/s, "
            f"{lane['success']} success"
        )
        for metric in ("ttft_s", "itl_ms_mean", "tokens_per_s"):
            values = lane.get("workload_metrics", {}).get(metric)
            if values:
                lines.append(
                    f"{pad}  {metric}: p50={values['p50']:.4f}, "
                    f"p99={values['p99']:.4f}"
                )
    return lines


def _safe_meta(meta: dict) -> dict:
    out = {}
    for k, v in meta.items():
        try:
            json.dumps(v)
            out[k] = v
        except (TypeError, ValueError):
            out[k] = repr(v)
    return out
