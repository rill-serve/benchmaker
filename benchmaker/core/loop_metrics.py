"""Estimate closed-loop replay load from configured think times and service intervals."""

from collections import defaultdict
from statistics import mean

from benchmaker.core.types import Sample

LOW_UTILISATION = 0.5
RATE_TOLERANCE = 0.05


def _busy_seconds(samples: list[Sample], start: float, end: float) -> float:
    intervals = []
    for sample in samples:
        request_start = sample.meta.get("service_start_ts", sample.start_ts)
        intervals.append((max(start, request_start), min(end, request_start + sample.latency_s)))
    busy = 0.0
    covered_until = start
    for left, right in sorted(intervals):
        busy += max(0.0, right - max(left, covered_until))
        covered_until = max(right, covered_until)
    return busy


def closed_loop_summary(
    samples: list[Sample], start: float, end: float, *, throughput_rps: float
) -> dict:
    if not samples or any("replay_concurrent_sessions" not in s.meta for s in samples):
        return {}
    # Parallel lanes offer additive load. Sequential phases each have their own window.
    if len({s.meta.get("phase") for s in samples}) != 1:
        return {}
    lanes = defaultdict(list)
    for sample in samples:
        lanes[sample.meta.get("lane")].append(sample)
    sessions = 0
    offered = 0.0
    for lane in lanes.values():
        slots = lane[0].meta["replay_concurrent_sessions"]
        cycle = mean(s.meta["replay_think_s"] + s.latency_s for s in lane)
        if cycle <= 0:
            return {}
        sessions += slots
        offered += slots / cycle
    think = mean(s.meta["replay_think_s"] for s in samples)
    service = mean(s.latency_s for s in samples)
    wall = max(end - start, 1e-9)
    busy = _busy_seconds(samples, start, end)
    utilisation = busy / wall
    endpoints = defaultdict(list)
    for sample in samples:
        endpoints[sample.meta.get("endpoint") or "(default)"].append(sample)
    endpoint_utilisation = {
        endpoint: _busy_seconds(bucket, start, end) / wall
        for endpoint, bucket in endpoints.items()
    }
    return {
        "sessions": sessions,
        "mean_think_s": think,
        "mean_service_s": service,
        "offered_load_rps": offered,
        "busy_time_s": busy,
        "utilisation": utilisation,
        "endpoint_utilisation": endpoint_utilisation,
        "loop_bound": think > 0 and (
            max(endpoint_utilisation.values()) < LOW_UTILISATION
            or abs(throughput_rps - offered) / offered <= RATE_TOLERANCE
        ),
    }
