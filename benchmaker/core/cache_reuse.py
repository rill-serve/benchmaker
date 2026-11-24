"""First-touch and retouch attribution for requests with a workload reuse key."""

from collections import Counter

from benchmaker.core.types import Sample


def _scope(sample: Sample) -> tuple:
    meta = sample.meta
    return meta.get("endpoint"), meta.get("endpoint_model") or meta.get("model")


def annotate_reuse(samples: list[Sample]) -> None:
    """Track touches across phases and warmup, in request dispatch order."""
    from benchmaker.workloads.llm import hint_tier

    seen = set()
    seen_by_tier = set()
    keyed = [s for s in samples if isinstance(s.meta.get("reuse_key"), (str, int))]
    for sample in sorted(keyed, key=lambda s: s.meta.get("service_start_ts", s.start_ts)):
        meta = sample.meta
        key = (*_scope(sample), meta["reuse_key"])
        cls = meta.get("cache_class")
        if not sample.request_ok:
            continue
        touch = "retouch" if key in seen else "first_touch"
        seen.add(key)
        if cls not in {"miss", "partial", "device_hit", "restore", "unknown"}:
            continue
        meta["reuse_touch"] = touch
        if cls == "miss":
            meta["cache_miss_kind"] = meta["reuse_touch"]
        tier = hint_tier(meta.get("cache_hint"))
        if tier in {"device", "restore"}:
            tier_key = (*key, tier)
            meta["reuse_tier"] = tier
            meta["reuse_tier_touch"] = "retouch" if tier_key in seen_by_tier else "first_touch"
            seen_by_tier.add(tier_key)
            if cls == "miss":
                meta["cache_tier_miss_kind"] = meta["reuse_tier_touch"]


def _counts(samples: list[Sample], touch_key: str, miss_key: str) -> dict:
    touch = Counter(s.meta[touch_key] for s in samples)
    misses = Counter(s.meta.get(miss_key) for s in samples)
    return {
        "requests": len(samples),
        "first_touch_requests": touch["first_touch"],
        "retouch_requests": touch["retouch"],
        "first_touch_misses": misses["first_touch"],
        "retouch_misses": misses["retouch"],
        "eviction_miss_share": misses["retouch"] / len(samples),
    }


def reuse_summary(samples: list[Sample]) -> dict:
    keyed = [s for s in samples if "reuse_touch" in s.meta]
    if not keyed:
        return {}
    scopes: dict[tuple, list[Sample]] = {}
    for s in keyed:
        scopes.setdefault(_scope(s), []).append(s)
    out = _scoped_counts(keyed)
    out["by_endpoint_model"] = [
        {"endpoint": endpoint, "model": model, **_scoped_counts(bucket)}
        for (endpoint, model), bucket in scopes.items()
    ]
    return out


def _scoped_counts(samples: list[Sample]) -> dict:
    out = _counts(samples, "reuse_touch", "cache_miss_kind")
    out["by_tier"] = {
        tier: _counts(bucket, "reuse_tier_touch", "cache_tier_miss_kind")
        for tier in ("device", "restore")
        if (bucket := [s for s in samples if s.meta.get("reuse_tier") == tier])
    }
    return out
