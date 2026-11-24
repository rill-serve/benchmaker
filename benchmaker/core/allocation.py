"""Normalize endpoint KV allocation with provenance for each reported value."""

from __future__ import annotations

import logging
import re
from decimal import Decimal, InvalidOperation
from math import isfinite
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

GIB = 1024**3
GB = 10**9
ALLOCATION_FIELDS = {
    "device_kv_tokens": "tokens",
    "host_tier_bytes": "bytes",
    "host_tier_tokens": "tokens",
    "bytes_per_token": "bytes/token",
    "sharing": "identity",
    "fleet_host_total_bytes": "bytes",
}

_NUMBER = r"[0-9]+(?:\.[0-9]+)?"
_SIZE_FIELDS = {
    "kv_offloading_size": GIB,
    "max_local_cpu_size": GIB,
    "hicache_size": GB,
    "ram_gb": GIB,
    "ram_quota_per_model_gb": GIB,
}
_KVD_LOG = re.compile(r"\brill-kvd: serving (.+?) \(ram_gb=")
_DEVICE_LOG = re.compile(
    r"(?:GPU KV cache size:\s*|(?:KV Cache is allocated|Post-capture KV sizing: KV cache allocated)"
    r"[^\n]*?#tokens:\s*)([0-9][0-9,]*)",
    re.IGNORECASE,
)
_HOST_LOG = re.compile(
    r"Allocating (?:\w+[ -])*hierarchical KV host pool:\s*([0-9][0-9,]*) tokens"
    rf"(?:,\s*({_NUMBER}) GB host memory)?",
    re.IGNORECASE,
)


def normalize_allocation_config(spec: Any, endpoint_names: set[str]) -> dict:
    """Validate the log and conversion inputs for a run or its named endpoints."""
    if spec is None:
        return {}
    if not isinstance(spec, dict):
        raise ValueError("allocation must be a mapping")
    options = {"startup_log", "kvd_log", "bytes_per_token", "sharing"}
    unknown = spec.keys() - options - {"endpoints"}
    if unknown:
        raise ValueError(f"unknown allocation fields {sorted(unknown)}")
    per = spec.get("endpoints", {})
    if not isinstance(per, dict) or per.keys() - endpoint_names:
        raise ValueError("allocation.endpoints must name configured endpoints")
    for name, entry in [(None, spec), *per.items()]:
        if not isinstance(entry, dict) or (name is not None and entry.keys() - options):
            raise ValueError(f"invalid allocation options for endpoint {name!r}")
        for key in ("startup_log", "kvd_log"):
            if key in entry and (not isinstance(entry[key], str) or not entry[key]):
                raise ValueError(f"allocation.{key} must be a path")
        if "sharing" in entry and not _sharing(entry["sharing"]):
            raise ValueError("allocation.sharing must be private or shared:<group>")
        if "bytes_per_token" in entry and not _number(entry["bytes_per_token"]):
            raise ValueError("allocation.bytes_per_token must be positive")
    return spec


def _sharing(value: Any) -> bool:
    return value == "private" or (
        isinstance(value, str) and value.startswith("shared:") and bool(value[7:].strip())
    )


def _number(value: Any) -> Decimal | None:
    try:
        result = Decimal(str(value).replace(",", ""))
    except (ValueError, InvalidOperation):
        return None
    return result if result.is_finite() and result >= 0 else None


def _put(dest: dict, field: str, raw: Any, source: str, *, factor: int = 1,
         unit: str | None = None, approximate: bool = False) -> None:
    number = _number(raw)
    if number is None:
        return
    value = float(number) if field == "bytes_per_token" else int(number * factor)
    if field == "bytes_per_token" and (not isfinite(value) or value <= 0):
        return
    record = {"source": source, "unit": unit or ALLOCATION_FIELDS[field]}
    if unit:
        record["value"] = float(number)
    if approximate:
        record["approximate"] = True
    dest[field] = (value, record)


def _put_sharing(dest: dict, sharing: str, source: str) -> None:
    dest["sharing"] = (sharing, {"source": source, "unit": "identity"})


def _host_size(dest: dict, key: str, value: Any, source: str) -> None:
    field = "fleet_host_total_bytes" if key == "ram_gb" else "host_tier_bytes"
    factor = _SIZE_FIELDS[key]
    _put(dest, field, value, source, factor=factor, unit="GiB" if factor == GIB else "GB")


def _read_logs(path: str) -> tuple[dict, dict]:
    """Return engine fields and any single identifiable kvd pool in this log."""
    engine: dict = {}
    pools: dict[str, dict] = {}
    sizes: dict = {}
    try:
        with Path(path).open(encoding="utf-8", errors="replace") as stream:
            for lineno, line in enumerate(stream, 1):
                source = f"log:{path}:{lineno}"
                if match := _DEVICE_LOG.search(line):
                    _put(engine, "device_kv_tokens", match[1].rstrip(","), source)
                if match := _HOST_LOG.search(line):
                    _put(engine, "host_tier_tokens", match[1], source)
                    if match[2] is not None:
                        _put(engine, "host_tier_bytes", match[2], source,
                             factor=GB, unit="GB", approximate=True)
                kvd = _KVD_LOG.search(line)
                dest = pools.setdefault(kvd[1], {}) if kvd else engine
                if kvd:
                    _put_sharing(dest, f"shared:{kvd[1]}", source)
                for key in _SIZE_FIELDS:
                    value = re.search(
                        rf"(?<![\w-]){key}['\"]?\s*[:=]\s*['\"]?({_NUMBER})\b"
                        rf"|--{key.replace('_', '-')}[= ]+({_NUMBER})\b", line,
                    )
                    if value:
                        _host_size(dest if kvd else sizes, key, value[1] or value[2], f"{source}:{key}")
                if "LMCache Configuration" in line and re.search(
                    r"['\"]?local_cpu['\"]?\s*[:=]\s*false\b", line, re.IGNORECASE,
                ):
                    _host_size(sizes, "max_local_cpu_size", 0, f"{source}:local_cpu")
    except OSError as exc:
        logger.warning("cannot read allocation log %s: %s", path, exc)
    # Several daemon sockets in one log cannot identify an endpoint's pool.
    return {**sizes, **engine}, next(iter(pools.values())) if len(pools) == 1 else {}


def _snapshot_fields(snapshots: dict) -> dict:
    values: dict = {}
    for when in ("end", "start"):
        snapshot = snapshots.get(when) or {}
        doc = snapshot.get("data")
        if not snapshot.get("ok") or not isinstance(doc, dict):
            continue
        kv_info = doc.get("kv_cache_info")
        kvd = doc.get("kvd")
        docs = [("", doc)]
        if isinstance(kv_info, dict):
            docs.append(("kv_cache_info.", kv_info))
        if isinstance(kvd, dict):
            docs.append(("kvd.", kvd))
        for prefix, fields in docs:
            source = f"capability:{when}:{snapshot.get('url') or ''}#{prefix}"
            for key in _SIZE_FIELDS:
                _host_size(values, key, fields.get(key), source + key)
            if fields.get("local_cpu") is False and "max_local_cpu_size" in fields:
                _host_size(values, "max_local_cpu_size", 0, source + "local_cpu")
            for key in ALLOCATION_FIELDS.keys() - {"sharing"}:
                _put(values, key, fields.get(key), source + key)
            for field in ("kv_capacity_tokens", "max_total_num_tokens"):
                _put(values, "device_kv_tokens", fields.get(field), source + field)
            sharing = fields.get("sharing")
            if _sharing(sharing):
                _put_sharing(values, sharing, source + "sharing")
            elif prefix == "kvd." and isinstance(fields.get("socket"), str) and fields["socket"]:
                _put_sharing(values, f"shared:{fields['socket']}", source + "socket")
    return values


def _derive(fields: dict) -> None:
    host_bytes = fields.get("host_tier_bytes")
    host_tokens = fields.get("host_tier_tokens")
    bpt = fields.get("bytes_per_token")
    if bpt is None and host_bytes and host_tokens and host_bytes[0] > 0 and host_tokens[0] > 0:
        fields["bytes_per_token"] = (
            host_bytes[0] / host_tokens[0],
            {"source": "derived:host_tier_bytes/host_tier_tokens", "unit": "bytes/token",
             "approximate": bool(host_bytes[1].get("approximate"))},
        )
    elif bpt and host_bytes and not host_tokens:
        fields["host_tier_tokens"] = (
            int(Decimal(host_bytes[0]) / Decimal(str(bpt[0]))),
            {"source": "derived:host_tier_bytes/bytes_per_token", "unit": "tokens",
             "approximate": bool(host_bytes[1].get("approximate"))},
        )
    elif bpt and host_tokens and not host_bytes:
        fields["host_tier_bytes"] = (
            int(Decimal(host_tokens[0]) * Decimal(str(bpt[0]))),
            {"source": "derived:host_tier_tokens*bytes_per_token", "unit": "bytes"},
        )


def _record(fields: dict) -> dict:
    return {
        **{field: fields.get(field, (None,))[0] for field in ALLOCATION_FIELDS},
        "sources": {field: record for field, (_, record) in fields.items()},
    }


def _fleet_total(allocations: dict[str, dict]) -> tuple[int, dict] | None:
    pools: dict[str, set[int]] = {}
    for endpoint, allocation in allocations.items():
        sharing = allocation["sharing"]
        if sharing == "private":
            group = f"private:{endpoint}"
            capacity = allocation["host_tier_bytes"]
        elif sharing:
            group = sharing
            capacity = allocation["fleet_host_total_bytes"]
        else:
            return None
        pools.setdefault(group, set())
        if capacity is not None:
            pools[group].add(capacity)
    if not pools or any(len(totals) != 1 for totals in pools.values()):
        return None
    totals = {group: next(iter(sizes)) for group, sizes in pools.items()}
    return sum(totals.values()), {
        "source": "sum:distinct_host_pools", "unit": "bytes", "pools": totals,
        "approximate": any(
            (allocation["sources"].get(field) or {}).get("approximate")
            for allocation in allocations.values()
            for field in ("host_tier_bytes", "fleet_host_total_bytes")
        ),
    }


def build_allocation(
    spec: dict, capability: dict, endpoints: dict, *, launch_log: str | None = None,
) -> tuple[dict, dict[str, dict]]:
    """Build arm and endpoint records from snapshots, the launch, and scoped logs."""
    logged: dict[str, tuple[dict, dict]] = {}

    def logs(path: str | None) -> tuple[dict, dict]:
        if not path:
            return {}, {}
        if path not in logged:
            logged[path] = _read_logs(path)
        return logged[path]

    def daemon_log(path: str) -> dict:
        engine, kvd = logs(path)
        if kvd or "fleet_host_total_bytes" not in engine:
            return kvd
        fields = {k: v for k, v in engine.items() if k in {"fleet_host_total_bytes", "host_tier_bytes"}}
        _put_sharing(fields, f"shared:{path}", f"log:{path}")
        return fields

    base_engine, base_kvd = logs(spec.get("startup_log", launch_log))
    if endpoints and not spec.get("startup_log"):
        # The managed launch hook can collect output from several engines.
        # A shared daemon must be anchored by a configured log in that case.
        base_kvd = {}
    if spec.get("kvd_log"):
        base_kvd = daemon_log(spec["kvd_log"])
    per = spec.get("endpoints", {})
    records = {}
    for name in endpoints or {None: None}:
        options = {**spec, **per.get(name, {})}
        scoped = per.get(name, {})
        engine, kvd = logs(scoped.get("startup_log")) if endpoints else (base_engine, base_kvd)
        if scoped.get("kvd_log"):
            kvd = daemon_log(scoped["kvd_log"])
        # A global startup log can describe a shared daemon, but model-specific
        # engine metrics must come from a capability snapshot or a scoped log.
        snaps = (capability.get("endpoints") or {}).get(name) if endpoints else capability
        daemon = kvd
        if not daemon and not scoped.get("kvd_log") and (
            options.get("sharing") != "private" or not endpoints
        ):
            daemon = base_kvd
        fields = {**daemon, **engine, **_snapshot_fields(snaps or {})}
        if "bytes_per_token" in options:
            _put(fields, "bytes_per_token", options["bytes_per_token"], "config:bytes_per_token")
        if "sharing" in options:
            _put_sharing(fields, options["sharing"], "config:sharing")
        elif "sharing" not in fields:
            if "fleet_host_total_bytes" in fields and not endpoints:
                _put_sharing(fields, "private", "single_endpoint")
            elif "fleet_host_total_bytes" not in fields and (
                "host_tier_bytes" in fields or "host_tier_tokens" in fields
            ):
                _put_sharing(fields, "private", "endpoint_host_pool")
        sharing = fields.get("sharing", (None,))[0]
        if sharing == "private" and "fleet_host_total_bytes" in fields:
            fields.setdefault("host_tier_bytes", fields["fleet_host_total_bytes"])
        if "host_tier_bytes" in fields and "fleet_host_total_bytes" in fields:
            # A daemon quota larger than its pool cannot be used by one model.
            if fields["host_tier_bytes"][0] > fields["fleet_host_total_bytes"][0]:
                fields["host_tier_bytes"] = fields["fleet_host_total_bytes"]
        _derive(fields)
        records[name] = _record(fields)

    fleet = _fleet_total(records)
    if not endpoints:
        arm = records[None]
        if fleet and arm["fleet_host_total_bytes"] is None:
            arm["fleet_host_total_bytes"], arm["sources"]["fleet_host_total_bytes"] = fleet
        return arm, {}

    arm = _record({"fleet_host_total_bytes": fleet} if fleet else {})
    sharing = {record["sharing"] for record in records.values()}
    if len(sharing) == 1 and None not in sharing:
        arm["sharing"] = next(iter(sharing))
        arm["sources"]["sharing"] = {"source": "endpoints", "unit": "identity"}
    return arm, records
