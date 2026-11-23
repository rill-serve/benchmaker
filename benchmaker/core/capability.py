"""Runtime capability capture and void-run rules (issue #4).

Two runs that share a config hash can still be incomparable: a quantized
kernel silently falling back, a decode backend that is not the configured one,
a different served-model revision. Benchmaker therefore records the endpoint's
self-description before and after a run ("capability snapshots") and lets a
run declare *void rules* — assertions on those snapshots that, when violated,
mark the run void so it is excluded from comparisons instead of polluting them.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Optional


# --------------------------------------------------------------------------- #
# Fetching                                                                     #
# --------------------------------------------------------------------------- #


def fetch_json(url: str, timeout_s: float = 10.0,
               headers: Optional[dict[str, str]] = None) -> dict:
    """GET ``url`` and return ``{"ok", "url", "data"|"error"}`` (never raises).

    Uses the stdlib client on purpose: capability capture runs once before and
    once after a benchmark, outside the hot path, and must not depend on the
    benchmark client's connection pool or event loop.
    """
    req = urllib.request.Request(url, headers=dict(headers or {}))
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:  # noqa: S310 (trusted lab endpoints)
            raw = resp.read()
        data = json.loads(raw.decode("utf-8", errors="replace"))
        return {"ok": True, "url": url, "data": data}
    except urllib.error.HTTPError as e:
        return {"ok": False, "url": url, "error": f"HTTP {e.code}", "status": e.code}
    except Exception as e:  # URLError, socket timeout, JSON decode, ...
        return {"ok": False, "url": url, "error": f"{type(e).__name__}: {e}"}


def normalize_capability(doc: Any) -> dict:
    """Normalize a server self-description.

    ``/v1/models`` answers ``{"object": "list", "data": [{"id": ...}]}``; it is
    turned into ``{"models": [...ids...], "models_detail": [...]}`` so a rule
    can say ``models contains "my-model"``. Anything else (vLLM/SGLang
    ``get_server_info``, ``get_model_info``, ...) is returned as-is.
    """
    if isinstance(doc, dict) and doc.get("object") == "list" and isinstance(doc.get("data"), list):
        ids = [m.get("id") for m in doc["data"] if isinstance(m, dict)]
        return {"models": ids, "models_detail": doc["data"]}
    return doc if isinstance(doc, dict) else {"value": doc}


# --------------------------------------------------------------------------- #
# Void rules                                                                   #
# --------------------------------------------------------------------------- #

OPS = ("empty", "eq", "ne", "lt", "le", "gt", "ge", "in", "not_in",
       "contains", "not_contains")


@dataclass
class VoidRule:
    """One assertion about a capability snapshot.

    ``at``: ``"start"``, ``"end"`` or ``"both"``. ``path``: dotted path into
    the snapshot (numeric segments index lists). ``op``: see :data:`OPS`.
    ``ignore_missing``: a missing path does not count as a violation.
    """

    name: str
    path: str
    op: str = "empty"
    value: Any = None
    at: str = "end"
    ignore_missing: bool = False

    def __post_init__(self) -> None:
        if self.op not in OPS:
            raise ValueError(f"void rule {self.name!r}: unknown op {self.op!r} (expected one of {OPS})")
        if self.at not in ("start", "end", "both"):
            raise ValueError(f"void rule {self.name!r}: 'at' must be start, end or both")

    @classmethod
    def from_dict(cls, d: dict) -> "VoidRule":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"void rule {d.get('name')!r}: unknown fields {sorted(unknown)}")
        return cls(**d)


def parse_void_rules(rules: Any) -> list[VoidRule]:
    """Accept a list of dicts (``name`` required) or ``{"name": ..., ...}``."""
    if not rules:
        return []
    if isinstance(rules, dict):
        rules = [rules]
    if not isinstance(rules, list):
        raise ValueError("void_rules must be a list of rule objects")
    out = [VoidRule.from_dict(dict(r)) for r in rules]
    names = [r.name for r in out]
    if len(names) != len(set(names)):
        raise ValueError(f"void rule names must be unique: {names}")
    return out


def _lookup(doc: Any, path: str) -> tuple[bool, Any]:
    """Dotted path lookup; returns ``(found, value)``."""
    cur = doc
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        elif isinstance(cur, list) and part.lstrip("-").isdigit():
            idx = int(part)
            if -len(cur) <= idx < len(cur):
                cur = cur[idx]
            else:
                return False, None
        else:
            return False, None
    return True, cur


def _is_empty(v: Any) -> bool:
    return v is None or v == "" or v == [] or v == {}


def check_rule(rule: VoidRule, snapshot: Optional[dict]) -> Optional[dict]:
    """Evaluate one rule against one snapshot.

    Returns ``None`` when the rule holds (or is skipped), else a violation
    record. ``snapshot`` is the *fetch result* (``fetch_json`` output).
    """
    if snapshot is None:
        return _violation(rule, None, "no snapshot")
    if not snapshot.get("ok"):
        return _violation(rule, None, f"snapshot unavailable: {snapshot.get('error')}")
    doc = normalize_capability(snapshot.get("data"))
    found, actual = _lookup(doc, rule.path)
    if not found:
        if rule.ignore_missing:
            return None
        return _violation(rule, None, "path missing")
    ok, reason = _apply_op(rule.op, actual, rule.value)
    if ok:
        return None
    return _violation(rule, actual, reason)


def _apply_op(op: str, actual: Any, expected: Any) -> tuple[bool, Optional[str]]:
    try:
        if op == "empty":
            return (_is_empty(actual), None) if _is_empty(actual) else (False, "not empty")
        if op == "eq":
            return (actual == expected, None)
        if op == "ne":
            return (actual != expected, None)
        if op in ("lt", "le", "gt", "ge"):
            a, b = float(actual), float(expected)  # type: ignore[arg-type]
            return {"lt": a < b, "le": a <= b, "gt": a > b, "ge": a >= b}[op], None
        if op == "in":
            # a list actual is "every element is expected" (e.g. models in [...])
            if isinstance(actual, (list, tuple, set)):
                return set(actual) <= set(expected or []), None
            return (actual in (expected or []), None)
        if op == "not_in":
            if isinstance(actual, (list, tuple, set)):
                return not (set(actual) & set(expected or [])), None
            return (actual not in (expected or []), None)
        if op == "contains":
            return (expected in actual, None) if hasattr(actual, "__contains__") else (False, "not a container")
        if op == "not_contains":
            return (expected not in actual, None) if hasattr(actual, "__contains__") else (True, None)
    except (TypeError, ValueError) as e:
        return False, f"{type(e).__name__}: {e}"
    return False, f"unknown op {op}"


def _violation(rule: VoidRule, actual: Any, reason: str) -> dict:
    return {
        "rule": rule.name, "at": rule.at, "path": rule.path, "op": rule.op,
        "expected": rule.value, "actual": actual, "reason": reason,
    }


@dataclass
class VoidVerdict:
    void: bool = False
    violations: list[dict] = field(default_factory=list)
    checks: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "void": self.void,
            "violations": self.violations,
            "checks": self.checks,
        }


def evaluate_void_rules(rules: list[VoidRule], start: Optional[dict],
                        end: Optional[dict]) -> VoidVerdict:
    """Evaluate every rule at its declared snapshot(s)."""
    verdict = VoidVerdict()
    for rule in rules:
        for at, snap in (("start", start), ("end", end)):
            if rule.at not in (at, "both"):
                continue
            v = check_rule(rule, snap)
            if v is not None and rule.at == "both":
                v["at"] = at  # which of the two snapshots violated
            verdict.checks.append({
                "rule": rule.name, "at": at, "path": rule.path, "op": rule.op,
                "expected": rule.value, "ok": v is None,
            })
            if v is not None:
                verdict.violations.append(v)
    verdict.void = bool(verdict.violations)
    return verdict


# --------------------------------------------------------------------------- #
# Served-model identity (for the manifest, issue #4)                           #
# --------------------------------------------------------------------------- #

# Keys a server self-description may use for a revision identifier. Model
# *name* keys are deliberately absent: a name is not a build identity.
_REVISION_KEYS = ("model_revision", "revision", "commit_hash", "commit", "git_commit",
                  "model_path", "checkpoint", "build")
_KNOWN_PLACEHOLDERS = {"main", "master", "HEAD", "latest", "unknown", "None", ""}


def extract_model_identity(snapshot: Optional[dict]) -> dict:
    """Best-effort ``{"model": ..., "revision": ..., "source": ...}`` from a snapshot.

    A snapshot revision is reported with ``source: "capability"`` only when it
    looks like a real identifier (a 7-40 char hex commit, a version tag or a
    non-placeholder name); ``"main"``/``"latest"`` are recorded as-is but not
    treated as a revision, because two different builds can both say ``main``.
    """
    out: dict[str, Any] = {"model": None, "revision": None, "source": None}
    if not snapshot or not snapshot.get("ok"):
        return out
    doc = normalize_capability(snapshot.get("data"))
    if not isinstance(doc, dict):
        return out
    for k in ("model", "model_name", "served_model_name", "model_id"):
        v = doc.get(k)
        if isinstance(v, str) and v:
            out["model"] = v
            break
    if out["model"] is None and isinstance(doc.get("models"), list) and len(doc["models"]) == 1:
        out["model"] = doc["models"][0]
    for k in _REVISION_KEYS:
        v = doc.get(k)
        if isinstance(v, str) and v and v not in _KNOWN_PLACEHOLDERS and v != out["model"]:
            out["revision"] = v
            out["source"] = f"capability:{k}"
            break
    return out
