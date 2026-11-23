"""Run manifest — a positive record of *what was actually benchmarked* (issue #4).

``meta.json`` says when a run happened and what its source config was. The
manifest adds the identity a reader needs before trusting a comparison:

* ``config_hash``      — hash of the resolved run configuration (not the YAML
                         text, which may contain env substitutions).
* ``prompt_set``       — content hash of the exact prompt/input set, so two
                         runs that both say ``dataset: sharegpt`` can be checked
                         for having replayed the same items.
* ``model``            — served model id and, when known, revision, with the
                         source of the revision (explicit config or the server's
                         capability snapshot).
* ``host``              — hostname and job id (Slurm/PBS/LSF/K8s), so reps that
                         silently landed on different nodes can be detected.
* ``env``               — a filtered subset of environment variables (server
                         knobs for in-process servers, CUDA device masks, arm
                         overrides).
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import socket
import subprocess
from typing import Any, Optional

from benchmaker.core.capability import extract_model_identity

MANIFEST_VERSION = 1
MANIFEST_FILENAME = "manifest.json"

# Environment variables recorded by default. Globs are allowed. These are the
# knobs that change what a server does without showing up in the request.
DEFAULT_RECORD_ENV = ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "VLLM_*", "SGLANG_*", "NCCL_*IB*")

# (variable, formatter) pairs probed in order for a scheduler job id.
_JOB_ID_SOURCES: tuple[tuple[str, Any], ...] = (
    ("SLURM_JOB_ID", None),
    ("SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID"),   # rendered as "<job>_<task>"
    ("PBS_JOBID", None),
    ("LSB_JOBID", None),                              # LSF
    ("KUBERNETES_JOB_ID", None),
    ("RILL_JOB_ID", None),
)


def canonical_json(obj: Any) -> str:
    """Deterministic JSON text (sorted keys, no whitespace, str() for unknown)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_json_default)


def config_hash(cfg: Any) -> str:
    """``sha256:<hex>`` of the canonical JSON of ``cfg``.

    Accepts the resolved config dict written to ``meta.source_config`` or any
    object exposing ``to_dict()``.
    """
    if hasattr(cfg, "to_dict"):
        cfg = cfg.to_dict()
    return "sha256:" + hashlib.sha256(canonical_json(cfg).encode("utf-8")).hexdigest()


def job_identity(env: Optional[dict[str, str]] = None) -> dict:
    """``{"job_id": ..., "source": ...}``; falls back to the hostname."""
    env = os.environ if env is None else env
    for var, companion in _JOB_ID_SOURCES:
        val = env.get(var)
        if val:
            if companion and env.get(companion):
                return {"job_id": f"{val}_{env[companion]}", "source": f"{var}+{companion}"}
            return {"job_id": val, "source": var}
    return {"job_id": socket.gethostname(), "source": "hostname"}


def host_identity() -> dict:
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "pid": os.getpid(),
        "cpu_count": os.cpu_count(),
    }


def filter_env(record: list[str], env: Optional[dict[str, str]] = None) -> dict[str, str]:
    """Return env vars whose name matches any glob in ``record``."""
    env = os.environ if env is None else env
    patterns = [re.compile("^" + re.escape(p).replace(r"\*", ".*") + "$") for p in record]
    return {k: v for k, v in sorted(env.items()) if any(p.match(k) for p in patterns)}


def git_commit_of(path: str) -> Optional[str]:
    """Best-effort HEAD commit of the repository containing ``path``."""
    try:
        out = subprocess.run(
            ["git", "-C", path, "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        commit = out.stdout.strip()
        return commit or None
    except Exception:
        return None


def build_manifest(
    *,
    source_config: Any,
    run_id: str,
    workload_type: Any = None,
    workload: Any = None,
    workload_name: str = "",
    labels: Optional[dict] = None,
    started_at: Optional[float] = None,
    model_revision: Optional[str] = None,
    capability: Optional[dict] = None,
    env: Optional[dict[str, str]] = None,
    record_env: Optional[list[str]] = None,
    benchmaker_version: str = "",
) -> dict:
    """Assemble the manifest dict. See module docstring for the fields."""
    import time as _time

    wt_model = getattr(workload_type, "model", None)
    cap = capability or {}
    # Identity from the start snapshot first, falling back to the end one (a
    # revision the server only reports after warm-up is still an identity).
    cap_identity = {
        **extract_model_identity(cap.get("end")),
        **{k: v for k, v in extract_model_identity(cap.get("start")).items() if v is not None},
    }

    revision: Optional[str] = None
    revision_source: Optional[str] = None
    if model_revision:
        revision, revision_source = model_revision, "config"
    elif cap_identity.get("revision"):
        revision, revision_source = cap_identity["revision"], cap_identity["source"]

    prompt_set: dict[str, Any] = {"name": workload_name}
    content_hash = getattr(workload, "content_hash", None)
    if callable(content_hash):
        try:
            prompt_set["hash"] = content_hash()
        except Exception:
            prompt_set["hash"] = None
    n_items = getattr(workload, "item_count", None)
    if callable(n_items):
        try:
            prompt_set["items"] = n_items()
        except Exception:
            prompt_set["items"] = None

    manifest: dict[str, Any] = {
        "manifest_version": MANIFEST_VERSION,
        "benchmaker_version": benchmaker_version,
        "run_id": run_id,
        "started_at": started_at,
        "config_hash": config_hash(source_config),
        "prompt_set": prompt_set,
        "model": {
            "id": wt_model or cap_identity.get("model"),
            "revision": revision,
            "revision_source": revision_source,
            "capability_model": cap_identity.get("model"),
        },
        "host": {**host_identity(), **job_identity()},
        "env": {
            **filter_env(list(record_env or DEFAULT_RECORD_ENV)),
            **(env or {}),
        },
        "labels": dict(labels or {}),
    }
    commit = git_commit_of(os.path.dirname(os.path.abspath(__file__)))
    if commit:
        manifest["benchmaker_git_commit"] = commit
    if isinstance(started_at, (int, float)):
        from datetime import datetime, timezone
        manifest["started_at"] = datetime.fromtimestamp(started_at, tz=timezone.utc).isoformat()
    manifest["recorded_at"] = _time.time()
    return manifest


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    if hasattr(obj, "to_dict"):
        try:
            return obj.to_dict()
        except Exception:
            pass
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in sorted(vars(obj).items()) if not k.startswith("_")}
    return repr(obj)
