"""Per-model endpoint maps (issue #8).

One arm (one ``BenchConfig``) may fan out to several servers — one per
model. ``endpoints: {model_a: url, model_b: url}`` names them; a request's
body ``model`` (set by a lane's ``model`` or carried by the workload item)
selects the endpoint, and capability capture, void rules and the manifest run
per endpoint instead of once.

An endpoint value is either

* a *base URL* (``http://host:8401`` — no path): the request keeps the path
  of ``workload_type.url`` (``/v1/chat/completions``) and only scheme/host
  are swapped; same for the capability URL; or
* a *full URL* (any path): used as-is for requests; or
* a mapping ``{url: ..., capability: ..., headers: ...}`` giving both the
  request URL and the capability URL explicitly.
"""

from __future__ import annotations

from typing import Any, Optional
from urllib.parse import urlsplit, urlunsplit

__all__ = ["normalize_endpoints", "route_url", "endpoint_model"]


def normalize_endpoints(spec: Any) -> dict[str, dict]:
    """``{model: str | {url, capability?, headers?}}`` -> ``{model: {url, capability, headers}}``."""
    if not spec:
        return {}
    if not isinstance(spec, dict):
        raise TypeError("'endpoints' must be a mapping of model -> url | {url, capability, headers}")
    out: dict[str, dict] = {}
    for name, v in spec.items():
        if isinstance(v, str):
            v = {"url": v}
        if not isinstance(v, dict) or not v.get("url"):
            raise ValueError(f"endpoints[{name!r}] needs a url")
        out[str(name)] = {
            "url": str(v["url"]),
            "capability": (str(v["capability"]) if v.get("capability") else None),
            "headers": dict(v.get("headers") or {}),
        }
    return out


def _is_base(url: str) -> bool:
    return urlsplit(url).path in ("", "/")


def route_url(endpoint: str, template: str) -> str:
    """Rewrite ``template`` (e.g. ``http://h:8000/v1/chat/completions``) for an
    endpoint. A base URL swaps scheme/host and keeps the template's path; a
    full URL wins outright."""
    if not endpoint:
        return template
    if not _is_base(endpoint):
        return endpoint
    ep = urlsplit(endpoint)
    tp = urlsplit(template)
    return urlunsplit((ep.scheme, ep.netloc, tp.path, tp.query, tp.fragment))


def endpoint_model(request_json: Any, request_meta: Optional[dict],
                   endpoints: dict[str, Any]) -> Optional[str]:
    """The endpoint key a request is for, or ``None``. Looks at the body
    ``model`` first; in passthrough mode a row's ``model`` only reaches
    ``meta.model_label``, which then selects the endpoint *and* becomes the
    body model (the server behind it serves that name)."""
    if not endpoints:
        return None
    body = request_json if isinstance(request_json, dict) else {}
    m = body.get("model")
    if isinstance(m, str) and m in endpoints:
        return m
    lbl = (request_meta or {}).get("model_label")
    if isinstance(lbl, str) and lbl in endpoints:
        body["model"] = lbl
        return lbl
    return None
