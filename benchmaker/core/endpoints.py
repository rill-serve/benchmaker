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
* a mapping ``{url: ..., model: ..., capability: ..., headers: ...}`` giving
  the request URL, the model name to put in the body when the key is an
  alias, and the capability URL explicitly.

Which endpoint a request goes to (``endpoint_model``): with ``endpoints:``
set, a request whose model names no endpoint is an error (``RoutingError``,
issue #15) — it is not silently sent to ``workload_type.url`` — and the rep
it belongs to is voided.
"""

from __future__ import annotations

from typing import Any, Optional
from urllib.parse import urlsplit, urlunsplit, urljoin

__all__ = ["normalize_endpoints", "route_url", "endpoint_model", "default_capability_url", "RoutingError"]


class RoutingError(ValueError):
    """A request whose model has no entry in ``endpoints:`` (issue #15)."""


def normalize_endpoints(spec: Any) -> dict[str, dict]:
    """``{model: str | {url, model?, capability?, headers?}}`` -> ``{model: {url, model, capability, headers}}``."""
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
        unknown = set(v) - {"url", "capability", "headers", "model"}
        if unknown:
            raise ValueError(f"endpoints[{name!r}]: unknown field(s) {sorted(unknown)}; "
                             "expected url, model, capability, headers")
        out[str(name)] = {
            "url": str(v["url"]),
            # served model name when the key is an alias (``small: {url, model: qwen2.5-7b}``)
            "model": (str(v["model"]) if v.get("model") else None),
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


def default_capability_url(url: str) -> str:
    """The OpenAI-style ``models`` listing next to a request URL: ``/v1/chat/completions``
    and ``/v1/completions`` -> ``/v1/models`` (keeping any path prefix);
    anything else -> ``<origin>/v1/models``."""
    parts = urlsplit(url)
    path = parts.path.rstrip("/")
    for suffix in ("/chat/completions", "/completions"):
        if path.endswith(suffix):
            return urlunsplit((parts.scheme, parts.netloc, path[: -len(suffix)] + "/models", "", ""))
    return urljoin(url, "/v1/models")


def endpoint_model(request_json: Any, request_meta: Optional[dict],
                   endpoints: dict[str, Any]) -> Optional[str]:
    """The endpoint key a request is for, or ``None`` (unroutable).

    Precedence: a lane's explicit ``model`` (``apply_overrides`` stamps it on
    ``meta.model``) > the row's own ``model_label`` (passthrough / session
    replay rows, which name the server the row was recorded for) > the body
    ``model`` (the workload type's default, which may itself be an endpoint
    key). Routing by label rewrites the body model — to the endpoint's
    ``model`` when the key is an alias, else to the key (the server behind it
    serves that name)."""
    if not endpoints:
        return None
    body = request_json if isinstance(request_json, dict) else {}
    meta = request_meta or {}
    m = meta.get("model")
    if isinstance(m, str) and m in endpoints:
        return m
    lbl = meta.get("model_label")
    if isinstance(lbl, str) and lbl in endpoints:
        body["model"] = endpoints[lbl].get("model") or lbl
        return lbl
    m = body.get("model")
    if isinstance(m, str) and m in endpoints:
        return m
    return None


def unroutable_model(request_json: Any, request_meta: Optional[dict]) -> Any:
    """The model name a request wanted (for the error message) — same
    precedence as ``endpoint_model``."""
    meta = request_meta or {}
    for v in (meta.get("model"), meta.get("model_label"),
              (request_json or {}).get("model") if isinstance(request_json, dict) else None):
        if v is not None:
            return v
    return None
