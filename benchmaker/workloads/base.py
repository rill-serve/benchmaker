"""Workload-type protocol.

A `WorkloadType` knows how to talk to a specific *kind* of HTTP service
(generic HTTP, OpenAI chat completions, …). It does NOT own the inputs —
inputs come from a `Workload` (see `benchmaker.workloads.datasets`), which
yields opaque per-request items.

Pairing:
    bench = WorkloadType("how to talk") + Workload("what to send") + LoadModel("when")
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Optional

from benchmaker.core.types import Request, Response, Sample, TicketContext, maybe_await

_logger = logging.getLogger(__name__)


class WorkloadType(ABC):
    """A request protocol/shape.

    Subclasses turn a workload item (any object yielded by a `Workload`) into a
    `Request`, and the resulting `Response` into a `Sample`. Workload-specific
    metrics (TTFT, tokens/sec, …) are attached via `Sample.extra`.
    """

    name: str = "workload-type"
    streaming: bool = False  # if True, runner reads the response chunk-by-chunk

    @abstractmethod
    async def make_request(self, item: Any) -> Request:
        """Build a `Request` from a workload item.

        `item` may be `None` if the workload yields placeholders (e.g. a fixed-
        request benchmark with no per-request data).
        """

    async def make_sample(self, item: Any, request: Request, response: Response,
                          start_ts: float) -> Sample:
        """Build a Sample. Override to attach workload-type-specific metrics."""
        return Sample(
            start_ts=start_ts,
            latency_s=response.elapsed_s,
            status=response.status,
            ok=response.ok,
            request_ok=response.ok,
            bytes_recv=len(response.body) if response.body else 0,
            bytes_sent=_estimate_request_size(request),
            error=response.error,
            workload=self.name,
            meta=dict(request.meta),
        )

    async def run_ticket(self, ctx: TicketContext) -> Sample:
        """Default ticket flow: one make_request → one fire → one make_sample.

        Override to implement multi-step protocols (e.g. create → exec → delete).
        Use `ctx.fire(req)` to issue requests; it applies pre-hooks and uses
        the shared httpx2 async client. Per-lane ``ctx.overrides`` are applied
        by :meth:`apply_overrides` right after the request is built.
        """
        req: Optional[Request] = None
        try:
            req = await self.make_request(ctx.item)
            if ctx.overrides:
                req = await maybe_await(self.apply_overrides(req, ctx.overrides))
            resp = await ctx.fire(req)
            sample = await self.make_sample(ctx.item, req, resp, ctx.start_mono)
        except Exception as e:  # noqa: BLE001 - a failing turn must not wedge its session (#16)
            return await failed_ticket(self, ctx, e, request=req)
        for hook in ctx.post_hooks:
            sample = await maybe_await(hook(req, resp, sample))
        return sample

    def apply_overrides(self, req: Request, overrides: dict[str, Any]) -> Request:
        """Apply per-lane request overrides (``TicketContext.overrides``).

        The default handles ``{"model": ...}``: a JSON body gets its ``model``
        field rewritten and the effective model is stamped into ``req.meta``
        (and therefore ``Sample.meta``) as ``model``. Subclasses whose protocol
        does not carry a model name override this to record the label only.
        """
        model = overrides.get("model")
        if model:
            if isinstance(req.json, dict):
                req.json["model"] = model
            req.meta["model"] = model
        return req

    async def aclose(self) -> None:
        return None


async def failed_ticket(workload_type: WorkloadType, ctx: TicketContext, exc: BaseException,
                       *, request: Optional[Request] = None) -> Sample:
    """Record a turn that raised (in ``make_request``, ``make_sample``, a
    pre-hook or ``ctx.fire``) as a failed sample — *through the post-hooks*
    — instead of letting the exception escape (issue #16).

    A workload that schedules on completion (session replay) installs its
    completion hook as a post-hook; if the exception skipped the hooks the
    session's slot would never be released and the run would only end at the
    load model's time limit. Post-hook failures on this path are logged, not
    raised, so every hook still sees the sample.
    """
    error = f"{type(exc).__name__}: {exc}"
    latency = max(time.monotonic() - ctx.start_mono, 0.0)
    resp = Response(status=0, headers={}, body=b"", elapsed_s=latency, ok=False, error=error)
    sample = Sample(
        start_ts=ctx.start_mono, latency_s=latency, status=0, ok=False, request_ok=False,
        error=error, workload=workload_type.name,
        meta=dict(request.meta) if request is not None else {},
    )
    for hook in ctx.post_hooks:
        try:
            sample = await maybe_await(hook(request, resp, sample))
        except Exception as e:  # noqa: BLE001 - the sample is already failed; keep releasing
            _logger.warning("%s: post-hook %r raised on a failed request: %s: %s",
                            workload_type.name, getattr(hook, "__name__", hook), type(e).__name__, e)
    return sample


def _estimate_request_size(req: Request) -> int:
    if req.body is not None:
        return len(req.body)
    if req.json is not None:
        import json as _json
        return len(_json.dumps(req.json).encode("utf-8"))
    return 0
