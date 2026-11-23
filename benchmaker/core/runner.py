"""BenchRunner: ties scheduler -> workload (dataset) -> workload-type ->
httpx2 async client -> metrics."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx2

from benchmaker.core.capability import (
    VoidRule,
    evaluate_void_rules,
    fetch_json,
    merge_void_verdicts,
)
from benchmaker.core.endpoints import endpoint_model, route_url
from benchmaker.core.load import LoadModel
from benchmaker.core.metrics import DEFAULT_STALL_MS, ITL_SERIES, WARMUP_META_KEY, MetricsAggregator  # noqa: F401 (re-exported)
from benchmaker.core.types import (
    PostResponseHook,
    PreRequestHook,
    Request,
    Response,
    Sample,
    TicketContext,
    maybe_await,
)
from benchmaker.core.monitors import Monitor, run_monitor_loop
from benchmaker.core.trace import TraceRecorder
from benchmaker.workloads.base import WorkloadType
from benchmaker.workloads.datasets import Workload, StaticWorkload

logger = logging.getLogger(__name__)


def _workload_url(wt: Any) -> Optional[str]:
    """The workload-type's request URL (public ``url`` attribute or ``_url``)."""
    return getattr(wt, "url", None) or getattr(wt, "_url", None)


@dataclass
class BenchLane:
    """One independently scheduled input lane in a mixed benchmark.

    All lanes share the enclosing :class:`BenchConfig`'s workload type and
    endpoint, but each owns its own data source and load model.  This keeps
    OpenAI/HTTP protocol configuration centralized while preserving independent
    arrival processes for phase-swing experiments.
    """

    name: str
    workload: Workload
    load: LoadModel
    # Route this lane's requests at a different model behind the same endpoint
    # (multi-model phase, issue #5). ``None`` keeps whatever the workload-type
    # put in the request.
    model: Optional[str] = None
    # Extra metadata stamped onto every sample of the lane (phase labels, arm
    # tags, ...).
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("lane name must be a non-empty string")


@dataclass
class BenchPhase:
    """One timed stage of a phased benchmark (issue #5).

    A phase owns its lanes; phases run back-to-back, each lane's load model
    starts at the phase's own t=0, and every sample is tagged with
    ``meta["phase"]``. Per-phase summaries (wall time, throughput, lanes, ITL)
    land in ``summary["phases"]``.
    """

    name: str
    lanes: list[BenchLane]
    duration_s: float
    # Wait for the phase's in-flight requests to finish before the next phase
    # starts (clean per-phase windows). Default False: phases switch
    # back-to-back, as a continuous moving hotspot does.
    drain: bool = False

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("phase name must be a non-empty string")
        if not self.lanes:
            raise ValueError(f"phase {self.name!r} needs at least one lane")
        if self.duration_s <= 0:
            raise ValueError(f"phase {self.name!r} duration_s must be > 0")


@dataclass
class BenchConfig:
    workload_type: WorkloadType                # how to talk to the service
    load: Optional[LoadModel] = None           # when to fire (single workload)
    workload: Workload = field(default_factory=StaticWorkload)  # what to send
    lanes: list[BenchLane] = field(default_factory=list)
    pre_hooks: list[PreRequestHook] = field(default_factory=list)
    post_hooks: list[PostResponseHook] = field(default_factory=list)
    monitors: list[Monitor] = field(default_factory=list)  # optional periodic samplers
    # Optional trace recorder. When set, each fired request is appended to a
    # JSONL file (with relative timestamp) so a later run can replay the bench
    # deterministically via `benchmaker.core.trace.TracePacedLoad` + `ReplayWorkloadType`.
    recorder: Optional[TraceRecorder] = None
    connection_limit: int = 1000
    timeout_s: float = 60.0
    max_in_flight: int = 10000
    progress_every_s: float = 1.0
    stop_on_exhausted: bool = True
    # --- issue #1: token-level ITL pooling and decode stalls -------------
    # Inter-token gap (ms) above which a decode step counts as a stall;
    # ``None`` disables the ``stalls`` summary block.
    stall_ms: Optional[float] = DEFAULT_STALL_MS
    # Write each sample's raw ``series`` (per-token gaps) to samples.jsonl.
    dump_series: bool = False
    # ``Sample.meta`` keys to break the summary down by (e.g. ``["cache_class"]``).
    group_by: list[str] = field(default_factory=list)
    # --- issue #5: warm-up exclusion and phases ---------------------------
    # Requests fired in the first ``warmup_s`` seconds, or the first
    # ``warmup_requests`` requests, are tagged ``meta["warmup"]`` and excluded
    # from every summary block (still written to samples.jsonl).
    warmup_requests: int = 0
    warmup_s: float = 0.0
    # Timed stages run back-to-back (alternative to ``load``/``lanes``).
    phases: list[BenchPhase] = field(default_factory=list)
    # --- issue #4: runtime capability capture and void rules -------------
    # ``{"url": ..., "headers": {...}, "timeout_s": 10}``: fetched before and
    # after the run; the snapshots are written to the bundle and feed void rules.
    capability: Optional[dict[str, Any]] = None
    void_rules: list[VoidRule] = field(default_factory=list)
    # --- issue #4: manifest inputs -----------------------------------------
    # ``model_revision`` overrides the served-model revision in the manifest
    # (otherwise taken from the capability snapshot); ``record_env`` are env
    # var globs recorded in the manifest in addition to the defaults.
    model_revision: Optional[str] = None
    record_env: list[str] = field(default_factory=list)
    # --- issue #8: several servers (one per model) behind one arm ----------
    # ``{model: {url, capability?, headers?}}`` (see core/endpoints.py). A
    # request whose body ``model`` matches a key is sent to that endpoint;
    # capability, void rules and the manifest run per endpoint.
    endpoints: dict[str, dict] = field(default_factory=dict)
    # Extra void violations to record (lifecycle checks run by `benchmaker
    # arms`, issue #9) — merged into ``summary["void"]`` on every write.
    extra_void: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.phases:
            if self.load is not None or self.lanes:
                raise ValueError("configure phases, or load/workload, or lanes — not both")
            names = [phase.name for phase in self.phases]
            if len(names) != len(set(names)):
                raise ValueError("phase names must be unique")
            for phase in self.phases:
                lane_names = [lane.name for lane in phase.lanes]
                if len(lane_names) != len(set(lane_names)):
                    raise ValueError(f"lane names in phase {phase.name!r} must be unique")
        elif self.lanes:
            if self.load is not None:
                raise ValueError("configure either load/workload or lanes, not both")
            names = [lane.name for lane in self.lanes]
            if len(names) != len(set(names)):
                raise ValueError("mixed benchmark lane names must be unique")
        elif self.load is None:
            raise ValueError("BenchConfig requires a load model or at least one lane")


@dataclass
class BenchResult:
    samples: list[Sample]
    summary: dict
    # Capability snapshots: ``{"start": ..., "end": ...}``, or with an
    # endpoint map (issue #8) ``{"endpoints": {model: {"start", "end"}}}``.
    capability: Optional[dict[str, Any]] = None


class BenchRunner:
    def __init__(self, config: BenchConfig):
        self.cfg = config
        self.metrics = MetricsAggregator(
            stall_ms=config.stall_ms,
            group_by=list(config.group_by),
            dump_series=config.dump_series,
        )
        self._fired = 0
        self.capability: Optional[dict[str, Any]] = None

    async def run(self) -> BenchResult:
        # httpx2 has no per-connector force_close / ttl_dns_cache knobs; pooling
        # is governed by Limits. force_close=False (keepalive on) maps to
        # allowing up to `connection_limit` keep-alive connections, matching the
        # previous aiohttp pool behaviour. httpx2 resolves DNS through the
        # system resolver per connection (no in-process TTL cache).
        limits = httpx2.Limits(
            max_connections=self.cfg.connection_limit,
            max_keepalive_connections=self.cfg.connection_limit,
        )
        timeout = httpx2.Timeout(self.cfg.timeout_s)
        if self.cfg.recorder is not None:
            self.cfg.recorder.open(start_mono=self.metrics.start_time)
        if self.cfg.capability or self.cfg.endpoints:
            await self._capture("start")
        async with httpx2.AsyncClient(limits=limits, timeout=timeout) as session:
            try:
                await self._drive(session)
            finally:
                await self._aclose_workloads()
                await self.cfg.workload_type.aclose()
                if self.cfg.recorder is not None:
                    self.cfg.recorder.close()
        self.metrics.finalize()
        if self.cfg.capability or self.cfg.endpoints:
            await self._capture("end")
        return BenchResult(
            samples=self.metrics.samples,
            summary=self._summary(),
            capability=self.capability,
        )

    def _summary(self) -> dict:
        summary = self.metrics.summary()
        if self.capability is not None:
            summary["capability"] = self.capability
            summary["void"] = self.void_verdict()
        elif self.cfg.extra_void:
            summary["void"] = self.void_verdict()
        return summary

    def void_verdict(self) -> dict:
        """``{"void", "violations", "checks"}`` over every endpoint's snapshots
        (issue #8) plus any lifecycle violations recorded in ``cfg.extra_void``
        (issue #9)."""
        verdicts: list[dict] = []
        per = (self.capability or {}).get("endpoints")
        if per:
            for name, snaps in per.items():
                verdicts.append(evaluate_void_rules(
                    self.cfg.void_rules, snaps.get("start"), snaps.get("end"),
                    endpoint=name).to_dict())
        else:
            verdicts.append(evaluate_void_rules(
                self.cfg.void_rules, (self.capability or {}).get("start"),
                (self.capability or {}).get("end")).to_dict())
        if self.cfg.extra_void:
            verdicts.append({"void": True, "violations": list(self.cfg.extra_void), "checks": []})
        return merge_void_verdicts(verdicts)

    def is_void(self) -> bool:
        """True when a [void rule](../core/capability.py) was violated (issue #4)."""
        verdict = self._summary().get("void")
        return bool(verdict and verdict.get("void"))

    async def _capture(self, at: str) -> None:
        """Snapshot every capability source at ``at`` (``start`` | ``end``).
        Without an endpoint map that is the single ``cfg.capability``; with
        one (issue #8) each endpoint's capability URL — given explicitly or
        derived from the shared ``capability.url`` by swapping scheme/host —
        is captured separately."""
        sources = self._capability_sources()
        if not sources:
            return
        if len(sources) == 1 and sources[0][0] is None:
            _, cap = sources[0]
            snapshot = await asyncio.to_thread(
                fetch_json, cap["url"], float(cap.get("timeout_s", 10.0)), cap.get("headers"))
            snapshot["at"] = at
            snapshot["wall_time"] = time.time()
            self.capability = {**(self.capability or {}), at: snapshot}
            if not snapshot.get("ok"):
                logger.warning("capability snapshot (%s) failed: %s", at, snapshot.get("error"))
            return
        per = dict((self.capability or {}).get("endpoints") or {})
        for name, cap in sources:
            snapshot = await asyncio.to_thread(
                fetch_json, cap["url"], float(cap.get("timeout_s", 10.0)), cap.get("headers"))
            snapshot["at"] = at
            snapshot["wall_time"] = time.time()
            snapshot["endpoint"] = name
            per[name] = {**per.get(name, {}), at: snapshot}
            if not snapshot.get("ok"):
                logger.warning("capability snapshot (%s, endpoint %s) failed: %s", at, name, snapshot.get("error"))
        self.capability = {**(self.capability or {}), "endpoints": per}

    def _capability_sources(self) -> list[tuple[Optional[str], dict]]:
        """``[(endpoint_name_or_None, {url, headers?, timeout_s?})]`` for every
        endpoint that has a capability URL."""
        base = dict(self.cfg.capability or {})
        if not self.cfg.endpoints:
            return [(None, base)] if base.get("url") else []
        out = []
        for name, spec in self.cfg.endpoints.items():
            cap = {**base, **{k: v for k, v in spec.items() if k != "url" and v}}
            if spec.get("capability"):
                cap["url"] = spec["capability"]
            elif base.get("url"):
                cap["url"] = route_url(spec["url"], base["url"])
            cap["headers"] = {**(base.get("headers") or {}), **(spec.get("headers") or {})}
            if cap.get("url"):
                out.append((name, cap))
        return out

    def _is_warmup(self, start_mono: float) -> bool:
        if self.cfg.warmup_requests and self._fired < self.cfg.warmup_requests:
            return True
        if self.cfg.warmup_s and start_mono - self.metrics.start_time < self.cfg.warmup_s:
            return True
        return False

    async def _drive(self, session: httpx2.AsyncClient) -> None:
        sem = asyncio.Semaphore(self.cfg.max_in_flight)
        tasks: set[asyncio.Task] = set()
        progress_task = asyncio.create_task(self._progress_loop())

        # Spawn monitor loops.
        monitor_stop = asyncio.Event()
        monitor_tasks: list[asyncio.Task] = []
        bench_start = self.metrics.start_time
        for mon in self.cfg.monitors:
            buf = self.metrics.monitor_buffer(mon.name)
            monitor_tasks.append(asyncio.create_task(
                run_monitor_loop(mon, buf, bench_start, monitor_stop)
            ))

        try:
            if self.cfg.phases:
                for phase in self.cfg.phases:
                    await self._drive_phase(session, sem, tasks, phase)
            elif self.cfg.lanes:
                await self._drive_lanes(session, sem, tasks, self.cfg.lanes)
            else:
                await self._drive_single(session, sem, tasks)
        finally:
            progress_task.cancel()
            try:
                await progress_task
            except (asyncio.CancelledError, Exception):
                pass

            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

            # Signal monitors to do one last tick and exit, then wait for them.
            monitor_stop.set()
            if monitor_tasks:
                await asyncio.gather(*monitor_tasks, return_exceptions=True)

    async def _drive_single(self, session: httpx2.AsyncClient,
                            sem: asyncio.Semaphore,
                            tasks: set[asyncio.Task]) -> None:
        assert self.cfg.load is not None
        async for _ in self.cfg.load.tickets():
            try:
                item = await self.cfg.workload.next_item()
            except StopAsyncIteration:
                if self.cfg.stop_on_exhausted:
                    break
                continue

            await sem.acquire()
            task = asyncio.create_task(
                self._fire(session, item, sem, self.cfg.load)
            )
            tasks.add(task)
            task.add_done_callback(tasks.discard)

    async def _drive_phase(self, session: httpx2.AsyncClient,
                           sem: asyncio.Semaphore,
                           tasks: set[asyncio.Task],
                           phase: BenchPhase) -> None:
        """Run one phase's lanes for ``phase.duration_s``, then drain (issue #5).

        Samples are tagged ``meta["phase"]`` so every per-phase summary is
        complete regardless of timing. With ``drain=True`` the window also
        covers the in-flight tail, so per-phase throughput is comparable to a
        standalone run of the same lanes for the same duration.
        """
        phase_start = time.monotonic()
        logger.info("phase %s: %s lanes for %.1fs", phase.name,
                    ",".join(lane.name for lane in phase.lanes), phase.duration_s)
        await self._drive_lanes(session, sem, tasks, phase.lanes, phase.name)
        if phase.drain and tasks:
            logger.info("phase %s: draining %d in-flight requests", phase.name, len(tasks))
            await asyncio.gather(*list(tasks), return_exceptions=True)
        phase_end = time.monotonic()
        self.metrics.phase_windows[phase.name] = (phase_start, phase_end)
        logger.info("phase %s: done after %.1fs (drained)", phase.name, phase_end - phase_start)

    async def _drive_lanes(self, session: httpx2.AsyncClient,
                           sem: asyncio.Semaphore,
                           tasks: set[asyncio.Task],
                           lanes: list[BenchLane],
                           phase_name: Optional[str] = None) -> None:
        """Run each lane's admission iterator concurrently.

        A finite workload ends only its own lane.  Other lanes keep producing
        tickets, which is required when one dataset is short or intentionally
        bursty and another drives a long complementary phase.
        """

        async def produce(lane: BenchLane) -> None:
            async for _ in lane.load.tickets():
                try:
                    item = await lane.workload.next_item()
                except StopAsyncIteration:
                    if self.cfg.stop_on_exhausted:
                        break
                    continue

                await sem.acquire()
                task = asyncio.create_task(
                    self._fire(session, item, sem, lane.load, lane.name,
                               phase=phase_name, model=lane.model, lane_meta=lane.meta)
                )
                tasks.add(task)
                task.add_done_callback(tasks.discard)

        producers = [asyncio.create_task(produce(lane)) for lane in lanes]
        try:
            await asyncio.gather(*producers)
        finally:
            for producer in producers:
                if not producer.done():
                    producer.cancel()
            await asyncio.gather(*producers, return_exceptions=True)

    async def _fire(self, session: httpx2.AsyncClient, item: Any,
                    sem: asyncio.Semaphore, load: LoadModel,
                    lane_name: Optional[str] = None,
                    phase: Optional[str] = None,
                    model: Optional[str] = None,
                    overrides: Optional[dict[str, Any]] = None,
                    lane_meta: Optional[dict[str, Any]] = None) -> None:
        start_mono = time.monotonic()
        warmup = self._is_warmup(start_mono)
        self._fired += 1
        lane_meta = dict(lane_meta or {})
        if model:
            overrides = {"model": model, **(overrides or {})}
        try:
            async def fire(req: Request) -> Response:
                if lane_name is not None:
                    # The config-defined lane is authoritative: callers may
                    # still attach arbitrary metadata, but cannot accidentally
                    # collapse a mixed run into an incorrect lane.
                    req.meta["lane"] = lane_name
                if phase is not None:
                    req.meta["phase"] = phase
                if warmup:
                    req.meta[WARMUP_META_KEY] = True
                for k, v in lane_meta.items():
                    req.meta.setdefault(k, v)
                for hook in self.cfg.pre_hooks:
                    req = await maybe_await(hook(req))
                self._route(req)
                fire_start = time.monotonic()
                if self.cfg.recorder is not None:
                    await self.cfg.recorder.record(req, fire_start)
                return await self._execute(session, req, fire_start)

            ctx = TicketContext(
                item=item,
                start_mono=start_mono,
                fire=fire,
                pre_hooks=tuple(self.cfg.pre_hooks),
                post_hooks=tuple(self.cfg.post_hooks),
                workload_name=self.cfg.workload_type.name,
                overrides=dict(overrides or {}),
            )
            sample = await self.cfg.workload_type.run_ticket(ctx)
            if lane_name is not None:
                sample.meta["lane"] = lane_name
            if phase is not None:
                sample.meta["phase"] = phase
            if warmup or (sample.meta.get(WARMUP_META_KEY)):
                sample.meta[WARMUP_META_KEY] = True
            for k, v in lane_meta.items():
                sample.meta.setdefault(k, v)
            self.metrics.add(sample)
        except Exception as e:
            fail_meta = {k: v for k, v in (("phase", phase), (WARMUP_META_KEY, True if warmup else None)) if v is not None}
            self.metrics.add(_failure_sample(
                f"{type(e).__name__}: {e}",
                self.cfg.workload_type.name,
                lane_name,
                meta=fail_meta,
            ))
        finally:
            sem.release()
            load.on_complete()

    def _route(self, req: Request) -> None:
        """Issue #8: with an endpoint map, the request's body ``model`` (a
        lane's ``model``, or a workload item's) selects the server. The
        endpoint's URL/headers are applied and the served model is stamped
        into ``meta`` so per-model summaries (`group_by: [model]`) work."""
        if not self.cfg.endpoints:
            return
        name = endpoint_model(req.json, req.meta, self.cfg.endpoints)
        if name is None:
            return
        spec = self.cfg.endpoints[name]
        req.url = route_url(spec["url"], req.url)
        for k, v in spec.get("headers", {}).items():
            req.headers.setdefault(k, v)
        req.meta.setdefault("model", name)
        req.meta["endpoint"] = name

    async def _aclose_workloads(self) -> None:
        workloads: list[Workload] = []
        if self.cfg.phases:
            for phase in self.cfg.phases:
                workloads.extend(lane.workload for lane in phase.lanes)
        elif self.cfg.lanes:
            workloads.extend(lane.workload for lane in self.cfg.lanes)
        else:
            workloads.append(self.cfg.workload)
        closed: set[int] = set()
        for workload in workloads:
            if id(workload) not in closed:
                closed.add(id(workload))
                await workload.aclose()

    async def _execute(self, session: httpx2.AsyncClient, req: Request,
                       start_mono: float) -> Response:
        kwargs: dict = {"headers": req.headers, "params": req.params}
        if req.json is not None:
            kwargs["json"] = req.json
        elif req.body is not None:
            kwargs["content"] = req.body
        if req.timeout_s is not None:
            kwargs["timeout"] = httpx2.Timeout(req.timeout_s)

        try:
            if self.cfg.workload_type.streaming:
                # Stream so each network read surfaces as a separate chunk,
                # preserving per-chunk timing (TTFT/ITL). httpx2's aiter_raw()
                # yields bytes at arbitrary boundaries exactly like aiohttp's
                # content.iter_any(), so the SSE reassembler in workloads/_sse.py
                # still rejoins events split across chunks (#13).
                chunks: list[bytes] = []
                chunk_times: list[float] = []
                async with session.stream(req.method, req.url, **kwargs) as resp:
                    async for chunk in resp.aiter_raw():
                        chunks.append(chunk)
                        chunk_times.append(time.monotonic() - start_mono)
                    body = b"".join(chunks)
                    elapsed = time.monotonic() - start_mono
                    return Response(
                        status=resp.status_code,
                        headers=dict(resp.headers),
                        body=body,
                        elapsed_s=elapsed,
                        ok=200 <= resp.status_code < 400,
                        stream_chunks=chunks,
                        stream_chunk_times=chunk_times,
                    )
            else:
                resp = await session.request(req.method, req.url, **kwargs)
                body = resp.content
                elapsed = time.monotonic() - start_mono
                return Response(
                    status=resp.status_code,
                    headers=dict(resp.headers),
                    body=body,
                    elapsed_s=elapsed,
                    ok=200 <= resp.status_code < 400,
                )
        except asyncio.TimeoutError:
            return Response(
                status=0, headers={}, body=b"",
                elapsed_s=time.monotonic() - start_mono,
                ok=False, error="timeout",
            )
        except httpx2.TimeoutException:
            return Response(
                status=0, headers={}, body=b"",
                elapsed_s=time.monotonic() - start_mono,
                ok=False, error="timeout",
            )
        except httpx2.HTTPError as e:
            return Response(
                status=0, headers={}, body=b"",
                elapsed_s=time.monotonic() - start_mono,
                ok=False, error=f"{type(e).__name__}: {e}",
            )

    async def _progress_loop(self) -> None:
        if self.cfg.progress_every_s <= 0:
            return
        last_n = 0
        last_t = time.monotonic()
        seen_errors: set[str] = set()
        try:
            while True:
                await asyncio.sleep(self.cfg.progress_every_s)
                now = time.monotonic()
                n = len(self.metrics.samples)
                dn = n - last_n
                dt = now - last_t
                inst = dn / dt if dt > 0 else 0.0
                window = self.metrics.samples[last_n:]
                ok = sum(1 for s in window if s.ok)
                # Wrong: request was delivered (HTTP success) but a post-hook /
                # workload graded the output as a failure (e.g. eval gate).
                wrong = sum(1 for s in window if not s.ok and s.request_ok)
                fail = dn - ok - wrong
                logger.info(
                    "+%5d req  (%7.1f rps, %d ok, %d wrong, %d fail) | total=%d",
                    dn, inst, ok, wrong, fail, n,
                )
                # Surface the first occurrence of each distinct error string —
                # one short line per kind, so failed runs are diagnosable
                # without grepping samples.jsonl.
                for s in window:
                    if s.error and s.error not in seen_errors:
                        seen_errors.add(s.error)
                        msg = s.error if len(s.error) <= 200 else s.error[:200] + "..."
                        bucket = "fail" if not s.request_ok else "wrong"
                        logger.warning("  first %s: %s", bucket, msg)
                last_n = n
                last_t = now
        except asyncio.CancelledError:
            return

    def write_bundle(self, out_dir: str, *, manifest: Optional[dict] = None,
                     source_config: Optional[dict] = None, labels: Optional[dict] = None,
                     model_revision: Optional[str] = None,
                     env_overrides: Optional[dict[str, str]] = None,
                     record_env: Optional[list[str]] = None, **kwargs) -> str:
        """Write a per-run directory bundle. See `benchmaker.io.bundle.write_bundle`.

        Builds the :mod:`benchmaker.io.manifest` unless one is passed in.
        ``model_revision`` / ``record_env`` default to the config's
        ``model_revision`` / ``record_env`` keys.
        """
        from benchmaker import __version__ as bm_version
        from benchmaker.io.bundle import write_bundle
        from benchmaker.io.manifest import build_manifest

        model_revision = model_revision if model_revision is not None else self.cfg.model_revision
        record_env = record_env if record_env is not None else (list(self.cfg.record_env) or None)

        workload_name = (
            "mix:" + ",".join(lane.name for lane in self.cfg.lanes)
            if self.cfg.lanes
            else "phases:" + ",".join(p.name for p in self.cfg.phases)
            if self.cfg.phases
            else self.cfg.workload.name
        )
        if manifest is None:
            manifest = build_manifest(
                source_config=source_config or {},
                run_id=str(kwargs.get("run_id") or ""),
                workload_type=self.cfg.workload_type,
                workload=self.cfg.workload,
                workload_name=workload_name,
                labels=labels,
                started_at=self.metrics.start_wall,
                model_revision=model_revision,
                capability=self.capability,
                env=env_overrides,
                record_env=record_env,
                benchmaker_version=bm_version,
                endpoints=self.cfg.endpoints or None,
                template_url=_workload_url(self.cfg.workload_type),
            )
        return write_bundle(
            out_dir,
            self.metrics,
            workload_type_name=self.cfg.workload_type.name,
            workload_name=workload_name,
            capability=self.capability,
            manifest=manifest,
            summary=self._summary(),
            source_config=source_config,
            labels=labels,
            **kwargs,
        )


def _failure_sample(error: str, workload: str,
                    lane_name: Optional[str] = None,
                    meta: Optional[dict[str, Any]] = None) -> Sample:
    m = dict(meta or {})
    if lane_name is not None:
        m["lane"] = lane_name
    return Sample(
        start_ts=time.monotonic(),
        latency_s=0.0,
        status=0,
        ok=False,
        request_ok=False,
        error=error,
        workload=workload,
        meta=m,
    )


async def run_bench(config: BenchConfig) -> BenchResult:
    return await BenchRunner(config).run()
