"""OpenAI-compatible completions workload-types (with SSE streaming).

Two classes share the metrics below:

* :class:`OpenAIChatWorkloadType` — ``/v1/chat/completions`` (``type: openai``).
* :class:`OpenAICompletionsWorkloadType` — ``/v1/completions`` (``type:
  openai-completions``), the text-completions wire format. Its ``prompt`` may
  be a string or a list of token ids, which is how ``send: token_ids`` trace
  replay ships pre-tokenized, page-aligned blocks without a chat template
  (issue #11).


Item interpretation:
  * `str`            → wrap as `[{"role": "user", "content": item}]`
  * `list[dict]`     → use as `messages`
  * `dict`           → merged with the request body. May contain any of:
                        messages, prompt (alias for a user-message str), model,
                        max_tokens, temperature, top_p, stop, …
                        plus any extra OpenAI-compat sampling params.

Captured metrics (on top of base latency/throughput):
    ttft_s              first-token latency (see `ttft_token`)
    content_ttft_s      time to first visible (content) token, when reasoning
                        preceded it (i.e. when it differs from ttft_s)
    itl_ms_mean/p50/p99 inter-token latency across the whole generation (ms),
                        counting reasoning tokens the same as content tokens
    tokens_out          completion tokens (from usage, else counted)
    reasoning_tokens    from usage.completion_tokens_details, when present
    content_tokens      completion_tokens - reasoning_tokens, when both known
    prompt_tokens       from usage block when available
    tokens_per_s        completion_tokens / generation_time
    cached_ratio       cached_tokens / prompt_tokens (when both known)

classification / provenance kept in `Sample.meta` (not aggregated):
    cache_class         "device_hit" | "restore" | "partial" | "miss", from
                        usage + cache-source hints (see `classify_cache`)
    cache_hint          raw cache-source hint the server gave (if any)
    ttft_token_kind     "content" or "reasoning": which kind of token the
                        headline `ttft_s` was stamped on
    ttft_token_index    index of that token in the stream (0 = first token),
                        so a TTFT drop caused by an earlier first *reasoning*
                        token is detectable (issue #4)

Raw per-token inter-token gaps (ms) are kept on every sample as
``sample.series["itl_ms"]`` and pooled across requests by the summary
(``itl_ms_pooled`` p50/p90/p99/p99.5/p99.9/max and the ``stalls`` block,
issue #1); they are written to samples.jsonl when ``dump_series`` is on.

`ttft_token` selects which token the headline `ttft_s` is measured to:
  * `"any"`     (default) — first token the server produces, reasoning *or*
                  content. The engine-cost signal a serving benchmark wants;
                  isolates prefill→decode from the model's reasoning policy.
  * `"content"` — first *visible* (non-reasoning) token, the latency a user
                  perceives before an answer appears.
The first-content time is always surfaced separately as `content_ttft_s`
when it is a distinct instant, so both signals are available regardless of
the knob. Inter-token latency always spans the whole stream (reasoning and
content decoded at the same per-token cost), so it reflects true decode
cadence for thinking models too (#14).
"""

from __future__ import annotations

import json
import logging
import os
import re
import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union

from benchmaker.env import load_dotenv
from benchmaker.core.metrics import ITL_SERIES
from benchmaker.core.types import Request, Response, Sample
from benchmaker.workloads._sse import reassemble_sse_lines
from benchmaker.workloads.base import WorkloadType

_logger = logging.getLogger(__name__)

# Guards a one-time warning: an endpoint that never returns a `usage` block (or
# whose usage we failed to read) silently zeroes prompt/cached-token accounting,
# which is the dangerous failure mode for cache-effectiveness runs. Warn once
# rather than per request.
_warned_missing_usage = False


# Env vars recognised by `OpenAIChatWorkloadType.from_env`.
# `OPENAI_BASE_URL` is the OpenAI SDK convention; `OPENAI_API_BASE_URL` and
# `OPENAI_API_BASE` are common alternates.
_ENV_BASE_URL_KEYS = ("OPENAI_API_BASE_URL", "OPENAI_BASE_URL", "OPENAI_API_BASE")
_ENV_MODEL_KEYS = ("OPENAI_COMPATIBLE_MODEL", "OPENAI_MODEL")
_ENV_KEY_KEYS = ("OPENAI_API_KEY",)


# Keys that are legal OpenAI/vLLM/SGLang chat-completions *body* params. In
# passthrough_meta mode, only these (plus configured extra_body keys) flow into
# the request body; every other row key is recorded into Request.meta instead.
# `stream` is intentionally excluded — we always force streaming for metrics.
# `max_new_tokens` is intentionally NOT in this set — it is handled separately
# via the `max_tokens` alias in `_split_passthrough`.
_OPENAI_BODY_KEYS = frozenset({
    "messages", "max_tokens", "temperature", "top_p", "top_k", "n", "stop",
    "presence_penalty", "frequency_penalty", "repetition_penalty", "seed",
    "logprobs", "top_logprobs", "logit_bias", "response_format", "tools",
    "tool_choice", "min_tokens", "ignore_eos", "guided_json", "guided_regex",
    "guided_choice",
})


class OpenAIChatWorkloadType(WorkloadType):
    streaming = True
    name = "openai-chat"

    def __init__(
        self,
        url: str,
        model: str,
        max_tokens: int = 128,
        temperature: float = 0.0,
        extra_body: Optional[dict] = None,
        headers: Optional[dict[str, str]] = None,
        api_key: Optional[str] = None,
        timeout_s: Optional[float] = 600.0,
        name: str = "openai-chat",
        passthrough_meta: bool = False,
        ttft_token: str = "any",
        cache_classifier: Optional[Callable[["CacheClassInfo"], Optional[str]]] = None,
        cache_full_hit_ratio: float = 0.9,
        **sampling: Any,
    ):
        """Any extra keyword argument (e.g. ``min_tokens``, ``ignore_eos``,
        ``top_p``, ``stop``, ``repetition_penalty``, ``guided_json``, ...) is
        forwarded into the request body. ``extra_body`` is still accepted as
        an explicit dict; ``**sampling`` overrides it on key conflict.

        ``ttft_token`` (``"any"`` | ``"content"``, default ``"any"``) selects
        which token the headline ``ttft_s`` is measured to — the first token
        of any kind, or the first visible content token. See the module
        docstring for the reasoning-model rationale.

        ``cache_classifier`` (issue #2) overrides :func:`classify_cache`; it
        receives a :class:`CacheClassInfo` and returns ``"device_hit"``,
        ``"restore"``, ``"partial"``, ``"miss"`` or ``None`` (unclassified).
        ``cache_full_hit_ratio`` is the cached/prompt ratio at which a hit
        counts as full (``device_hit``); a hit below it is ``partial``.
        """
        if ttft_token not in ("any", "content"):
            raise ValueError(
                f"ttft_token must be 'any' or 'content', got {ttft_token!r}. "
                "'any' (default) measures time to the first token the server "
                "produces (reasoning or content); 'content' measures time to "
                "the first visible (non-reasoning) token."
            )
        if not 0.0 < cache_full_hit_ratio <= 1.0:
            raise ValueError(f"cache_full_hit_ratio must be in (0, 1], got {cache_full_hit_ratio!r}")
        self._ttft_token = ttft_token
        self._cache_classifier = cache_classifier
        self._cache_full_hit_ratio = float(cache_full_hit_ratio)
        self.name = name
        self._url = url
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._passthrough_meta = passthrough_meta
        merged: dict[str, Any] = dict(extra_body or {})
        merged.update(sampling)
        self._extra_body = merged
        self._timeout_s = timeout_s
        hdrs = dict(headers or {})
        hdrs.setdefault("Content-Type", "application/json")
        hdrs.setdefault("Accept", "text/event-stream")
        if api_key:
            hdrs.setdefault("Authorization", f"Bearer {api_key}")
        self._headers = hdrs

    @property
    def model(self) -> Optional[str]:
        """Model id placed in the request body (used by the run manifest, #4)."""
        return self._model

    @property
    def url(self) -> str:
        """Chat-completions URL; the path every ``endpoints:`` base URL (#8) inherits."""
        return self._url

    @classmethod
    def from_env(
        cls,
        url: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        dotenv_path: Optional[str] = ".env",
        endpoint_path: str = "chat/completions",
        **kwargs: Any,
    ) -> "OpenAIChatWorkloadType":
        """Construct from environment variables.

        Reads (in order of preference):
          base URL : OPENAI_API_BASE_URL, OPENAI_BASE_URL, OPENAI_API_BASE
          model    : OPENAI_COMPATIBLE_MODEL, OPENAI_MODEL
          api key  : OPENAI_API_KEY

        `dotenv_path` is loaded into `os.environ` first (without overriding
        existing vars). Explicit kwargs win over env vars.

        The full URL becomes `<base>/<endpoint_path>`, with at most one slash
        between them — so both `https://x/v1` and `https://x/v1/` work.
        """
        if dotenv_path:
            load_dotenv(dotenv_path)

        if url is None:
            base = _first_env(_ENV_BASE_URL_KEYS)
            if not base:
                raise ValueError(
                    "OPENAI_API_BASE_URL (or OPENAI_BASE_URL) is not set; "
                    "pass url=... or set the env var."
                )
            url = base.rstrip("/") + "/" + endpoint_path.lstrip("/")

        if model is None:
            model = _first_env(_ENV_MODEL_KEYS)
            if not model:
                raise ValueError(
                    "OPENAI_COMPATIBLE_MODEL (or OPENAI_MODEL) is not set; "
                    "pass model=... or set the env var."
                )

        if api_key is None:
            api_key = _first_env(_ENV_KEY_KEYS)  # optional; may stay None

        return cls(url=url, model=model, api_key=api_key, **kwargs)

    def _split_passthrough(self, item: dict) -> tuple[dict, dict]:
        """Split a full row into (body_part, meta_part) for passthrough mode.

        Allowlisted body params + configured extra_body keys go to the body;
        `max_new_tokens` aliases to `max_tokens`; a row `model` becomes a
        recorded label (never the target); an explicit `meta:{}` merges into
        meta; everything else is recorded into meta (not sent).

        When BOTH `max_tokens` and `max_new_tokens` are present, the explicit
        `max_tokens` wins (flows into the body via the allowlist) and the stray
        `max_new_tokens` is recorded into meta.
        """
        allowed = _OPENAI_BODY_KEYS | set(self._extra_body.keys())
        body_part: dict[str, Any] = {}
        meta_part: dict[str, Any] = {}
        for k, v in item.items():
            if k == "meta" and isinstance(v, dict):
                meta_part.update(v)
            elif k == "prompt":
                body_part["prompt"] = v  # promoted to messages below
            elif k == "model":
                meta_part["model_label"] = v
            elif k == "max_new_tokens" and "max_tokens" not in item:
                body_part["max_tokens"] = v
            elif k in allowed:
                body_part[k] = v
            else:
                meta_part[k] = v
        return body_part, meta_part

    async def make_request(self, item: Any) -> Request:
        body: dict[str, Any] = {
            "model": self._model,
            "stream": True,
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
            **self._extra_body,
        }
        body.setdefault("stream_options", {"include_usage": True})
        meta_extra: dict[str, Any] = {}

        if item is None:
            body["messages"] = [{"role": "user", "content": ""}]
        elif isinstance(item, str):
            body["messages"] = [{"role": "user", "content": item}]
        elif isinstance(item, list):
            body["messages"] = item
        elif isinstance(item, dict):
            if self._passthrough_meta:
                # _split_passthrough builds fresh dicts; the caller's input is
                # never mutated (matches the dict(item) copy in the else branch).
                item, meta_extra = self._split_passthrough(item)
            else:
                item = dict(item)
                # A workload item's ``meta`` (dataset tags: visit, doc_id,
                # trace fields, ...) is provenance, not a request parameter:
                # keep it out of the body and carry it on the request so it
                # reaches ``sample.meta`` and the grouped summaries (#5, #6).
                item_meta = item.pop("meta", None)
                if isinstance(item_meta, dict):
                    meta_extra.update(item_meta)
            # Promote "prompt" -> messages if no messages provided.
            if "messages" not in item and "prompt" in item:
                item["messages"] = [{"role": "user", "content": item.pop("prompt")}]
            body.update(item)
            body.setdefault("messages", [{"role": "user", "content": ""}])
        else:
            raise TypeError(f"OpenAIChatWorkloadType cannot interpret item {type(item).__name__}")

        # Row metadata first, then canonical keys last so they always win over
        # any same-named row field (e.g. a dataset column named "prompt_messages").
        meta: dict[str, Any] = dict(meta_extra)
        meta["prompt_messages"] = body["messages"]
        meta["max_tokens"] = body.get("max_tokens")
        return Request(
            method="POST",
            url=self._url,
            headers=dict(self._headers),
            json=body,
            timeout_s=self._timeout_s,
            meta=meta,
        )

    async def make_sample(self, item: Any, request: Request, response: Response,
                          start_ts: float) -> Sample:
        sample = await super().make_sample(item, request, response, start_ts)
        stats = parse_chat_stream(response.stream_chunks or [], response.stream_chunk_times or [])
        return finish_llm_sample(
            sample, request, response, stats, ttft_token=self._ttft_token,
            cache_classifier=self._cache_classifier,
            cache_full_hit_ratio=self._cache_full_hit_ratio)


# --------------------------------------------------------------------------- #
# Stream parsing shared by the chat and completions workload-types            #
# --------------------------------------------------------------------------- #


@dataclass
class _StreamStats:
    """What one streamed OpenAI completion told us, independent of wire format."""
    first_token: Optional[float] = None            # first token of any kind
    first_content_token: Optional[float] = None    # first visible token
    token_arrival_times: list = field(default_factory=list)
    token_kinds: list = field(default_factory=list)
    usage_completion_tokens: Optional[int] = None
    # Tokens carried by each streamed event (exact, from a server that reports
    # cumulative completion_tokens per event — sglang); index 0 unused.
    tokens_per_event: Optional[list] = None
    reasoning_tokens: Optional[int] = None
    prompt_tokens: Optional[int] = None
    cached_tokens: Optional[int] = None
    restored_tokens: Optional[int] = None
    cache_hint: Optional[str] = None
    finish_reason: Optional[str] = None
    usage_seen: bool = False
    usage_raw: dict = field(default_factory=dict)


def _iter_sse_objects(chunks: list, chunk_times: list):
    """Reassemble across chunk boundaries first: httpx2's aiter_raw() yields
    bytes at arbitrary offsets, so the final `usage` event is often split in
    two and would be silently dropped if each chunk were parsed on its own
    (#13). Yields ``(obj, arrival_time)``."""
    for line, t in reassemble_sse_lines(chunks, chunk_times):
        line = line.strip()
        if not line:
            continue
        if line.startswith(b"data:"):
            line = line[5:].strip()
        if line == b"[DONE]":
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict):
            yield obj, t


def _absorb_usage(obj: dict, st: _StreamStats) -> None:
    u = obj.get("usage")
    if not u:
        return
    st.usage_seen = True
    if not isinstance(u, dict):
        return
    st.usage_raw = dict(u)
    if u.get("completion_tokens") is not None:
        st.usage_completion_tokens = int(u["completion_tokens"])
    if u.get("prompt_tokens") is not None:
        st.prompt_tokens = int(u["prompt_tokens"])
    details = u.get("prompt_tokens_details")
    if isinstance(details, dict) and details.get("cached_tokens") is not None:
        st.cached_tokens = int(details["cached_tokens"])
    elif u.get("cached_tokens") is not None:
        st.cached_tokens = int(u["cached_tokens"])
    # Which tier served the cached prefix (device / host / remote), when the
    # server says so: usage, prompt_tokens_details or response headers
    # (issue #2). Also restored-from-host counts.
    for src in (u, details if isinstance(details, dict) else None):
        if isinstance(src, dict):
            if st.restored_tokens is None and src.get("restored_tokens") is not None:
                st.restored_tokens = int(src["restored_tokens"])
            if st.cache_hint is None:
                st.cache_hint = _cache_source_of(src)
    # Thinking models report generated reasoning tokens separately from
    # visible content; surfacing the breakdown keeps tokens_out honest and
    # exposes the reasoning share (#14).
    details = u.get("completion_tokens_details")
    if isinstance(details, dict) and details.get("reasoning_tokens") is not None:
        st.reasoning_tokens = int(details["reasoning_tokens"])


def parse_chat_stream(chunks: list, chunk_times: list) -> _StreamStats:
    """``/v1/chat/completions`` SSE: ``choices[*].delta.content`` and
    ``.reasoning_content`` are the produced tokens; the final event carries
    ``usage`` when ``stream_options.include_usage`` was requested."""
    st = _StreamStats()
    for obj, t in _iter_sse_objects(chunks, chunk_times):
        _absorb_usage(obj, st)
        for ch in obj.get("choices") or []:
            delta = ch.get("delta") or {}
            # Count reasoning tokens the same as content tokens: a thinking
            # model streams its chain-of-thought in `reasoning_content`
            # before any `content`, and those bytes are real engine output.
            # Without this, ttft/itl/tokens are wrong for the whole class of
            # reasoning models (#14).
            content_piece = delta.get("content")
            reasoning_piece = delta.get("reasoning_content")
            if content_piece:
                if st.first_token is None:
                    st.first_token = t
                if st.first_content_token is None:
                    st.first_content_token = t
                st.token_arrival_times.append(t)
                st.token_kinds.append("content")
            if reasoning_piece:
                if st.first_token is None:
                    st.first_token = t
                st.token_arrival_times.append(t)
                st.token_kinds.append("reasoning")
            if ch.get("finish_reason"):
                st.finish_reason = ch["finish_reason"]
    return st


def parse_completions_stream(chunks: list, chunk_times: list) -> _StreamStats:
    """``/v1/completions`` SSE: ``choices[*].text`` is the produced delta; the
    final event carries ``usage`` (vLLM/SGLang honour ``stream_options.
    include_usage`` here too). Chunks with empty ``text`` (e.g. the closing
    ``finish_reason`` event) are not token arrivals."""
    st = _StreamStats()
    for obj, t in _iter_sse_objects(chunks, chunk_times):
        _absorb_usage(obj, st)
        for ch in obj.get("choices") or []:
            piece = ch.get("text")
            if piece:
                if st.first_token is None:
                    st.first_token = t
                if st.first_content_token is None:
                    st.first_content_token = t
                st.token_arrival_times.append(t)
                st.token_kinds.append("content")
            if ch.get("finish_reason"):
                st.finish_reason = ch["finish_reason"]
    return st


def finish_llm_sample(sample: Sample, request: Request, response: Response, st: _StreamStats, *,
                      ttft_token: str = "any",
                      cache_classifier: Optional[Callable[[CacheClassInfo], Optional[str]]] = None,
                      cache_full_hit_ratio: float = 0.9) -> Sample:
    """Turn parsed stream statistics into the shared LLM metrics on ``sample``
    (ttft/itl/tokens/cache class/trace fields) — used by every streaming
    OpenAI-style workload-type so their metrics mean the same thing."""
    first_token, first_content_token = st.first_token, st.first_content_token
    token_arrival_times, token_kinds = st.token_arrival_times, st.token_kinds
    prompt_tokens, cached_tokens = st.prompt_tokens, st.cached_tokens
    cache_hint, finish_reason = st.cache_hint, st.finish_reason
    usage_raw = st.usage_raw

    # Loud-but-once diagnostic: a benchmark that requested usage yet parsed
    # none produces a structurally-zero cache hit rate — surface it instead
    # of silently recording misleading metrics.
    if response.ok and not st.usage_seen:
        _warn_missing_usage_once(request)

    # When the endpoint omits `usage`, fall back to the count of streamed
    # tokens — which now includes reasoning tokens, so a thinking model is
    # no longer undercounted by its entire chain-of-thought (#14).
    completion_tokens = (
        st.usage_completion_tokens if st.usage_completion_tokens is not None
        else len(token_arrival_times)
    )
    if cache_hint is None:
        cache_hint = _cache_source_of(dict(response.headers or {}))

    # `ttft_token` selects which token the headline `ttft_s` is measured
    # to. The first-content time is always surfaced separately as
    # `content_ttft_s` when it is a distinct instant, so a serving run
    # (ttft = first-any token) also captures perceived latency.
    ttft = first_content_token if ttft_token == "content" else first_token
    if ttft is not None:
        sample.extra["ttft_s"] = ttft
        # Provenance of the headline TTFT: which token it was stamped on
        # and where in the stream. A TTFT that dropped because a reasoning
        # token now arrives first is not a prefill win (issue #4).
        idx = _first_index(token_arrival_times, ttft)
        kind = token_kinds[idx] if idx is not None else None
        if kind is not None:
            sample.meta["ttft_token_kind"] = kind
        if idx is not None:
            sample.meta["ttft_token_index"] = idx
            sample.extra["ttft_token_index"] = float(idx)
    if first_content_token is not None and first_content_token != ttft:
        sample.extra["content_ttft_s"] = first_content_token

    if len(token_arrival_times) >= 2:
        # Per-token gaps (issue #17): the server's own token counts (usage, or
        # per-event cumulative counts) decide how many tokens each event
        # carried, and a burst of events read together splits its gap.
        gaps = token_gaps(
            token_arrival_times, st.tokens_per_event,
            completion_tokens=st.usage_completion_tokens if st.usage_seen else None)
        itl = gaps["itl_ms"]
        if itl:
            sample.extra["itl_ms_mean"] = statistics.mean(itl)
            sample.extra["itl_ms_p50"] = _pct(itl, 50)
            sample.extra["itl_ms_p99"] = _pct(itl, 99)
            # Raw gaps: pooled across requests by the summary (issue #1).
            sample.series[ITL_SERIES] = itl
        sample.meta["itl_mode"] = gaps["mode"]
        if gaps.get("usage_mismatch"):
            sample.meta["itl_usage_mismatch"] = True
        # The per-event (per network read) gaps the headline used to be: kept
        # next to it so a change in event batching is visible, not hidden.
        chunk = gaps["itl_chunk_ms"]
        if chunk:
            sample.extra["itl_chunk_ms_mean"] = statistics.mean(chunk)
            sample.extra["itl_chunk_ms_p50"] = _pct(chunk, 50)
            sample.extra["itl_chunk_ms_p99"] = _pct(chunk, 99)
        sample.extra["itl_coalesced_events"] = float(gaps["coalesced_events"])
        kpe = gaps["tokens_per_event"]
        if kpe:
            sample.extra["itl_tokens_per_event_mean"] = statistics.mean(kpe)

    sample.extra["tokens_out"] = float(completion_tokens)
    # Surface the reasoning/content breakdown from `usage` when the server
    # reports it, so a run can see how much of the generation was thinking
    # vs. the visible answer (#14).
    if st.reasoning_tokens is not None:
        sample.extra["reasoning_tokens"] = float(st.reasoning_tokens)
        if completion_tokens >= st.reasoning_tokens:
            sample.extra["content_tokens"] = (
                float(completion_tokens) - float(st.reasoning_tokens))
    if prompt_tokens is not None:
        sample.extra["prompt_tokens"] = float(prompt_tokens)
    if cached_tokens is not None:
        sample.extra["cached_tokens"] = float(cached_tokens)
    if cached_tokens is not None and prompt_tokens:
        sample.extra["cached_ratio"] = float(cached_tokens) / float(prompt_tokens)
    # Cache class (issue #2): device_hit / restore / partial / miss. Only
    # classifiable when the server reported a usage block.
    if response.ok and prompt_tokens is not None:
        info = CacheClassInfo(
            prompt_tokens=prompt_tokens,
            cached_tokens=cached_tokens or 0,
            restored_tokens=st.restored_tokens,
            hint=cache_hint,
            usage=_usage_cache_fields(usage_raw),
            headers=dict(response.headers or {}),
        )
        cache_class = (
            cache_classifier(info) if cache_classifier
            else classify_cache(info, cache_full_hit_ratio)
        )
        if cache_class is not None:
            sample.meta["cache_class"] = cache_class
        if cache_hint is not None:
            sample.meta["cache_hint"] = cache_hint
    # Dataset workloads can provide a pre-run prompt-size estimate and a
    # RAG packing depth through Request.meta. Preserve them as numeric
    # metrics even when an endpoint omits usage information.
    for metric in ("prompt_tokens_hint", "prefix_tokens_hint", "rag_depth"):
        value = request.meta.get(metric)
        if isinstance(value, (int, float)):
            sample.extra[metric] = float(value)
    # Numeric trace fields (trace_input_tokens, trace_output_tokens,
    # trace_blocks, ...) become metrics too, so a replayed run reports the
    # trace's token distribution next to what the server measured (#6).
    for key, value in request.meta.items():
        if key.startswith("trace_") and isinstance(value, (int, float)) and not isinstance(value, bool):
            sample.extra[key] = float(value)

    if completion_tokens > 0 and ttft is not None and response.elapsed_s > ttft:
        sample.extra["tokens_per_s"] = completion_tokens / (response.elapsed_s - ttft)
    elif completion_tokens > 0 and response.elapsed_s > 0:
        sample.extra["tokens_per_s"] = completion_tokens / response.elapsed_s

    if finish_reason:
        sample.meta["finish_reason"] = finish_reason
    if response.ok and completion_tokens == 0:
        sample.ok = False
        sample.error = sample.error or "no tokens received"

    return sample

# --------------------------------------------------------------------------- #
# Inter-token latency (issue #17)                                             #
# --------------------------------------------------------------------------- #

# Events closer together than this arrived in one network read (or in a burst
# the client only saw once): the gap is split over the read's tokens.
ITL_BURST_GAP_S = 2e-4


def token_gaps(arrivals: list[float], tokens_per_event: Optional[list] = None, *,
               completion_tokens: Optional[int] = None,
               burst_gap_s: float = ITL_BURST_GAP_S) -> dict:
    """Per-token inter-token latency from the arrival times of streamed events.

    ``arrivals``: monotonic arrival times (s) of the token-bearing events.
    How many tokens each event carried —

    * ``tokens_per_event``: exactly (sglang's cumulative ``completion_tokens``
      per event; index 0 unused);
    * else the final ``usage.completion_tokens``: fewer tokens than events
      means the usage undercounts (reasoning tokens not included, say) and
      each event stays one token; more means the server batched tokens per
      event and the surplus is spread evenly (``mode = per_token_estimated``);
    * else one token per event (``mode = per_event`` — the client cannot know
      whether an event carried several tokens).

    Consecutive events within ``burst_gap_s`` of their burst's first event
    came in one read; the gap to the previous burst is divided by the burst's
    token count instead of landing whole on its first token. ``itl_chunk_ms``
    keeps the plain per-event gaps for comparison.

    Returns ``{"itl_ms", "itl_chunk_ms", "mode", "tokens_per_event",
    "coalesced_events", "burst_gap_s" [, "usage_mismatch"]}``.
    """
    n = len(arrivals)
    out: dict = {"itl_ms": [], "itl_chunk_ms": [], "mode": "per_event",
                 "tokens_per_event": [1] * n, "coalesced_events": 0, "burst_gap_s": burst_gap_s}
    if n < 2:
        return out
    k = [1] * n
    exact = None
    if tokens_per_event is not None and len(tokens_per_event) == n:
        exact = [max(int(x), 0) for x in tokens_per_event]
        if sum(exact[1:]) <= 0:  # counts never advanced: no information
            exact = None
    if exact is not None:
        k = exact
        out["mode"] = "per_token"
    elif completion_tokens is not None:
        if completion_tokens >= n:
            base, rem = divmod(int(completion_tokens), n)
            k = [base + (1 if i < rem else 0) for i in range(n)]
            out["mode"] = "per_token" if completion_tokens == n else "per_token_estimated"
        else:
            out["mode"] = "per_token"
            out["usage_mismatch"] = True
    out["tokens_per_event"] = k
    out["itl_chunk_ms"] = [(arrivals[i] - arrivals[i - 1]) * 1000.0 for i in range(1, n)]
    # Group events into bursts (reads).
    groups: list[tuple[float, int]] = []  # (first arrival, tokens in the burst)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and arrivals[j + 1] - arrivals[i] <= burst_gap_s:
            j += 1
        groups.append((arrivals[i], sum(k[i:j + 1])))
        i = j + 1
    out["coalesced_events"] = n - len(groups)
    gaps: list[float] = []
    pending_ms = 0.0
    for g in range(1, len(groups)):
        gap_ms = (groups[g][0] - groups[g - 1][0]) * 1000.0 + pending_ms
        pending_ms = 0.0
        tokens = groups[g][1]
        if tokens <= 0:
            pending_ms = gap_ms  # an event that carried no tokens: its time goes to the next
            continue
        gaps.extend([gap_ms / tokens] * tokens)
    if pending_ms and gaps:
        gaps[-1] += pending_ms
    out["itl_ms"] = gaps
    return out


# --------------------------------------------------------------------------- #
# Cache classification (issue #2)                                             #
# --------------------------------------------------------------------------- #

CACHE_CLASSES = ("device_hit", "restore", "partial", "miss", "unknown")

# Keys (usage block, prompt_tokens_details, response headers) a server may use
# to say which tier served the cached prefix. Checked case-insensitively for
# headers; the first non-empty value wins.
_CACHE_SOURCE_KEYS = ("cache_source", "cache_tier", "kv_cache_source", "kv_cache_tier",
                      "prefix_cache_source", "cached_tokens_source", "cache_hit_tier")


class CacheClassInfo:
    """What the response said about the prefix cache, for :func:`classify_cache`."""

    def __init__(self, prompt_tokens: int, cached_tokens: int = 0,
                 restored_tokens: Optional[int] = None, hint: Optional[str] = None,
                 usage: Optional[dict] = None, headers: Optional[dict] = None):
        self.prompt_tokens = int(prompt_tokens)
        self.cached_tokens = int(cached_tokens)
        self.restored_tokens = restored_tokens
        self.hint = hint
        self.usage = dict(usage or {})
        self.headers = dict(headers or {})

    @property
    def cached_ratio(self) -> float:
        return self.cached_tokens / self.prompt_tokens if self.prompt_tokens else 0.0


def _cache_source_of(src: dict) -> Optional[str]:
    if not isinstance(src, dict):
        return None
    # usage fields are verbatim; HTTP headers may be prefixed (X-Cache-Source)
    # and use either separator.
    lowered = {str(k).lower().replace("-", "_"): v for k, v in src.items()}
    for key in _CACHE_SOURCE_KEYS:
        v = lowered.get(key)
        if v in (None, "", 0, False):
            for k, hv in lowered.items():
                if k.endswith("_" + key) and hv not in (None, "", 0, False):
                    v = hv
                    break
        if v not in (None, "", 0, False):
            return str(v)
    return None


def _usage_cache_fields(raw: dict) -> dict:
    return {k: v for k, v in raw.items() if "cache" in k.lower() or "restored" in k.lower()}


def classify_cache(info: CacheClassInfo, full_hit_ratio: float = 0.9) -> Optional[str]:
    """Map a response's cache information to one of :data:`CACHE_CLASSES`.

    * ``miss``     — nothing was cached.
    * ``restore``  — the server said the cached prefix was (partly) restored
                     from a remote/host tier: ``restored_tokens > 0``, or a
                     ``cache_hint`` naming a restore tier (restore words win
                     over device words: a hint such as ``host_dram`` must not
                     read as a device hit, issue #17).
    * ``device_hit`` — cached tokens cover at least ``full_hit_ratio`` of the
                     prompt, no restore was involved, and the hint (if any)
                     names a device tier or no tier at all.
    * ``partial``  — some tokens cached, below the full-hit ratio.
    * ``unknown``  — a full hit by ratio whose hint names neither kind of
                     tier (a vocabulary this module does not know): recorded
                     as such rather than guessed ``device_hit`` (issue #17);
                     plug a ``cache_classifier`` for the server's vocabulary.

    The ratio is a heuristic: a hit that stops short of the prompt is usually
    the genuinely new suffix (the new turn), which is why it is configurable.
    Returns ``None`` only for ``prompt_tokens == 0``.
    """
    if info is None or info.prompt_tokens <= 0:
        return None
    if info.cached_tokens <= 0 and not (info.restored_tokens or 0):
        return "miss"
    tier = hint_tier(info.hint)
    if info.restored_tokens or tier == "restore":
        return "restore"
    if info.cached_ratio < full_hit_ratio:
        return "partial"
    return "unknown" if tier == "unknown" else "device_hit"


# Tier vocabularies a server may use in ``cache_source`` / ``cache_tier`` /
# ``X-Cache-Source``. Matched as whole words — ``host_dram``, ``HostDram``,
# ``HOST-DRAM`` all split into words — so ``dram`` (host memory) is a restore
# tier, only ``hbm`` / ``gpu`` / ``device`` / ``local`` / ``vram`` are device
# tiers, and ``localhost`` does not read as ``local``. Extend them (or plug a
# ``cache_classifier``) for a server with its own vocabulary; a hint with no
# word from either list is ``unknown`` (issue #17).
RESTORE_WORDS = frozenset({
    "restore", "restored", "restoration", "remote", "host", "cpu", "dram", "kvstore",
    "store", "mooncake", "lmcache", "hicache", "ssd", "nvme", "disk", "peer", "mixed",
    "offload", "memory", "transfer",
})
DEVICE_WORDS = frozenset({"device", "gpu", "hbm", "local", "vram", "devicekv"})
# Values that carry no tier information at all.
NO_TIER_HINTS = frozenset({"", "none", "unknown", "n/a", "na", "null"})
_WORD_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[a-z]+|\d+")


def hint_words(hint: str) -> set[str]:
    """Lower-cased words of a hint: ``host_dram``/``HostDram``/``HOST-DRAM`` -> ``{host, dram}``."""
    return {w.lower() for w in _WORD_RE.findall(hint)}


def hint_tier(hint: Optional[str]) -> Optional[str]:
    """``"restore"`` | ``"device"`` | ``"unknown"`` | ``None``: which kind of
    tier a cache-source hint names (``None``: the hint says nothing, e.g.
    ``none``). Restore words win over device words — a ``host_dram`` hit must
    never read as a device hit (issue #17)."""
    if hint is None or str(hint).strip().lower() in NO_TIER_HINTS:
        return None
    words = hint_words(str(hint))
    if words & RESTORE_WORDS:
        return "restore"
    if words & DEVICE_WORDS:
        return "device"
    return "unknown"


def _hint_is_restore(hint: str) -> bool:
    """A hint names a restore tier (kept for callers that predate :func:`hint_tier`)."""
    return hint_tier(hint) == "restore"


def _first_index(times: list[float], t: float) -> Optional[int]:
    for i, v in enumerate(times):
        if v == t:
            return i
    return None


def _warn_missing_usage_once(request: Request) -> None:

    """Warn once if the request asked for usage but the response carried none.

    Only fires when ``stream_options.include_usage`` was set on the request, so
    endpoints that were never asked for usage don't trigger noise.
    """
    global _warned_missing_usage
    if _warned_missing_usage:
        return
    body = request.json if isinstance(request.json, dict) else {}
    if not (body.get("stream_options") or {}).get("include_usage"):
        return
    _warned_missing_usage = True
    _logger.warning(
        "openai-chat: requested stream_options.include_usage but no usage block "
        "was parsed from the streamed response; prompt_tokens/cached_tokens will "
        "be absent and prefix-cache hit rate will read as 0. Check that the "
        "endpoint emits a usage event (e.g. SGLang needs --enable-cache-report). "
        "Further identical warnings are suppressed."
    )


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


def _first_env(keys: tuple[str, ...]) -> Optional[str]:
    for k in keys:
        v = os.environ.get(k)
        if v:
            return v
    return None


# --------------------------------------------------------------------------- #
# /v1/completions (issue #11)                                                  #
# --------------------------------------------------------------------------- #

_COMPLETIONS_BODY_KEYS = _OPENAI_BODY_KEYS - {"messages", "tools", "tool_choice", "response_format"} | {
    "prompt", "best_of", "user", "suffix", "echo", "logprobs", "top_logprobs",
}


class OpenAICompletionsWorkloadType(WorkloadType):
    """OpenAI-compatible ``/v1/completions`` with SSE streaming (vLLM, SGLang,
    TGI's ``/v1/completions``, OpenAI itself).

    Item interpretation:
      * ``str``            → ``{"prompt": item}``
      * ``list[int]``      → ``{"prompt": [token ids]}`` — pre-tokenized
        prompt; the server skips its tokenizer entirely
      * ``dict``           → merged into the body; ``prompt`` (str or ids),
        ``max_tokens`` and sampling keys; a ``meta`` key is carried on the
        request (recorded, never sent). ``messages`` is rejected — use
        ``type: openai`` for chat.

    Metrics are the ones documented at the top of this module (``ttft_s``,
    ``itl_ms_*``, ``tokens_out``, ``prompt_tokens``, ``cached_ratio``,
    ``cache_class``, ``tokens_per_s``). ``echo`` is refused: echoing the
    prompt back would stamp TTFT on the prompt and inflate token counts.
    ``passthrough_meta`` behaves as for chat (allowlisted body keys, a row
    ``model`` kept as ``meta.model_label``).
    """

    streaming = True
    name = "openai-completions"

    def __init__(
        self,
        url: str,
        model: str,
        max_tokens: int = 128,
        temperature: float = 0.0,
        extra_body: Optional[dict] = None,
        headers: Optional[dict[str, str]] = None,
        api_key: Optional[str] = None,
        timeout_s: Optional[float] = 600.0,
        name: str = "openai-completions",
        passthrough_meta: bool = False,
        cache_classifier: Optional[Callable[[CacheClassInfo], Optional[str]]] = None,
        cache_full_hit_ratio: float = 0.9,
        **sampling: Any,
    ):
        if not 0.0 < cache_full_hit_ratio <= 1.0:
            raise ValueError(f"cache_full_hit_ratio must be in (0, 1], got {cache_full_hit_ratio!r}")
        merged: dict[str, Any] = dict(extra_body or {})
        merged.update(sampling)
        if merged.get("echo"):
            raise ValueError("echo=True would stream the prompt back and corrupt ttft/itl/token counts")
        self.name = name
        self._url = url
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._passthrough_meta = passthrough_meta
        self._extra_body = merged
        self._timeout_s = timeout_s
        self._cache_classifier = cache_classifier
        self._cache_full_hit_ratio = float(cache_full_hit_ratio)
        hdrs = dict(headers or {})
        hdrs.setdefault("Content-Type", "application/json")
        hdrs.setdefault("Accept", "text/event-stream")
        if api_key:
            hdrs.setdefault("Authorization", f"Bearer {api_key}")
        self._headers = hdrs

    @property
    def model(self) -> Optional[str]:
        return self._model

    @property
    def url(self) -> str:
        return self._url

    @classmethod
    def from_env(cls, url: Optional[str] = None, model: Optional[str] = None,
                 api_key: Optional[str] = None, dotenv_path: Optional[str] = ".env",
                 endpoint_path: str = "completions", **kwargs: Any) -> "OpenAICompletionsWorkloadType":
        chat = OpenAIChatWorkloadType.from_env(url=url, model=model, api_key=api_key,
                                               dotenv_path=dotenv_path,
                                               endpoint_path=endpoint_path, **kwargs)
        return cls(url=chat._url, model=chat.model, api_key=api_key, **kwargs)

    def _split_passthrough(self, item: dict) -> tuple[dict, dict]:
        allowed = _COMPLETIONS_BODY_KEYS | set(self._extra_body.keys())
        body_part: dict[str, Any] = {}
        meta_part: dict[str, Any] = {}
        for k, v in item.items():
            if k == "meta" and isinstance(v, dict):
                meta_part.update(v)
            elif k == "model":
                meta_part["model_label"] = v
            elif k == "max_new_tokens" and "max_tokens" not in item:
                body_part["max_tokens"] = v
            elif k in allowed:
                body_part[k] = v
            else:
                meta_part[k] = v
        return body_part, meta_part

    async def make_request(self, item: Any) -> Request:
        body: dict[str, Any] = {
            "model": self._model,
            "stream": True,
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
            **self._extra_body,
        }
        body.setdefault("stream_options", {"include_usage": True})
        meta_extra: dict[str, Any] = {}

        if item is None:
            body["prompt"] = ""
        elif isinstance(item, str):
            body["prompt"] = item
        elif isinstance(item, list) and all(isinstance(t, int) and not isinstance(t, bool) for t in item):
            body["prompt"] = list(item)
        elif isinstance(item, dict):
            if self._passthrough_meta:
                item, meta_extra = self._split_passthrough(item)
            else:
                item = dict(item)
                item_meta = item.pop("meta", None)
                if isinstance(item_meta, dict):
                    meta_extra.update(item_meta)
            if "messages" in item and "prompt" not in item:
                raise TypeError("OpenAICompletionsWorkloadType got chat `messages`; use type: openai")
            body.update(item)
            body.setdefault("prompt", "")
        else:
            raise TypeError(f"OpenAICompletionsWorkloadType cannot interpret item {type(item).__name__}")

        prompt = body["prompt"]
        meta: dict[str, Any] = dict(meta_extra)
        is_ids = isinstance(prompt, list)
        meta["prompt_kind"] = "token_ids" if is_ids else "text"
        # The prompt itself is not copied into meta (token-id prompts run to
        # tens of thousands of ints per request); its size is.
        meta["prompt_len"] = len(prompt) if is_ids else len(str(prompt))
        meta["prompt_tokens_hint"] = len(prompt) if is_ids else meta.get("prompt_tokens_hint")
        meta["max_tokens"] = body.get("max_tokens")
        return Request(
            method="POST",
            url=self._url,
            headers=dict(self._headers),
            json=body,
            timeout_s=self._timeout_s,
            meta=meta,
        )

    async def make_sample(self, item: Any, request: Request, response: Response,
                          start_ts: float) -> Sample:
        sample = await super().make_sample(item, request, response, start_ts)
        stats = parse_completions_stream(response.stream_chunks or [], response.stream_chunk_times or [])
        return finish_llm_sample(
            sample, request, response, stats, ttft_token="any",
            cache_classifier=self._cache_classifier,
            cache_full_hit_ratio=self._cache_full_hit_ratio)
