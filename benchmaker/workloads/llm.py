"""OpenAI-compatible chat-completions workload-type (with SSE streaming).

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
import statistics
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

        chunks = response.stream_chunks or []
        chunk_times = response.stream_chunk_times or []
        # First token of *any* kind (reasoning or content): the engine-cost
        # signal a serving benchmark wants — when the server began producing
        # output, isolating prefill→decode from the model's reasoning policy.
        first_token: Optional[float] = None
        # First *visible* (content) token: the latency a user perceives before
        # an answer appears. Tracked separately so it's surfaced as
        # `content_ttft_s` even when `ttft_token="any"` (#14).
        first_content_token: Optional[float] = None
        # Every token the stream produced, reasoning and content alike, so
        # inter-token latency reflects true decode cadence across the whole
        # generation instead of conflating the reasoning phase.
        token_arrival_times: list[float] = []
        token_kinds: list[str] = []
        usage_completion_tokens: Optional[int] = None
        reasoning_tokens: Optional[int] = None
        prompt_tokens: Optional[int] = None
        cached_tokens: Optional[int] = None
        restored_tokens: Optional[int] = None
        cache_hint: Optional[str] = None
        finish_reason: Optional[str] = None
        usage_seen = False
        usage_raw: dict = {}

        # Reassemble across chunk boundaries first: httpx2's aiter_raw()
        # yields bytes at arbitrary offsets, so the final `usage` event is
        # often split in two and would be silently dropped if each chunk were
        # parsed on its own (#13).
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
            if not isinstance(obj, dict):
                continue
            if obj.get("usage"):
                usage_seen = True
                u = obj["usage"]
                usage_raw = dict(u) if isinstance(u, dict) else {}
                if u.get("completion_tokens") is not None:
                    usage_completion_tokens = int(u["completion_tokens"])
                if u.get("prompt_tokens") is not None:
                    prompt_tokens = int(u["prompt_tokens"])
                details = u.get("prompt_tokens_details")
                if isinstance(details, dict) and details.get("cached_tokens") is not None:
                    cached_tokens = int(details["cached_tokens"])
                elif u.get("cached_tokens") is not None:
                    cached_tokens = int(u["cached_tokens"])
                # Which tier served the cached prefix (device / host / remote),
                # when the server says so: usage, prompt_tokens_details or
                # response headers (issue #2). Also restored-from-host counts.
                for src in (u, details if isinstance(details, dict) else None):
                    if isinstance(src, dict):
                        if restored_tokens is None and src.get("restored_tokens") is not None:
                            restored_tokens = int(src["restored_tokens"])
                        if cache_hint is None:
                            cache_hint = _cache_source_of(src)
                if cache_hint is None:
                    cache_hint = _cache_source_of(dict(response.headers or {}))
                # Thinking models report generated reasoning tokens separately
                # from visible content; surfacing the breakdown keeps tokens_out
                # honest and exposes the reasoning share (#14).
                details = u.get("completion_tokens_details")
                if isinstance(details, dict) and details.get("reasoning_tokens") is not None:
                    reasoning_tokens = int(details["reasoning_tokens"])
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
                    if first_token is None:
                        first_token = t
                    if first_content_token is None:
                        first_content_token = t
                    token_arrival_times.append(t)
                    token_kinds.append("content")
                if reasoning_piece:
                    if first_token is None:
                        first_token = t
                    token_arrival_times.append(t)
                    token_kinds.append("reasoning")
                if ch.get("finish_reason"):
                    finish_reason = ch["finish_reason"]

        # Loud-but-once diagnostic: a benchmark that requested usage yet parsed
        # none produces a structurally-zero cache hit rate — surface it instead
        # of silently recording misleading metrics.
        if response.ok and not usage_seen:
            _warn_missing_usage_once(request)

        # When the endpoint omits `usage`, fall back to the count of streamed
        # tokens — which now includes reasoning tokens, so a thinking model is
        # no longer undercounted by its entire chain-of-thought (#14).
        completion_tokens = (
            usage_completion_tokens
            if usage_completion_tokens is not None
            else len(token_arrival_times)
        )

        # `ttft_token` selects which token the headline `ttft_s` is measured
        # to. The first-content time is always surfaced separately as
        # `content_ttft_s` when it is a distinct instant, so a serving run
        # (ttft = first-any token) also captures perceived latency.
        ttft = first_content_token if self._ttft_token == "content" else first_token
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
            itl = [
                (token_arrival_times[i] - token_arrival_times[i - 1]) * 1000.0
                for i in range(1, len(token_arrival_times))
            ]
            sample.extra["itl_ms_mean"] = statistics.mean(itl)
            sample.extra["itl_ms_p50"] = _pct(itl, 50)
            sample.extra["itl_ms_p99"] = _pct(itl, 99)
            # Raw gaps: pooled across requests by the summary (issue #1).
            sample.series[ITL_SERIES] = itl

        sample.extra["tokens_out"] = float(completion_tokens)
        # Surface the reasoning/content breakdown from `usage` when the server
        # reports it, so a run can see how much of the generation was thinking
        # vs. the visible answer (#14).
        if reasoning_tokens is not None:
            sample.extra["reasoning_tokens"] = float(reasoning_tokens)
            if completion_tokens >= reasoning_tokens:
                sample.extra["content_tokens"] = (
                    float(completion_tokens) - float(reasoning_tokens))
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
                restored_tokens=restored_tokens,
                hint=cache_hint,
                usage=_usage_cache_fields(usage_raw),
                headers=dict(response.headers or {}),
            )
            cache_class = (
                self._cache_classifier(info) if self._cache_classifier
                else classify_cache(info, self._cache_full_hit_ratio)
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
# Cache classification (issue #2)                                             #
# --------------------------------------------------------------------------- #

CACHE_CLASSES = ("device_hit", "restore", "partial", "miss")

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
                     from a remote/host tier: a ``cache_hint`` naming a
                     restore/remote/host tier, or ``restored_tokens > 0``.
    * ``device_hit`` — cached tokens cover at least ``full_hit_ratio`` of the
                     prompt and no restore was involved: the prefix cache
                     did its whole job from the device.
    * ``partial``  — some tokens cached, below the full-hit ratio.

    The ratio is a heuristic: a hit that stops short of the prompt is usually
    the genuinely new suffix (the new turn), which is why it is configurable.
    Returns ``None`` only for ``prompt_tokens == 0``.
    """
    if info is None or info.prompt_tokens <= 0:
        return None
    if info.cached_tokens <= 0 and not (info.restored_tokens or 0):
        return "miss"
    if info.restored_tokens or (info.hint and _hint_is_restore(info.hint)):
        return "restore"
    if info.cached_ratio >= full_hit_ratio:
        return "device_hit"
    return "partial"


_RESTORE_WORDS = ("restore", "remote", "host", "cpu", "kvstore", "store", "mooncake", "lmcache", "hicache")
_DEVICE_WORDS = ("device", "gpu", "hbm", "local", "dram", "vram")


def _hint_is_restore(hint: str) -> bool:
    """A hint names a restore tier if it mentions one and not a device tier."""
    h = hint.lower()
    if any(w in h for w in _DEVICE_WORDS):
        return False
    return any(w in h for w in _RESTORE_WORDS)


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
