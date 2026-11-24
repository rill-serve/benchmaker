"""SGLang native ``/generate`` workload-type (SSE streaming).

SGLang's /generate is not OpenAI-compatible: the body is
``{"text": ..., "sampling_params": {...}, "stream": true}`` and the streamed
events carry *cumulative* ``text`` plus a ``meta_info`` object with
``prompt_tokens``, ``completion_tokens``, ``cached_tokens`` and ``finish_reason``.
Prefer the OpenAI chat path where possible; this type exists for raw-text
parity checks where /generate is the only endpoint.

Item interpretation:
  * ``str``   -> ``{"text": item}``
  * ``dict``  -> ``text``/``input_ids`` pulled out; remaining sampling keys merged
                into ``sampling_params``. With ``passthrough_meta`` (default on
                via the recipe), non-sampling keys are recorded into
                ``Request.meta`` instead of sent.

Captured metrics: ttft_s, itl_ms_mean/p50/p99, tokens_out, prompt_tokens,
cached_tokens, cached_ratio, tokens_per_s, meta.finish_reason,
meta.cache_class (device_hit/restore/partial/miss from ``meta_info``), and
the raw per-token gap series ``itl_ms`` pooled by the summary (#1).
"""

from __future__ import annotations

import json
import os
import statistics
from typing import Any, Callable, Optional

from benchmaker.env import load_dotenv
from benchmaker.core.metrics import ITL_SERIES
from benchmaker.workloads.llm import token_gaps
from benchmaker.core.types import Request, Response, Sample
from benchmaker.workloads._sse import reassemble_sse_lines
from benchmaker.workloads.base import WorkloadType
from benchmaker.workloads.llm import CacheClassInfo, _cache_source_of, _pct, classify_cache


_ENV_BASE_URL_KEYS = ("SGLANG_API_BASE_URL", "SGLANG_BASE_URL")

# Keys that belong in sampling_params (everything else in a dict item, in
# passthrough mode, is recorded into Request.meta).
_SAMPLING_KEYS = frozenset({
    "temperature", "max_new_tokens", "top_p", "top_k", "min_p", "stop",
    "stop_token_ids", "frequency_penalty", "presence_penalty",
    "repetition_penalty", "ignore_eos", "skip_special_tokens", "n", "seed",
    "min_new_tokens", "regex", "json_schema", "ebnf",
})


class SGLangGenerateWorkloadType(WorkloadType):
    streaming = True
    name = "sglang-generate"

    def __init__(
        self,
        url: str,
        max_tokens: int = 128,
        temperature: float = 0.0,
        extra_body: Optional[dict] = None,
        headers: Optional[dict[str, str]] = None,
        timeout_s: Optional[float] = 600.0,
        name: str = "sglang-generate",
        passthrough_meta: bool = False,
        cache_classifier: Optional[Callable[["CacheClassInfo"], Optional[str]]] = None,
        cache_full_hit_ratio: float = 0.9,
        **sampling: Any,
    ):
        if not 0.0 < cache_full_hit_ratio <= 1.0:
            raise ValueError(f"cache_full_hit_ratio must be in (0, 1], got {cache_full_hit_ratio!r}")
        self.name = name
        self._cache_classifier = cache_classifier
        self._cache_full_hit_ratio = float(cache_full_hit_ratio)
        self._url = url
        self._max_tokens = max_tokens
        self._temperature = temperature
        merged: dict[str, Any] = dict(extra_body or {})
        merged.update(sampling)
        self._extra_sampling = merged
        self._timeout_s = timeout_s
        self._passthrough_meta = passthrough_meta
        hdrs = dict(headers or {})
        hdrs.setdefault("Content-Type", "application/json")
        hdrs.setdefault("Accept", "text/event-stream")
        self._headers = hdrs

    @property
    def model(self) -> None:
        """``/generate`` has no model field; the served model is the server's."""
        return None

    def apply_overrides(self, req: Request, overrides: dict[str, Any]) -> Request:
        # The SGLang body carries no model name; a per-lane model override can
        # only be recorded as a label (the endpoint decides what is served).
        if overrides.get("model"):
            req.meta["model"] = overrides["model"]
        return req

    @classmethod
    def from_env(
        cls,
        url: Optional[str] = None,
        dotenv_path: Optional[str] = ".env",
        endpoint_path: str = "generate",
        **kwargs: Any,
    ) -> "SGLangGenerateWorkloadType":
        if dotenv_path:
            load_dotenv(dotenv_path)
        if url is None:
            base = None
            for k in _ENV_BASE_URL_KEYS:
                base = os.environ.get(k)
                if base:
                    break
            if not base:
                raise ValueError(
                    "SGLANG_API_BASE_URL (or SGLANG_BASE_URL) is not set; "
                    "pass url=... or set the env var.")
            url = base.rstrip("/") + "/" + endpoint_path.lstrip("/")
        return cls(url=url, **kwargs)

    async def make_request(self, item: Any) -> Request:
        sampling: dict[str, Any] = {
            "temperature": self._temperature,
            "max_new_tokens": self._max_tokens,
            **self._extra_sampling,
        }
        body: dict[str, Any] = {"stream": True}
        meta_extra: dict[str, Any] = {}

        if item is None:
            body["text"] = ""
        elif isinstance(item, str):
            body["text"] = item
        elif isinstance(item, dict):
            item = dict(item)
            item_meta = item.pop("meta", None)  # dataset tags -> sample.meta, not the body
            if isinstance(item_meta, dict):
                meta_extra.update(item_meta)
            if "text" in item:
                body["text"] = item.pop("text")
            elif "input_ids" in item:
                body["input_ids"] = item.pop("input_ids")
            if "max_tokens" in item and "max_new_tokens" not in item:
                item["max_new_tokens"] = item.pop("max_tokens")
            inner = item.pop("sampling_params", None)
            if isinstance(inner, dict):
                sampling.update(inner)
            for k, v in item.items():
                if k in _SAMPLING_KEYS:
                    sampling[k] = v
                elif self._passthrough_meta:
                    if k == "meta" and isinstance(v, dict):
                        meta_extra.update(v)
                    elif k == "model":
                        meta_extra["model_label"] = v
                    else:
                        meta_extra[k] = v
                else:
                    body[k] = v
            if "input_ids" not in body:
                body.setdefault("text", "")
        else:
            raise TypeError(
                f"SGLangGenerateWorkloadType cannot interpret {type(item).__name__}")

        body["sampling_params"] = sampling
        # Row metadata first, then canonical keys last so they always win over
        # any same-named row field (mirrors OpenAIChatWorkloadType).
        meta: dict[str, Any] = dict(meta_extra)
        meta["prompt_text"] = body.get("text")
        meta["max_tokens"] = sampling.get("max_new_tokens")
        return Request(method="POST", url=self._url, headers=dict(self._headers),
                       json=body, timeout_s=self._timeout_s, meta=meta)

    async def make_sample(self, item: Any, request: Request, response: Response,
                          start_ts: float) -> Sample:
        sample = await super().make_sample(item, request, response, start_ts)
        chunks = response.stream_chunks or []
        chunk_times = response.stream_chunk_times or []
        ttft: Optional[float] = None
        arrivals: list[float] = []
        cumulative: list[int] = []  # meta_info.completion_tokens at each event (issue #17)
        last_text = ""
        completion_tokens: Optional[int] = None
        prompt_tokens: Optional[int] = None
        cached_tokens: Optional[int] = None
        finish_reason: Any = None
        mi_raw: dict = {}

        # Reassemble across chunk boundaries: httpx2's aiter_raw() yields
        # bytes at arbitrary offsets, so a `meta_info` event can be split
        # across chunks and would be silently dropped if each chunk were
        # parsed independently (#13).
        for line, t in reassemble_sse_lines(chunks, chunk_times):
            line = line.strip()
            if line.startswith(b"data:"):
                line = line[5:].strip()
            if not line or line == b"[DONE]":
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            text = obj.get("text")
            mi = obj.get("meta_info")
            mi_ct = mi.get("completion_tokens") if isinstance(mi, dict) else None
            if isinstance(text, str) and text != last_text:
                if ttft is None:
                    ttft = t
                arrivals.append(t)
                # The event's own count when it carried one, else the last seen.
                cumulative.append(int(mi_ct) if mi_ct is not None else (cumulative[-1] if cumulative else 0))
                last_text = text
            if isinstance(mi, dict):
                mi_raw = dict(mi)
                if mi_ct is not None:
                    completion_tokens = int(mi_ct)
                if mi.get("prompt_tokens") is not None:
                    prompt_tokens = int(mi["prompt_tokens"])
                if mi.get("cached_tokens") is not None:
                    cached_tokens = int(mi["cached_tokens"])
                fr = mi.get("finish_reason")
                if fr is not None:
                    finish_reason = fr

        out_tokens = completion_tokens if completion_tokens is not None else len(arrivals)

        if ttft is not None:
            sample.extra["ttft_s"] = ttft
            sample.meta["ttft_token_kind"] = "content"
            sample.meta["ttft_token_index"] = 0
            sample.extra["ttft_token_index"] = 0.0
        if len(arrivals) >= 2:
            # Per-token gaps (issue #17): sglang reports the cumulative
            # completion_tokens with every event, so each event's exact token
            # count is known; a read that delivered several events splits its
            # gap over them.
            per_event = [cumulative[i] - cumulative[i - 1] for i in range(1, len(cumulative))]
            gaps = token_gaps(arrivals, [0] + per_event, completion_tokens=completion_tokens)
            itl = gaps["itl_ms"]
            if itl:
                sample.extra["itl_ms_mean"] = statistics.mean(itl)
                sample.extra["itl_ms_p50"] = _pct(itl, 50)
                sample.extra["itl_ms_p99"] = _pct(itl, 99)
                sample.series[ITL_SERIES] = itl  # pooled across requests (#1)
            sample.meta["itl_mode"] = gaps["mode"]
            chunk = gaps["itl_chunk_ms"]
            if chunk:
                sample.extra["itl_chunk_ms_mean"] = statistics.mean(chunk)
                sample.extra["itl_chunk_ms_p50"] = _pct(chunk, 50)
                sample.extra["itl_chunk_ms_p99"] = _pct(chunk, 99)
            sample.extra["itl_coalesced_events"] = float(gaps["coalesced_events"])
        sample.extra["tokens_out"] = float(out_tokens)
        if prompt_tokens is not None:
            sample.extra["prompt_tokens"] = float(prompt_tokens)
        if cached_tokens is not None:
            sample.extra["cached_tokens"] = float(cached_tokens)
            if prompt_tokens:
                sample.extra["cached_ratio"] = float(cached_tokens) / float(prompt_tokens)
        if response.ok and prompt_tokens is not None:
            info = CacheClassInfo(prompt_tokens=prompt_tokens, cached_tokens=cached_tokens or 0,
                                  hint=_cache_source_of(mi_raw))
            if info.hint is not None:
                sample.meta["cache_hint"] = info.hint
            cache_class = (self._cache_classifier(info) if self._cache_classifier
                           else classify_cache(info, self._cache_full_hit_ratio))
            if cache_class is not None:
                sample.meta["cache_class"] = cache_class
        if out_tokens > 0 and ttft is not None and response.elapsed_s > ttft:
            sample.extra["tokens_per_s"] = out_tokens / (response.elapsed_s - ttft)
        elif out_tokens > 0 and response.elapsed_s > 0:
            sample.extra["tokens_per_s"] = out_tokens / response.elapsed_s

        if finish_reason is not None:
            sample.meta["finish_reason"] = (
                finish_reason.get("type") if isinstance(finish_reason, dict)
                else finish_reason)
        if response.ok and out_tokens == 0:
            sample.ok = False
            sample.error = sample.error or "no tokens received"
        return sample
