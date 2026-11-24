"""Capacity-revisit workload: warm pass over N long documents, then revisits (issue #5).

Measures what a prefix cache does when the working set is larger than the
device cache: a *warm* pass over ``N`` distinct long documents (every prompt
is a cold miss — prefill throughput at full batch), then ``K`` *revisit*
passes over the same documents. If a revisit is served from the device cache,
TTFT drops to near the warm level; if the document had to be evicted to (and
restored from) a host/remote tier, TTFT and ITL show the restore cost. Each
visit carries a unique short suffix so a revisit is never a whole-prompt
duplicate (a real revisit asks a new question about the same document).

Items are tagged for the summary:

* ``meta.phase``  — ``"capacity-warm"`` (first pass) / ``"capacity-revisit"``
* ``meta.visit``  — ``"warm"`` / ``"revisit"``
* ``meta.pass_index``, ``meta.doc_id``, ``meta.doc_tokens``
* ``meta.prompt_tokens_hint``

so ``group_by: [phase]`` (or ``[visit]``) splits TTFT/throughput/ITL per
pass, and :meth:`warm_workload` / :meth:`revisit_workload` give one workload
per stage for a phased recipe with independent load models.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from typing import Any, Optional

from benchmaker.workloads import _tokens
from benchmaker.workloads.datasets import IterableWorkload, Workload

WARM_PHASE = "capacity-warm"
REVISIT_PHASE = "capacity-revisit"


@dataclass(frozen=True)
class _Doc:
    doc_id: int
    text: str
    doc_tokens: int


class CapacityRevisitWorkload(Workload):
    """See the module docstring.

    Args:
        documents: list of document strings.
        path: JSONL file of documents (``text_field``, default ``"text"``).
        n_docs / doc_tokens: synthesize ``n_docs`` distinct documents of
            ``doc_tokens`` tokens each (token-exact with a tokenizer).
        tokenizer: HF tokenizer id (exact sizes) or ``None`` for char mode
            (``chars_per_token``).
        revisits: number of revisit passes after the warm pass.
        revisit_order: ``"sequential"`` replays the documents in order;
            ``"random"`` draws every revisit visit in a seeded uniform-random
            order (so an eviction policy can't win on recency alone).
        seed: seed for ``revisit_order="random"``.
        unique_suffix: append a visit-unique ``suffix_tokens``-token tail.
        max_tokens: generation budget per request.
        loop: restart (warm pass again) when exhausted.
    """

    def __init__(self, documents: Optional[list[str]] = None, *,
                 path: Optional[str] = None, text_field: str = "text",
                 n_docs: Optional[int] = None, doc_tokens: Optional[int] = None,
                 tokenizer: Optional[str] = None, chars_per_token: float = 4.0,
                 revisits: int = 1, revisit_order: str = "sequential", seed: int = 0,
                 unique_suffix: bool = True, suffix_tokens: int = 16,
                 max_tokens: int = 128, loop: bool = False, stage: str = "all",
                 name: str = "capacity-revisit"):
        if stage not in ("all", "warm", "revisit"):
            raise ValueError(f"stage must be all, warm or revisit, got {stage!r}")
        self.name = name
        self.stage = stage
        self._sizer = _tokens.build_sizer(tokenizer, chars_per_token)
        self._revisits = max(int(revisits), 0)
        self._unique_suffix = bool(unique_suffix)
        self._suffix_tokens = max(int(suffix_tokens), 1)
        self._max_tokens = int(max_tokens)
        self._loop = loop
        if revisit_order not in ("sequential", "random", "shuffle"):
            raise ValueError(f"revisit_order must be sequential or random, got {revisit_order!r}")
        self._docs = self._load_documents(documents, path, text_field, n_docs, doc_tokens)
        self._reuse_keys = {
            doc.doc_id: "doc:" + hashlib.sha256(doc.text.encode("utf-8")).hexdigest()
            for doc in self._docs
        }
        self._items = self._build_items(revisit_order, seed)
        if stage == "warm":
            self._items = [it for it in self._items if it["meta"]["visit"] == "warm"]
        elif stage == "revisit":
            self._items = [it for it in self._items if it["meta"]["visit"] == "revisit"]
        self._i = 0

    # -- construction -----------------------------------------------------
    def _load_documents(self, documents, path, text_field, n_docs, doc_tokens) -> list[_Doc]:
        if documents:
            return [_Doc(i, d, self._count(d)) for i, d in enumerate(documents)]
        if path:
            docs: list[_Doc] = []
            with open(path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    text = obj[text_field] if isinstance(obj, dict) else str(obj)
                    docs.append(_Doc(len(docs), text, self._count(text)))
            if not docs:
                raise ValueError(f"{path}: no documents")
            return docs
        if n_docs and doc_tokens:
            return [_Doc(i, self._sizer.block(("capacity-doc", i), int(doc_tokens)), int(doc_tokens))
                    for i in range(int(n_docs))]
        raise ValueError("capacity workload needs documents, path, or n_docs+doc_tokens")

    def _count(self, text: str) -> int:
        if getattr(self._sizer, "exact", False):
            return len(self._sizer._encode(text))  # noqa: SLF001
        return max(1, round(len(text) / self._sizer._cpt))  # noqa: SLF001

    def _make_item(self, doc: _Doc, pass_index: int) -> dict:
        warm = pass_index == 0
        suffix = ""
        if self._unique_suffix:
            suffix = " " + self._sizer.block(("capacity-suffix", doc.doc_id, pass_index), self._suffix_tokens)
        return {
            "messages": [{"role": "user", "content": doc.text + suffix}],
            "max_tokens": self._max_tokens,
            "meta": {
                "phase": WARM_PHASE if warm else REVISIT_PHASE,
                "visit": "warm" if warm else "revisit",
                "pass_index": pass_index,
                "doc_id": doc.doc_id,
                "reuse_key": self._reuse_keys[doc.doc_id],
                "doc_tokens": doc.doc_tokens,
                "prompt_tokens_hint": doc.doc_tokens + (self._suffix_tokens if self._unique_suffix else 0),
            },
        }

    def _build_items(self, revisit_order: str, seed: int) -> list[dict]:
        # Warm pass: every document once, in id order (all cold misses).
        items = [self._make_item(doc, 0) for doc in self._docs]
        revisits = [self._make_item(doc, k) for k in range(1, 1 + self._revisits) for doc in self._docs]
        if revisit_order in ("random", "shuffle"):
            random.Random(seed).shuffle(revisits)
        return items + revisits

    # -- Workload API -----------------------------------------------------
    async def next_item(self) -> Any:
        if self._i >= len(self._items):
            if not self._loop:
                raise StopAsyncIteration
            self._i = 0
        item = self._items[self._i]
        self._i += 1
        return item

    def item_count(self) -> Optional[int]:
        return None if self._loop else len(self._items)

    def content_hash(self) -> Optional[str]:
        return "sha256:" + hashlib.sha256(json.dumps(
            self._items, sort_keys=True, default=str).encode("utf-8")).hexdigest()

    # -- stage views ------------------------------------------------------
    @property
    def n_docs(self) -> int:
        return len(self._docs)

    def warm_items(self) -> list[dict]:
        return [it for it in self._items if it["meta"]["visit"] == "warm"]

    def revisit_items(self) -> list[dict]:
        return [it for it in self._items if it["meta"]["visit"] == "revisit"]

    def warm_workload(self) -> IterableWorkload:
        return IterableWorkload(self.warm_items(), name=f"{self.name}-warm")

    def revisit_workload(self) -> IterableWorkload:
        return IterableWorkload(self.revisit_items(), name=f"{self.name}-revisit")
