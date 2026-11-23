"""Token-sized synthetic text, shared by the trace-replay workloads.

Two sizers turn "N tokens of deterministic filler for seed S" into text:

* :class:`CharSizer`  — ``N * chars_per_token`` characters of filler. No
  dependency; the served tokenizer decides the real token count (≈ N).
* :class:`TokenizerSizer` — exact with a HuggingFace tokenizer. ``fill`` repeats
  a single-token unit (byte-prefix exact, used for nested prefixes);
  ``block`` produces text that is *distinct per seed* while still encoding to
  exactly N tokens (used for cache-block replay where different block ids must
  not collide).
* :class:`TokenIdSampler` — no text at all: a seeded block of exactly N token
  *ids* drawn from a whitelist, sent as ``prompt: [ids]`` to
  ``/v1/completions`` (issue #11). The server's tokenizer never runs, so
  nothing merges across block boundaries and no chat template shifts the
  alignment: equal seeds give byte-identical, page-aligned prefixes.

All text is a pure function of ``(seed, n)`` via a *stable* hash — never
Python's salted ``hash()`` — so two processes replaying the same trace send
byte-identical prompts (issue #6; also what makes a prompt-set hash in the
run manifest meaningful, issue #4).
"""

from __future__ import annotations

import hashlib
import io
import random
from collections import OrderedDict
from typing import Any, Sequence

__all__ = [
    "CharSizer", "TokenizerSizer", "TokenIdSampler", "build_sizer", "load_tokenizer",
    "filler_text", "stable_hash", "FILLER_WORDS",
]

# Filler vocabulary (shared with TraceLab replay). Joined with spaces it looks
# vaguely like prose, tokenizes sanely, and — being a pure function of the
# seed — keeps repeated blocks byte-identical across runs and processes.
FILLER_WORDS = (
    "def", "return", "self", "data", "value", "context", "model", "request",
    "response", "config", "cache", "input", "output", "token", "prompt",
    "message", "session", "round", "tool", "result", "agent", "stream",
    "client", "server", "param", "state", "event", "trace", "parse", "load",
)

SINGLE_TOKEN_CANDIDATES = ("x", " x", "a", " a", "0", ".", "word")


def stable_hash(seed: Any) -> int:
    """Process-stable 63-bit int from any seed (issue #4/#6 reproducibility)."""
    h = hashlib.sha256(repr(seed).encode("utf-8", errors="replace")).digest()
    return int.from_bytes(h[:8], "big") & ((1 << 63) - 1)


def load_tokenizer(tokenizer: Any):
    """A HF tokenizer id -> the tokenizer; anything else is returned as-is, so
    the Python API (and tests) can pass an already-loaded tokenizer."""
    if isinstance(tokenizer, str):
        try:
            from transformers import AutoTokenizer  # type: ignore
        except ImportError as e:  # pragma: no cover - exercised via message
            raise ImportError(
 "tokenizer=... needs the `transformers` package. "
                "Install with `pip install -e .[tokenizer]`."
            ) from e
        return AutoTokenizer.from_pretrained(tokenizer)
    return tokenizer


def build_sizer(tokenizer: Any, chars_per_token: float):
    if tokenizer:
        return TokenizerSizer(tokenizer)
    return CharSizer(chars_per_token)


class CharSizer:
    """Char-based filler: deterministic word stream sliced to a char target."""

    exact = False

    def __init__(self, chars_per_token: float):
        if chars_per_token <= 0:
            raise ValueError("chars_per_token must be > 0")
        self._cpt = float(chars_per_token)

    def fill(self, seed: Any, target_tokens: int) -> str:
        target_tokens = max(1, int(target_tokens))
        target_chars = max(1, round(target_tokens * self._cpt))
        return filler_text(seed, target_chars)

    def block(self, seed: Any, n_tokens: int) -> str:
        # Distinct per seed by construction: the seed drives the word stream.
        return self.fill(seed, n_tokens)


class TokenizerSizer:
    """Exact filler with a HuggingFace tokenizer."""

    exact = True

    def __init__(self, tokenizer: Any):
        self._tok = load_tokenizer(tokenizer)
        self._unit = self._single_token_unit()
        self._blocks: dict[tuple[int, int], str] = {}
        self._word_cost: dict[str, int] = {}

    def _encode(self, text: str) -> list[int]:
        return self._tok.encode(text, add_special_tokens=False)

    def _single_token_unit(self) -> str:
        for cand in SINGLE_TOKEN_CANDIDATES:
            if len(self._encode(cand)) == 1:
                return cand
        return "x"

    def fill(self, seed: Any, target_tokens: int) -> str:
        """``unit * n``: exactly n tokens, and every prefix is byte-exact."""
        n = max(1, int(target_tokens))
        return self._unit * n

    def _cost(self, word: str) -> int:
        c = self._word_cost.get(word)
        if c is None:
            c = len(self._encode(" " + word))  # cost of the word after a space
            self._word_cost[word] = c
        return c

    def block(self, seed: Any, n_tokens: int) -> str:
        """Distinct text per seed encoding to exactly ``n_tokens`` tokens.

        Fast path: take words from the seed's deterministic stream while their
        (memoized) token cost fits, fill the remainder with the single-token
        unit, verify with one encode. Slow path (rare: cross-word merges):
        trim/extend one word at a time until the tokenizer agrees. Results are
        memoized per (seed, n) — cache-block replay sends the same block many
        times (#6).
        """
        n = max(1, int(n_tokens))
        key = (stable_hash(seed), n)
        cached = self._blocks.get(key)
        if cached is not None:
            return cached
        unit_word = self._unit.strip() or self._unit
        words = filler_text(seed, max(16, n * 6)).split(" ") if n > 1 else []
        parts: list[str] = []
        remaining = n
        for w in words:
            c = self._cost(w)
            if c > remaining:
                continue
            parts.append(w)
            remaining -= c
            if remaining == 0:
                break
        parts.extend([unit_word] * remaining)
        text = " ".join(parts)
        ids = self._encode(text)
        # Slow path: adjust word by word until exact.
        guard = 0
        while len(ids) != n and guard < 4 * n + 8:
            guard += 1
            if len(ids) > n:
                if " " in text:
                    text = text[: text.rfind(" ")]
                else:
                    text = ""
            else:
                text = (text + " " if text else "") + unit_word
            ids = self._encode(text)
        self._blocks[key] = text
        return text


class TokenIdSampler:
    """Seeded blocks of exactly ``n`` token ids from a whitelist (issue #11).

    Used by ``send: token_ids`` trace replay: every hash block is a list of
    ints, so the server's tokenizer never touches it — no cross-block merges,
    no chat-template prefix — and ``block(seed, r)`` is exactly the first
    ``r`` ids of ``block(seed, 512)``, which keeps a truncated last block a
    true prefix of the full one.

    The whitelist excludes ids a server treats specially (BOS/EOS/PAD, chat
    control tokens such as ``<|im_start|>``, reserved ranges): those can end
    generation early or trip a parser even when they appear in a prompt.
    With a tokenizer they are found from ``all_special_ids`` plus every
    *added* token; without one (``from_vocab``) the caller states the reserved
    ids explicitly.

    Blocks are memoized per ``(seed, n)`` in a bounded LRU: rendering is
    cheap, but not free at thousands of requests per second, and a full
    Mooncake trace has far too many distinct blocks to keep them all.
    """

    def __init__(self, whitelist: Sequence[int], *, source: str = "", cache_size: int = 4096):
        ids = sorted({int(i) for i in whitelist})
        if not ids:
            raise ValueError("token-id whitelist is empty")
        self._ids = ids
        self.source = source
        self._cache: OrderedDict[tuple[int, int], list[int]] = OrderedDict()
        self._cache_size = max(1, int(cache_size))

    @classmethod
    def from_tokenizer(cls, tok: Any, **kw: Any) -> "TokenIdSampler":
        tok = load_tokenizer(tok)
        n = len(tok)
        special = set(int(i) for i in getattr(tok, "all_special_ids", ()) or ())
        special |= {int(i) for i in (getattr(tok, "added_tokens_encoder", None) or {}).values()}
        return cls([i for i in range(n) if i not in special],
                   source=f"tokenizer vocab={n} excluded={len(special)}", **kw)

    @classmethod
    def from_vocab(cls, vocab_size: int, exclude_ids: Sequence[int] = (),
                   exclude_last: int = 0, **kw: Any) -> "TokenIdSampler":
        """``[0, vocab_size)`` minus ``exclude_ids`` minus the last ``exclude_last``
        ids — the escape hatch when the tokenizer is not available locally."""
        vocab_size = int(vocab_size)
        if vocab_size <= 0:
            raise ValueError("vocab_size must be > 0")
        excluded = {int(i) for i in exclude_ids} | set(range(max(0, vocab_size - int(exclude_last)), vocab_size))
        return cls([i for i in range(vocab_size) if i not in excluded],
                   source=f"vocab={vocab_size} excluded={len(excluded)}", **kw)

    @property
    def size(self) -> int:
        """Number of whitelisted ids."""
        return len(self._ids)

    def block(self, seed: Any, n_tokens: int) -> list[int]:
        n = max(1, int(n_tokens))
        key = (stable_hash(seed), n)
        hit = self._cache.get(key)
        if hit is not None:
            self._cache.move_to_end(key)
            return list(hit)
        rng = random.Random(key[0])
        ids = self._ids
        m = len(ids)
        out = [ids[rng.randrange(m)] for _ in range(n)]
        self._cache[key] = out
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return list(out)


def filler_text(seed: Any, char_count: int) -> str:
    """Deterministic filler of ~char_count chars, stable per seed (process-stable)."""
    rng = random.Random(stable_hash(seed))
    words = FILLER_WORDS
    nwords = len(words)
    out = io.StringIO()
    written = 0
    i = 0
    while written < char_count:
        w = words[(i + rng.randrange(0, nwords)) % nwords]
        if i:
            out.write(" ")
            written += 1
        out.write(w)
        written += len(w)
        i += 1
    return out.getvalue()[:char_count]
