# Decode stalls, cache classes, and TTFT parity

Issues #1 and #2: a mean ITL hides the stalls that make decode *feel* slow,
an aggregate TTFT hides which cache class paid for it, and a TTFT compared
across servers is meaningless if the two servers stamp "first token" on
different events. This page covers the three mechanisms benchmaker adds for
that:

1. **Pooled ITL and stall accounting** (`itl_ms_pooled`, `stalls`) — issue #1.
2. **Cache classes and `group_by`** — per-class TTFT and the mix/latency
   decomposition of a TTFT difference — issue #2.
3. **TTFT parity** — what each server's TTFT was measured on — issue #4.

All of it is derived from the same `samples.jsonl` and `summary.json` that
every run bundle already writes; nothing here needs a special server.

## 1. Pooled ITL and stalls (#1)

### The problem

`itl_ms_mean` and `itl_ms_p99` in `workload_metrics` are computed **per
request** and then averaged. A single 300 ms decode stall inside a 400-token
answer moves that request's mean ITL by less than 1 ms, and its p99 is
unaffected (the stall is one observation out of 400). The stall exists only in
the raw per-token gaps, which the summary used to throw away.

### What is recorded

For every streamed request the workload-type keeps the raw inter-token gap
series on `Sample.series["itl_ms"]` (one value per decode step). The
aggregator then **pools** the observations of every successful request:

```python
"workload_metrics": {
    "itl_ms_mean":  {...},          # per-request mean, then summarized (unchanged)
    "itl_ms_p99":   {...},          # per-request p99, then summarized (unchanged)

    "itl_ms_pooled": {              # NEW: every decode step of every request, one pool
        "count": 51234,             # pooled observations
        "mean": 11.8, "min": 0.9, "max": 612.0,
        "p50": 11.2, "p90": 12.1, "p99": 24.6, "p995": 41.0, "p999": 180.2
    },

    "stalls": {                     # NEW: gaps above the stall threshold
        "threshold_ms": 12.0,       # the threshold used
        "requests": 1200,           # streamed requests that contributed gaps
        "count": 1840,              # stalls in total
        "per_request": {"mean": 1.53, "p50": 1.0, "p90": 3.0, "p99": 12.0, "max": 44},  # stalls per request
        "time_per_request_ms": {"mean": 40.1, "p50": 18.0, "p99": 310.0},  # stalled time per request
        "stall_ms": {"p50": 18.4, "p90": 31.2, "p99": 88.0, "max": 612.0},  # length of one stall
        "total_time_ms": 48110.0,   # Σ stalls of all requests: time spent stalled
        "excess_ms": 24320.0,       # Σ (gap − threshold): time lost beyond a normal step
        "histogram": [              # log2 bins over *all* gaps, for plotting
            {"lo_ms": 8.0, "hi_ms": 16.0, "count": 49800, "share": 0.97},
            ...
        ]
    }
}
```

### Per-token, not per network read (issue #17)

An SSE event is not a token. Servers batch tokens per event under load
(vLLM/SGLang stream one chunk per scheduler step, often several tokens),
and the client's network reads coalesce events that were produced apart.
Measuring "ITL" as the gap between events therefore reported (a) how the
server *batched* its output and (b) when the kernel woke the reader — an
arm with a busier GPU looked *faster* because it emitted fewer, fatter
events, with a 6 ms gap charged whole to the first token of a read and
0 ms to the rest.

`itl_ms` is now **per token**:

- how many tokens each event carried comes from the server when it says so
  — the final `usage.completion_tokens` (OpenAI-compatible types) or the
  cumulative `meta_info.completion_tokens` on every event (sglang, exact);
- with only the final count, the surplus over the number of events is
  spread evenly (`itl_mode: per_token_estimated`); with more events than
  tokens (a usage that excludes reasoning tokens, say) each event stays one
  token and `itl_usage_mismatch` is set on the sample;
- consecutive events closer than 0.2 ms came in one read: the read's gap
  is divided by the read's token count instead of landing whole on its
  first token.

Each sample records `meta.itl_mode` (`per_token` / `per_token_estimated` /
`per_event`), `itl_coalesced_events`, `itl_tokens_per_event_mean` and the
old per-event gaps as `itl_chunk_ms_*`; the summary adds an `itl_modes`
block (`modes: {per_token: 1180, per_event: 4}`, mean coalesced events and
tokens per event) and the terminal render prints it under the pooled ITL.
If `tokens_per_event_mean` drifts between arms, the server's event batching
changed — not its decode speed.

Three things to read off this block:

- **`itl_ms_pooled.p99.5` vs `itl_ms_p99`**: if the pooled tail is much worse
  than the per-request tail, the stalls are *spread* over many requests (a
  scheduling/batching problem), not concentrated in a few (a few pathological
  prompts).
- **`stalls.per_request` vs `stalls.time_per_request_ms`** separate *who
  stalls often* from *who stalls long*: many short stalls spread over every
  request is a scheduler problem; a few very long ones in a handful of
  requests is a prefill/eviction problem.
- **`stalls.excess_ms`** is the stall cost you can compare across arms:
  "arm B spent 24.3 s beyond a normal decode step, arm A 3.1 s".

### Knobs (YAML)

```yaml
stall_ms: 12          # gap above this many ms is a stall (default 12; "none" disables stalls)
dump_series: true     # write every sample's raw series into samples.jsonl (large!)
```

With `dump_series: true` each `samples.jsonl` record carries
`"series": {"itl_ms": [...], "ttft_ms": [...]}`, which is what you want when
plotting a timeline or recomputing stalls offline at a different threshold.

`stalls_per_req`, `stall_excess_ms`, `itl_p99.5` and `itl_p99.9` are metric
aliases understood by `benchmaker collect`, `benchmaker arms` and
`benchmaker compare` (see [Multi-arm comparisons](arms.md)).

## 2. Cache classes and `group_by` (#2)

### The problem

"Mean TTFT improved by 40 ms" is ambiguous: did the server get faster, or did
it simply get a better hit mix? A run where 70 % of prompts are full device
hits has a great mean TTFT and tells you nothing about the restore path.

### Tagging

`OpenAIChatWorkloadType` classifies every streamed response from the `usage`
block the server returns (`prompt_tokens`, `cached_tokens`) into
`sample.meta["cache_class"]`:

| Class         | Meaning                                                        |
| ------------- | -------------------------------------------------------------- |
| `device_hit`  | the whole cached prefix was served from the device (KV) cache  |
| `restore`     | a prefix was cached but had to be restored from a host/remote tier |
| `partial`     | only part of the prefix was cached                              |
| `miss`        | nothing cached (cold prefill)                                   |
| `unknown`     | a full hit whose `cache_source` hint names a tier this module does not know (issue #17) |

The tier comes from the server's own words — `cache_source` / `cache_tier` /
`X-Cache-Source` in the usage block, `prompt_tokens_details` or headers —
read as **whole words** (`host_dram`, `HostDram`, `HOST-DRAM` all split into
`host` + `dram`; `hint_tier()`). `dram` is *host* memory, so `host_dram` is a
restore tier and never a device hit; only `device`, `gpu`, `hbm`, `local`,
`vram` are device tiers; restore words (`restore`, `remote`, `host`, `cpu`,
`dram`, `mooncake`, `lmcache`, `hicache`, `ssd`, `mixed`, …) win over device
words. A hint with no word from either list (`l2`, `kv_cache`, `localhost`)
is `unknown` — recorded as such instead of a guessed `device_hit` — and
`none` means "no tier information", leaving the ratio to decide.

Which combination of `cached_tokens`/`prompt_tokens` means `device_hit` vs
`restore` is server-specific. The default classifier uses a token-rate
threshold (`CacheClassInfo` → `classify_cache`); extend the vocabularies
(`RESTORE_WORDS` / `DEVICE_WORDS`) or pass your own for your server:

```yaml
workload_type:
  type: openai
  cache_classifier: my_pkg.classify:rill_cache_class   # fn(CacheClassInfo) -> str | None
```

A classifier returning `None` leaves the sample untagged.

### Breaking the summary down

```yaml
group_by: [cache_class]
```

adds, to the top level **and** to every phase and lane block of
`summary.json`:

```python
"groups": {
    "cache_class": {
        "device_hit": {"share": 0.71, "wall_time_s": 60.0, "workload_metrics": {"ttft_s": {...}, ...},
                        "stalls": {...}, "latency_s": {...}, "throughput_rps": ...},
        "restore":   {"share": 0.18, ...},
        "miss":      {"share": 0.11, ...}
    }
}
```

`group_by` accepts any `Sample.meta` key (`visit`, `phase`, `lane`, `doc_id`,
trace fields, …), so the same mechanism serves the capacity-revisit and trace
replay workloads.

### Mix vs latency decomposition

Given two summaries with the same `group_by` key, benchmaker splits the
difference in mean TTFT exactly into a *mix* term (the hit distribution
changed) and a *latency* term (each class is served faster or slower), using
the symmetric Shapley/midpoint decomposition:

```
Δ = Σ_c (s_b − s_a)·(μ_a + μ_b)/2   +   Σ_c (s_a + s_b)/2·(μ_b − μ_a)
    |------- mix: hit-rate change --|   |------ latency: per-class -----|
```

It is order-independent (swapping A and B only flips signs) and the two terms
sum to the raw difference. `benchmaker arms` and `benchmaker compare` print it
for every non-reference arm:

```
TTFT decomposition (ref = vllm) — rill
  class       share vllm→rill   mean TTFT vllm→rill (ms)   mix (ms)   latency (ms)
  device_hit      0.42 → 0.71          38 → 36           −10.9        −0.6
  restore         0.20 → 0.18         210 → 140            +0.5        −1.4
  miss            0.38 → 0.11         150 → 152            +8.8        −0.0
  total                              104 →  84            −1.6       −2.0
```

Interpretation: of the 20 ms improvement, 1.6 ms is a better hit *mix* and
2.0 ms is genuinely faster *per class*; the rest of the difference is
interaction captured inside the two terms (the decomposition is exact, so
`mix + latency == Δ`).

From Python:

```python
from benchmaker.core.compare import decompose_ttft
dec = decompose_ttft(summary_a, summary_b, key="cache_class")
```

Void reps (failed readiness/stop checks, void rules) are left out of the
decomposition and of the TTFT parity medians exactly as the table leaves
them out (issue #17) — a rep voided for an unready server must not weigh on
"how much of the TTFT delta is latency". (Before #17 the decomposition was
silently skipped whenever an arm had two or more reps — the synthetic median
summary it built could not be read back; fixed.)

## 3. TTFT parity (#4)

TTFT is stamped on the arrival of the first streamed token *of the kind the
workload is configured to look for* (`ttft_token: any | content`, see
[Workloads](workloads.md#openaichatworkloadtype)). Two servers can therefore
report TTFTs that are not comparable:

- one arm has a reasoning parser and emits `reasoning_content` deltas; with
  `ttft_token: content` its TTFT skips the whole reasoning phase while the
  other arm's does not;
- the first token the server emits may be at a different chunk *index* (an
  empty role delta, a leading newline, a parsed `<think>` marker…).

Every sample therefore records the chunk index its TTFT was stamped on
(`extra.ttft_token_index`; summarized as
`workload_metrics.ttft_token_index`), and the bundle keeps the `ttft_token`
setting in `meta.source_config`. `benchmaker arms` / `benchmaker compare`
collect these per arm (`ttft_parity` in `comparison.json`) and **warn** (not
fail — it is for you to judge) when:

- `kind` differs between arms ("arms stamp TTFT on different token kinds"), or
- the median first-token index differs by ≥ 1 ("arms' first-token index
  differs").

The raw data is on every sample (`extra.ttft_token_index`,
`extra.content_ttft_s`), so a run with `ttft_token: any` still lets you
recompute a content-only TTFT offline.

## Reading order within a comparison

For a multi-arm comparison, the recommended order is:

1. Fix TTFT parity first (same `ttft_token`, same parser situation) —
   otherwise every TTFT number below is off by a reasoning phase.
2. Look at `groups.cache_class` shares: are the arms even seeing the same
   hit mix? If not, the decomposition tells you how much of Δ is mix.
3. Compare `itl_ms_pooled.p99.5` and `stalls.excess_ms` rather than
   `itl_ms_mean`.
4. Only then read the headline `ttft_p50/p99` and `throughput_rps`.

See [Multi-arm comparisons](arms.md) for the commands that put all of this
in one table.
