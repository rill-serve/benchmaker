# Phased benchmarks, warm-up, and the capacity-revisit workload

Issue #5. Prefix-cache behaviour has to be measured over *time*: a cache that
is warm is a different machine from one that is cold, and a cache whose
working set exceeds the device is a third. A single 60-second run with one
load model averages over all three. Benchmaker therefore adds:

1. **Phases** — timed stages run back-to-back, each with its own lanes, load
   models and optional model routing; every sample is tagged with its phase
   and summarized per phase.
2. **Warm-up** — `warmup_requests` / `warmup_s` are recorded but excluded
   from every summary.
3. **The capacity-revisit workload** — a warm pass over N long documents,
   then K revisit passes, tagged per visit, so TTFT/throughput/ITL can be
   split into *cold prefill*, *device hit* and *restore*.

## 1. Phases (`phases:`)

```yaml
workload_type: {type: openai, url: http://localhost:8000/v1/chat/completions, model: qwen2.5-32b, stream: true}

phases:
  - name: warm                      # 1. cold prefill: every document is a miss
    duration: 120s
    lanes:
      - name: warm
        workload: {type: capacity, n_docs: 64, doc_tokens: 8192, stage: warm, max_tokens: 64}
        rate: 4
  - name: revisit                   # 2. same documents again: hit or restore?
    duration: 120s
    drain: true                     # let in-flight requests finish before the next phase
    lanes:
      - name: revisit
        workload: {type: capacity, n_docs: 64, doc_tokens: 8192, stage: revisit,
                   revisit_order: random, seed: 7, max_tokens: 64}
        rate: 4
      - name: background            # 3. background traffic in the same phase
        workload: {type: jsonl, path: .local/sharegpt_v3.jsonl, loop: true}
        rate: poisson:6
        model: qwen2.5-7b           # route this lane to another model on the same endpoint
        meta: {tier: background}    # extra tags stamped on every sample of the lane

group_by: [phase, visit, cache_class]
warmup_requests: 16
stall_ms: 12
```

Rules:

- `phases` is an alternative to `load`/`workload` and to `mix`; combining
  them is a config error. Each phase needs a `duration` (a top-level
  `duration:` is the fallback).
- Each phase owns its **lanes** (same shape as [`mix:` lanes](deeprag-mix.md)):
  a `workload`, a `rate`/`load` spec, optional `duration`/`max_requests`,
  optional `model` (overrides the model in the request for that lane) and
  optional `meta` (arbitrary tags merged into every sample's `meta`).
- Every lane's load model starts at **its phase's** t=0, so `ramp:0..32:60s`
  inside a 60 s phase ramps over exactly that phase.
- When the phase's duration ends, the runner stops issuing tickets. With
  `drain: true` it then waits for the phase's in-flight requests to finish
  before starting the next phase, so a slow tail does not compete with the
  next phase's traffic (the window then covers the tail, logged as
  "drained"). Without `drain`, the next phase starts immediately while the
  tail completes in the background; those samples stay tagged with the phase
  that issued them, so the per-phase latency numbers are unaffected — only
  the next phase's load is slightly higher for a moment.
- A lane whose workload is exhausted (a finite dataset without `loop`) ends
  its phase early; the phase window closes when every lane is done. The
  summary records both the budget (`wall_time_s`) and the producing time
  (`active_s`).
- `meta.phase` is set on every sample (from the phase name); `meta.lane`
  from the lane name. Samples carry both into `samples.jsonl`, so any
  offline analysis can cut by phase.

### Summary

`summary.json` gains a `phases` block with, per phase, the same shape as the
top-level summary — including `lanes`, `groups` (from `group_by`) and the
pooled ITL/stall blocks:

```python
{
  "wall_time_s": 240.3, "total_requests": ..., "workload_metrics": {...}, "groups": {...},
  "warmup_excluded": 16,
  "phases": {
    "warm":    {"start_s": 0.0, "end_s": 120.0, "wall_time_s": 120.0, "active_s": 119.6,
                "total_requests": 480, "throughput_rps": 4.0, "workload_metrics": {"ttft_s": {...}},
                "stalls": {...}, "itl_ms_pooled": {...},
                "lanes": {"warm": {...}}, "groups": {"visit": {"warm": {...}}, "cache_class": {"miss": {...}}}},
    "revisit": {"start_s": 120.4, "end_s": 241.0, ...}
  }
}
```

The **phase comparison is the result**: TTFT p50 in `warm` is the cold
prefill latency; in `revisit` it is the hit (or restore) latency; the
`groups.cache_class` shares inside `revisit` tell you how much of the working
set the device cache retained, and the `restore` row's TTFT/ITL is the cost
of the tier that served the rest.

## 2. Warm-up

```yaml
warmup_requests: 16      # the first 16 tickets fired are warm-up
warmup_s: 5s             # or: everything fired in the first 5 seconds
```

Warm-up samples are **recorded** in `samples.jsonl` with `meta.warmup: true`
and **excluded** from every summary block — the aggregate, `lanes`, `phases`
and `groups`. `summary.warmup_excluded` says how many. Warm-up tickets are
issued by the same load model at the same rate, so the measured window starts
with the server already in its steady state. The equivalent per-arm setting
for multi-arm runs is `arms.warmup_reps` (whole reps excluded, see
[Multi-arm comparisons](arms.md)).

## 3. The capacity-revisit workload

```yaml
workload:
  type: capacity               # capacity | capacity-revisit
  # either real documents …
  path: .local/long_docs.jsonl # one JSON object per line, text in `text_field`
  text_field: text
  # … or synthesized ones (token-exact with a tokenizer, approximate without):
  # n_docs: 64
  # doc_tokens: 8192
  tokenizer: Qwen/Qwen2.5-7B    # optional: exact token counts (needs [hf]); default char mode (chars_per_token: 4)
  revisits: 1                   # revisit passes after the warm pass
  revisit_order: random         # sequential | random — random defeats a pure-recency eviction policy
  seed: 0
  unique_suffix: true           # append a visit-unique tail (so a revisit is a new question about the same doc)
  suffix_tokens: 16
  max_tokens: 128
  stage: all                    # all | warm | revisit — one stage only (for phased recipes)
  loop: false
load: 4
```

The item sequence is: every document once (the **warm** pass, in id order —
every request is a cold miss), then `revisits` more passes over the same
documents (in order, or in a seeded random order). Each item is a chat
message `"<document text> <unique suffix>"`; the document part is
byte-identical across passes so a prefix cache *can* hit, the suffix is new
per (document, pass) so the prompt as a whole never repeats.

Every item carries `meta`:

| Key                  | Value                                                     |
| -------------------- | --------------------------------------------------------- |
| `phase`              | `capacity-warm` / `capacity-revisit`                      |
| `visit`              | `warm` / `revisit`                                        |
| `pass_index`         | 0 for the warm pass, 1..K for revisits                    |
| `doc_id`, `doc_tokens` | which document, how long                                |
| `prompt_tokens_hint` | `doc_tokens + suffix_tokens`                              |

These flow into `sample.meta` (so `group_by: [visit]` works) and
`prompt_tokens_hint` into `sample.extra`. The workload's `content_hash()`
covers the exact item list, so two bundles can be checked for having used
the same documents and visit order (`manifest.prompt_set.hash`).

What to look at, with `group_by: [visit, cache_class]`:

| Observation                                   | Meaning                                                     |
| --------------------------------------------- | ----------------------------------------------------------- |
| `revisit` TTFT ≈ `warm` TTFT, `cache_class = miss` | nothing was retained: working set > cache, no lower tier |
| `revisit` TTFT ≈ a few ms, `cache_class = device_hit` | whole working set retained on device        |
| `revisit` TTFT in between, `cache_class = restore` | retained in a host/remote tier; the gap is the restore cost |
| `revisit` **ITL** worse than `warm`            | the restore path competes with decode (a stall problem)    |
| `revisit` throughput per lane lower            | restore bandwidth bound                                     |

The `cache_class` values come from the server's `usage` block — see
[Decode stalls, cache classes, TTFT parity](itl-cache-metrics.md#2-cache-classes-and-group_by-2)
for the default classifier and how to plug in a server-specific one.

### Recipes

Single phase, both stages, split after the fact:

```yaml
workload: {type: capacity, n_docs: 64, doc_tokens: 8192, revisits: 2, revisit_order: random}
load: 4
group_by: [visit, cache_class]
```

Two phases with independent load models (the `stage:` views share one
document set because `n_docs`/`doc_tokens`/`tokenizer` are the same, so the
content is identical — the synthesizer is seeded by `doc_id`):

```yaml
phases:
  - {name: warm,    duration: 120s, lanes: [{name: warm,    workload: {type: capacity, n_docs: 64, doc_tokens: 8192, stage: warm},    rate: 4}]}
  - {name: revisit, duration: 120s, lanes: [{name: revisit, workload: {type: capacity, n_docs: 64, doc_tokens: 8192, stage: revisit}, rate: 4}], drain: true}
```

Python:

```python
from benchmaker.workloads.capacity import CapacityRevisitWorkload

wl = CapacityRevisitWorkload(n_docs=64, doc_tokens=8192, revisits=2, revisit_order="random")
warm, revisit = wl.warm_workload(), wl.revisit_workload()   # IterableWorkload views per stage
```

### Sizing the working set

To make the revisit pass actually exercise eviction, pick
`n_docs × doc_tokens × bytes_per_token` larger than the device KV cache
(for a 7B model with GQA, roughly `2 × n_layers × n_kv_heads × head_dim × 2
bytes` per token — e.g. ≈ 0.2 MB/token, so 8 GB holds ≈ 40k tokens: four
8k documents). The `tokenizer:` option makes `doc_tokens` exact; without it
the char-mode sizing is within a few percent for English prose.
