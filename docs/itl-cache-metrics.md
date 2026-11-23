# Decode stalls and pooled ITL

Issue #1: a mean ITL hides the stalls that make decode *feel* slow.
This page covers what benchmaker records about them:

1. **Pooled ITL and stall accounting** (`itl_ms_pooled`, `stalls`).

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
series on `Sample.series["itl_ms"]` (one value per decode step, measured on
SSE chunk arrival). The aggregator then **pools** the observations of every
successful request:

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
