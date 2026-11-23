# Multi-arm comparisons (ABBA, per-rep bundles, same-job check)

Issue #3. A/B-ing two serving stacks by running one, then the other, an hour
apart, is not an experiment: node drift, thermal state, background jobs and
cache warm-up all land on whichever arm happened to run second. A
*multi-arm run* in benchmaker instead:

- runs the **same** benchmark config against several servers ("arms"),
- **several times each** ("reps"), **interleaved** (ABBA / rep-major / random),
- writes **one run bundle per (arm, rep)** with its own manifest,
- builds the comparison table, same-job check, TTFT parity and TTFT
  decomposition **from the bundles**, so it can be re-run and audited later.

```
runs/20260701T120000Z/
    plan.json                       # the (arm, rep) execution order
    arms.log                        # one line per launch / run
    arms/
        rill/
            launch-rep-0.log        # stdout/stderr of the launch hook
            rep-0/                  # a complete run bundle (meta/summary/samples/manifest…)
            rep-1/
        vllm/
            rep-0/
            rep-1/
    comparison.json                 # table + checks, machine-readable
    comparison.md                   # the table, for pasting into a PR
    decomposition-vllm.md           # TTFT mix/latency decomposition vs the reference
```

## Config

```yaml
# arms.yaml
arms:
  reps: 4
  order: abba                 # abba | rep-major | random
  warmup_reps: 1              # the first rep of every arm is excluded from the table
  reference: vllm             # ratios are against this arm (default: the first)
  seed: 0                     # for order: random
  allow_mixed_jobs: false     # fail (exit 2) if reps came from different jobs/hosts
  launch_timeout_s: 600
  metrics: [ttft_p50, ttft_p99, itl_p99.5, stalls_per_req, stall_excess_ms, rps, err_pct]
  arms:
    vllm:
      config:                               # deep-merged over the shared config below
        workload_type: {url: "http://vllm:8000/v1/chat/completions"}
      env: {VLLM_USE_V1: "1"}               # env for the launch hook; recorded in the manifest
      launch: "./scripts/launch-vllm.sh"    # run before the arm's first measured run
      relaunch: once                        # once | every_rep
      labels: {image: "vllm:0.9.1"}
    rill:
      config:
        workload_type: {url: "http://rill:8000/v1/chat/completions"}
      env: {RILL_KV_CACHE_TIER: host}
      launch: "./scripts/launch-rill.sh"
      relaunch: every_rep
      labels: {image: "rill:0.9.1"}

# ---- shared base config: everything below is what a single run would have ----
workload_type: {type: openai, model: qwen2.5-32b, stream: true, max_tokens: 256}
workload: {type: jsonl, path: .local/sharegpt_v3.jsonl, max_items: 400}
load: poisson:8
duration: 60s
group_by: [cache_class]
capability: {url: "http://HOST:PORT/v1/models"}      # see docs/capability-manifest.md
void_rules:
  - name: served-model
    path: models.0
    op: eq
    value: qwen2.5-32b
    at: both
```

Rules:

- `arms.<name>.config` is **deep-merged** over the shared config (dicts merge
  recursively; scalars and lists are replaced). Anything not overridden —
  workload, load, duration, hooks, monitors, `group_by` — is identical for
  every arm by construction.
- `env` is exported to the `launch` hook and recorded in the bundle's
  `manifest.json` under `env`, so a reader can see which knobs differed.
  The hook also receives `BENCHMAKER_ARM` and `BENCHMAKER_REP`.
- `launch` runs through the shell from the working directory; its output goes
  to `arms/<arm>/launch-rep-<k>.log`. A non-zero exit aborts the whole run.
  `relaunch: once` (default) launches each arm once; `every_rep` relaunches
  before every rep (needed when the server's state cannot be reset between
  reps otherwise).
- `warmup_reps` reps per arm are run first (per the order) and **excluded**
  from the table; they are kept on disk and listed as `excluded` in the
  comparison.

## Running

```bash
benchmaker arms arms.yaml --out-dir ./runs --label gpus=8xh100 --notes "kv-tier experiment"

benchmaker arms arms.yaml --out-dir ./runs --dry-run     # print the plan only
benchmaker arms arms.yaml --out-dir ./runs --reps 2 --allow-mixed-jobs
```

`--dry-run` prints the execution plan — e.g. for 2 arms, 4 reps, `abba`:

```
vllm rep-0 (warmup)
rill rep-0 (warmup)
rill rep-1
vllm rep-1
vllm rep-2
rill rep-2
rill rep-3
vllm rep-3
```

The command prints the comparison table at the end and exits with:

| Code | Meaning                                                              |
| ---- | -------------------------------------------------------------------- |
| 0    | done; same-job check passed (or `allow_mixed_jobs`)                  |
| 2    | reps came from different jobs/hosts and `--allow-mixed-jobs` not set |

Exit 2 still leaves every bundle and `comparison.*` on disk — the point is
that the table is not to be trusted, not that the data is discarded.

## Reading the table

```
| arm     | reps | ttft_p50 (ms)        | ttft_p99 (ms)       | itl_p99.5 (ms)    | stalls_per_req     | stall_excess_ms (ms) | rps           | err_pct |
| ------- | ---- | -------------------- | ------------------- | ------------------ | ------------------- | --------------------- | ------------- | ------- |
| **vllm**| 3    | 92.1 [89.0–95.4]     | 310.2 [300.1–322.0]  | 38.2 [37.0–39.9]   | 0.41 [0.38–0.45]    | 24310 [23900–24800]   | 7.98 [7.9–8.0]| 0.0     |
| rill    | 3    | 74.5 [73.9–75.2] (×0.809 ✓) | 288.0 [281–296] (×0.929 ✓) | 31.4 [30.8–32.2] (×0.822 ✓) | 0.12 [0.11–0.13] (×0.293 ✓) | 8120 [7900–8400] (×0.334 ✓) | 7.99 [7.9–8.1] (×1.001 ✗) | 0.0 (×1.0) |
_excluded (void/warm-up): rill: 1, vllm: 1_
```

Each cell is `median [min–max]` over the arm's non-warm-up reps, followed by
the ratio to the reference arm's median and `✓`/`✗` when the metric has a
known better direction (`lower` for latencies, `higher` for throughput). The
spread in brackets is your noise floor: a ratio whose arms' `[min–max]`
overlap is not a result.

Metric names are the aliases in `benchmaker.core.compare.METRIC_ALIASES`
(`ttft_p50`, `ttft_p99`, `lat_p99`, `itl_p99.5`, `stalls_per_req`,
`stall_excess_ms`, `rps`, `goodput_rps`, `err_pct`, `cached_pct`, …) or any
dotted path into `summary.json` (`workload_metrics.ttft_s.p99`).

## What else is checked

All of these are computed from the bundles and stored in `comparison.json`:

- **Same-job / same-host** — every non-warm-up bundle's manifest must agree
  on `host.job_id` and `host.hostname`. Different values means the reps were
  rescheduled across nodes (or someone ran an arm on another machine); the
  command reports the grouping and exits 2 unless `--allow-mixed-jobs`.
  Bundles without a manifest count as a violation too.
- **TTFT parity** — which token kind each arm's TTFT was stamped on and the
  median chunk index of the first token; warns when they differ. See
  [Decode stalls, cache classes, TTFT parity](itl-cache-metrics.md#3-ttft-parity-4).
- **TTFT mix/latency decomposition** — when the runs carry a `group_by:
  [cache_class]`, the TTFT difference to the reference is split into a
  hit-mix and a per-class latency part (`decomposition-<arm>.md`).
- **Void runs** — bundles whose `summary.void.void` is true (a
  [void rule](capability-manifest.md) failed) are excluded and listed.

## Re-running the comparison

The comparison never needs the servers. `benchmaker compare` rebuilds it
from finished bundles (any depth: an arms root, or a directory of roots):

```bash
benchmaker compare ./runs/20260701T120000Z
benchmaker compare ./runs --reference vllm --metric ttft_p99 --metric itl_p99.9 --format json
benchmaker compare ./runs/20260701T120000Z ./runs/20260702T090000Z --warmup-reps 2
```

| Flag                 | Meaning                                                       |
| -------------------- | ------------------------------------------------------------- |
| `--reference ARM`    | Ratio baseline (default: the first arm in the plan)           |
| `--metric ALIAS`     | Add a metric column (repeatable; default: `DEFAULT_METRICS`)   |
| `--warmup-reps N`    | Override the number of excluded reps (default: from `plan.json`) |
| `--allow-mixed-jobs` | Do not exit 2 on a same-job violation                         |
| `--format md\|json`  | Table for humans / `comparison.json` on stdout                |

The same functions are importable (`benchmaker.core.arms.discover_arm_dirs`,
`runs_from_arm_dirs`, `build_comparison`) for notebooks.

## From Python

```python
from benchmaker.core.arms import ArmsRunner

runner = ArmsRunner(cfg_dict, out_dir="./runs", labels={"gpus": "8xh100"})
result = runner.run()            # or `await runner.run_async()` inside an event loop
print(result["arms"]["rill"]["metrics"]["ttft_p99"]["ratio_vs_ref"])
```

`ArmsRunner.run_async()` is what `benchmaker arms` calls; it is safe to drive
from an existing loop (tests do).
