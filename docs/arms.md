# Multi-arm comparisons (ABBA, per-rep bundles, same-job check)

Issues #3, #8 and #9. A/B-ing two serving stacks by running one, then the other, an hour
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
    arms.log                        # one line per launch / readiness / run / teardown
    arms/
        rill/
            launch-rep-0.log        # stdout/stderr of the start hook
            ready-check-rep-0.log   # readiness check (URL statuses / command output)
            stop-rep-3.log          # stop hook, post_stop_check.log after it
            rep-0/                  # a complete run bundle (meta/summary/samples/manifest…; meta.hooks)
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
  launch_timeout_s: 600       # start hook
  ready_timeout_s: 600        # ready_check polling budget
  stop_timeout_s: 120         # stop hook
  metrics: [ttft_p50, ttft_p99, itl_p99.5, stalls_per_req, stall_excess_ms, rps, err_pct]
  arms:
    vllm:
      config:                               # deep-merged over the shared config below
        workload_type: {url: "http://vllm:8000/v1/chat/completions"}
      env: {VLLM_USE_V1: "1"}               # env for the hooks; recorded in the manifest
      start: "./scripts/launch-vllm.sh"     # run before the arm's first measured run ("launch" is an alias)
      stop: "./scripts/stop-vllm.sh"        # run after the arm's last rep — also on failure or Ctrl-C
      relaunch: once                        # once | every_rep | visit
      ready_check:                          # poll until the server answers, then start the rep
        urls: ["http://vllm:8000/health"]
        status: 200
        timeout_s: 600
      post_stop_check:                      # the teardown really freed the GPUs
        run: "! nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q ."
      labels: {image: "vllm:0.9.1"}
    rill:
      config:
        workload_type: {url: "http://rill:8000/v1/chat/completions"}
      env: {RILL_KV_CACHE_TIER: host}
      start: "./scripts/launch-rill.sh"
      stop: "./scripts/stop-rill.sh"
      relaunch: every_rep
      ready_check: {urls: "http://rill:8000/health"}   # a single URL may be given inline
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
- `env` is exported to the hooks and recorded in the bundle's
  `manifest.json` under `env`, so a reader can see which knobs differed.
  Hooks also receive `BENCHMAKER_ARM` and `BENCHMAKER_REP`.
- `warmup_reps` reps per arm are run first (per the order) and **excluded**
  from the table; they are kept on disk and listed as `excluded` in the
  comparison.

## Lifecycle hooks (issue #9)

A *launch* of an arm is one `start` → reps → `stop` cycle. `relaunch`
decides how many there are per arm:

| `relaunch`   | `start` runs                                   | `stop` runs                                                 |
| ------------ | ---------------------------------------------- | ----------------------------------------------------------- |
| `once`       | before the arm's first rep                     | after the arm's last rep, in plan order                      |
| `every_rep`  | before every rep                               | after every rep                                              |
| `visit`      | before every *contiguous run* of the arm's reps in the plan | when the next plan entry is another arm, so the GPUs are free before it starts |

In ABBA order (`A B B A …`) `visit` gives `A↑ A₀ A↓ B↑ B₀ B₁ B↓ A↑ A₁ A↓`:
both arms fit on the same GPUs, each is torn down before the other needs
them, and the comparison still interleaves in time. With `once` the plan is
`A↑ B↑ A₀ B₀ B₁ A₁ B↓ A↓`: **both servers are up at the same time** while
the other arm's reps run — the right choice when they do not share GPUs
(and you want to skip the restart cost), the wrong one when they do: each
arm's numbers then include the other's memory footprint and any background
load. The runner warns at start when two managed arms overlap this way
(`arms a and b will both be live ... use relaunch: visit or every_rep`),
`--dry-run` prints the same note, and `plan.json` records, for every plan
entry, whether a stop follows (`stops_after`) plus the list of `overlaps`
(issue #14). An arm with no `start` but a `stop` also works (you start it
by hand, benchmaker stops it); an arm with neither is never "live" and so
never overlaps.

Both hooks run through the shell from the working directory with a timeout
(`launch_timeout_s` / `stop_timeout_s`; a hung hook is killed with its whole
process group). Their records — command, exit code, duration, log path and
the last 2 KB of output — go into every rep bundle of the launch under
`meta.hooks` (`start`, `ready_check`, `stop`, `post_stop_check`). Logs are
kept next to the bundles (`launch-rep-<k>.log`, `stop-rep-<k>.log`, or
`stop-final.log` for the end-of-run teardown).

Failure handling:

| What fails                | What happens                                                                                       |
| ------------------------- | -------------------------------------------------------------------------------------------------- |
| `start` exits non-zero    | `stop` runs (cleanup), then the whole run aborts — nothing of this arm was measured yet.           |
| `ready_check` never passes| **Every rep of the launch** is void (`meta.void_violations: [{rule: ready_check, …}]`), not only the first one that reused it: the reps still run (so you see what an unready server does) but none of them reaches the table. The launch is stopped afterwards as usual. |
| `stop` exits non-zero     | Recorded (`meta.hooks.stop.ok: false`), warned about, and — with `stop_failure_voids` (default `true`) — every rep of the launch is **void**: a server that failed to stop was likely also failing in other ways (issue #14). Set `stop_failure_voids: false` to keep the reps and only get the warning. |
| `post_stop_check` fails   | The launch's reps are **void** after the fact (a server that did not go away makes the next arm's numbers meaningless): `patch_bundle` rewrites `meta`/`summary` of each bundle written under the launch, and the comparison reloads them. |
| Ctrl-C while `start` runs | The hook process group is killed (hooks are polled every 250 ms for cancellation, not waited out to `launch_timeout_s`), the arm counts as live so its `stop` hook runs in `finally`, and nothing of the server is left behind (issue #14). |
| A rep raises / Ctrl-C     | Every live arm's `stop` runs in the runner's `finally` before the error propagates.                |

The end-of-run teardown (`finally`) logs to `stop-final.log`; a stop that
follows from the plan (`visit`, `every_rep`, or `once` after its last
entry) is a regular `stop-rep-<k>.log` of the entry it follows.

`ready_check` / `post_stop_check` accept `urls` (or a single `url`) with the
expected `status` (default 200; connection refused counts as 0), a shell
`run` with `expect_exit` (default 0), `timeout_s` and `interval_s`
(readiness polls until the budget is spent; the teardown check runs once).
A URL check is a plain `GET` — point it at `/health`, `/v1/models` or the
like. Pass `--dry-run` to see the plan, and read `arms.log` for what each
hook did.

## Several servers per arm (issue #8)

An arm may fan out to one server per model — a router or a two-tier cache
with a small and a large model behind it — with an endpoint map inside the
arm (`endpoints:` at the top level of the shared config works the same way):

```yaml
arms:
  arms:
    rill:
      endpoints:
        qwen2.5-7b: http://rill:8401                       # base URL: path of workload_type.url is kept
        qwen2.5-32b: {url: "http://rill:8402/v1/chat/completions", capability: "http://rill:8402/server_info"}
      config: {workload_type: {url: "http://rill:8401/v1/chat/completions", model: qwen2.5-7b}}
workload_type: {type: openai, stream: true}
mix:
  lanes:
    - {name: small, workload: {type: static, items: ["hi"]}, rate: 4, model: qwen2.5-7b}
    - {name: big,   workload: {type: static, items: ["ho"]}, rate: 2, model: qwen2.5-32b}
```

A request's model — set by its lane's `model`, a session-replay/trace row's
`model` label, or carried by a workload item — selects the endpoint; the
request is re-routed to it (the endpoint's `headers` are added) and stamped
`meta.model` / `meta.endpoint`. The key is the name requests carry; an
endpoint entry may add `model: <served-name>` to rewrite the body's `model`
when the key is an alias (`small: {url: ..., model: qwen2.5-7b}`).
`group_by: [model]` is added automatically, so the summary (and every
lane/phase block) has per-model groups next to the arm aggregate. Capability
snapshots, `void_rules` and the manifest run **per endpoint**:
`capability.<model>.{start,end}.json` in the bundle (`read_bundle()` →
`capability.endpoints`), `manifest.endpoints[model]` with each server's URL
and reported model/revision.

**Nothing falls back to the default URL (issue #15).** With `endpoints:` set
every model a request can carry must name an endpoint, and `build_config`
checks this up front: the workload type's own `model` (a lane without
`model` sends it), each lane's `model`, and every distinct `model` label of
a session-replay or trace workload (`workload ... has rows labelled [...] with
no endpoint`) — filter the workload (`model_label:`) or add the endpoints.
A request that still slips through (`RoutingError`, e.g. a hook that rewrites
the body) is failed, counted (`routing_unmatched`), logged once, and the rep
is voided with a `routing` violation; it is never sent to `workload_type.url`.

Void rules apply **per endpoint**: an unscoped rule is checked against every
endpoint's snapshot; a rule with `endpoint: <model>` (or a path written
`endpoints.<model>.<rest>`) only against that one — so a rule for the small
model's fallbacks does not void the big model's reps, and vice versa. A
scoped rule that names no configured endpoint is a config error. Each
endpoint is captured even without an explicit `capability` URL: the shared
`capability.url` with scheme/host swapped, else the OpenAI-style `/v1/models`
listing next to its request URL (`capability: {enabled: false}` skips it);
the violation records the endpoint it came from.

Decomposition and TTFT parity (see [metrics.md](metrics.md)) ignore void
reps the same way the table does (issue #17).
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
  [void rule](capability-manifest.md) failed, or a lifecycle check did —
  see above) are excluded and listed.

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
