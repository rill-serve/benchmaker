# Capability snapshots, the run manifest, and void runs

Issue #4. Two bundles can share a config hash and still not be comparable: a
quantized kernel silently fell back to FP16, the decode backend was not the
one the config asked for, the server was restarted on a different revision
between reps, or the two arms ran in different Slurm jobs on different nodes.
Nothing in the request/response stream says so.

Benchmaker addresses this with three things that every run bundle now has:

1. **Capability snapshots** — the server's self-description, fetched before
   and after the run (`capability.start.json`, `capability.end.json`).
2. **The run manifest** — `manifest.json`, a positive record of *what was
   actually benchmarked*: config hash, prompt-set hash, model id/revision,
   host + job id, and the env knobs that change behavior.
3. **Void rules** — assertions on the snapshots; a violated rule marks the
   run **void** so comparisons exclude it instead of being polluted by it.

## Capability snapshots

```yaml
capability:
  url: http://localhost:8000/v1/models     # or /get_server_info, /get_model_info, ...
  headers: {Authorization: "Bearer ${API_KEY}"}
  timeout_s: 10
```

The URL is fetched once right before the first request and once right after
the last one, with the stdlib client (not the benchmark's connection pool),
and stored verbatim:

```json
// capability.end.json
{
  "at": "end",
  "ok": true,
  "url": "http://localhost:8000/get_server_info",
  "wall_time": 1751366403.12,
  "data": {"model": "qwen2.5-32b", "model_revision": "3b1f…", "dtype": "fp8",
           "attention_backend": "flashinfer", "max_num_seqs": 256, ...}
}
```

A failed fetch is recorded too (`"ok": false, "error": "HTTP 404"`), never
raised — a benchmark against a server that does not expose a description is
still a benchmark; it just cannot be void-checked.

`/v1/models` is special-cased: its `{"object": "list", "data": [...]}` is
normalized to `{"models": ["id", ...], "models_detail": [...]}` so rules can
say `models contains "qwen2.5-32b"`. Everything else (vLLM/SGLang
`get_server_info`, `get_model_info`, a custom endpoint) is kept as-is.

### Several endpoints (issue #8)

With `endpoints:` — one server per `model`, see
[Multi-arm comparisons](arms.md#several-servers-per-arm-issue-8) — the
snapshot is taken from **every** endpoint (each entry may carry its own
`capability:` URL, default `/v1/models` on the endpoint's base URL) and
stored as a pair per endpoint:

```
capability.<model>.start.json
capability.<model>.end.json      # model name with / : . @ replaced by _
```

`read_bundle()["capability"]` then has `{"start": null, "end": null,
"endpoints": {"<model>": {"start": {...}, "end": {...}}}}` (the plain keys
stay for single-endpoint runs), `void_rules` are evaluated against every
endpoint's snapshots (a violation names the endpoint in its `path`:
`endpoints.<model>.models.0`), and `manifest.endpoints.<model>` records that
server's URL and the model id / revision it reported. An endpoint without
its own `capability:` URL gets the shared one with scheme/host swapped, else
the `/v1/models` listing next to its request URL (`url_source: default` in
the snapshot; a failed default capture is logged at info level, a failed
URL you typed at warning level).

## Void rules

```yaml
void_rules:
  - name: served-model           # required, unique
    path: models.0               # dotted path into the (normalized) snapshot; ints index lists
    op: eq                       # empty | eq | ne | lt | le | gt | ge | in | not_in | contains | not_contains
    value: qwen2.5-32b
    at: both                     # start | end | both (default end)
    ignore_missing: false        # a missing path is not a violation
    endpoint: qwen2.5-7b         # only check this rule against that endpoint of an `endpoints:` map (#15)
  - name: no-fallback
    path: attention_backend
    op: eq
    value: flashinfer
  - name: dtype
    path: dtype
    op: in
    value: [fp8, bf16]
  - name: max-batch
    path: max_num_seqs
    op: ge
    value: 128
    at: start
  - name: one-model
    path: models
    op: in                      # list actual: every element must be in value
    value: [qwen2.5-32b]
```

Notes:

- A snapshot that is missing or failed to fetch is a violation for every
  rule evaluated against it ("snapshot unavailable"), because a run that
  could not verify its own preconditions is exactly the case to catch.
- `at: both` evaluates the rule twice and records which snapshot violated.
- Rules are evaluated when the summary is built; the result lands in
  `summary.json`:

```json
"void": {
  "void": true,
  "violations": [
    {"rule": "no-fallback", "at": "end", "path": "attention_backend",
     "op": "eq", "expected": "flashinfer", "actual": "xformers", "reason": null}
  ],
  "checks": [{"rule": "served-model", "at": "start", "path": "models.0", "op": "eq", "expected": "qwen2.5-32b", "ok": true}, ...]
}
```

- `benchmaker collect` and `benchmaker compare` **exclude void bundles** and
  list them (`_excluded (void/warm-up): …_`); `benchmaker arms` shows
  `_**arm VOID**: rule …_`. The bundle itself is untouched.
- `benchmaker run` exits with code 3 when the run is void (the bundle is
  still written), so a CI script can refuse to publish it.

With `endpoints:`, an unscoped rule is checked against **every** endpoint's
snapshot and a scoped one (`endpoint:` — or a path written
`endpoints.<model>.<rest>`) only against its own, so a rule about the small
model's fallbacks does not void the big model's reps (issue #15). A scoped
rule naming no configured endpoint, or naming the endpoint but no field of
its snapshot, is a config error (`validate_rule_scopes`).

From Python: `benchmaker.core.capability.parse_void_rules`,
`check_rule`, `evaluate_void_rules`.

## The run manifest

`manifest.json` is written next to `meta.json` by every `write_bundle` call
(`benchmaker run`, `benchmaker arms`, `runner.write_bundle(...)`):

```json
{
  "manifest_version": 1,
  "benchmaker_version": "0.1.4",
  "benchmaker_git_commit": "…",
  "run_id": "rep-1",
  "started_at": "2026-07-01T12:00:03+00:00",
  "config_hash": "sha256:…",             // canonical JSON of the *resolved* config, not the YAML text
  "prompt_set": {"name": "jsonl:sharegpt_v3.jsonl", "hash": "sha256:…", "items": 400,
                 "provenance": {"source": "mooncake", "records": 20000, "send": "token_ids", "namespace": "m",
                                "token_whitelist": {"size": 151808, "source": "tokenizer:Qwen/Qwen2.5-7B"},
                                "max_seq": 32768, "dropped": {"max_seq": 12}}},
  "model": {
    "id": "qwen2.5-32b",                 // workload_type.model, or the server's
    "revision": "3b1f…",                 // from config (model_revision) or the capability snapshot
    "revision_source": "capability:model_revision",
    "capability_model": "qwen2.5-32b"
  },
  "host": {"hostname": "gpu-07", "platform": "…", "python": "3.12.2", "pid": 4242, "cpu_count": 128,
           "job_id": "123456", "source": "SLURM_JOB_ID"},
  "env": {"CUDA_VISIBLE_DEVICES": "0,1,2,3", "VLLM_USE_V1": "1", "RILL_KV_CACHE_TIER": "host"},
  "labels": {"arm": "rill", "rep": 1, "gpus": "8xh100"},
  "endpoints": {"qwen2.5-7b": {"url": "http://rill:8401/v1/chat/completions", "capability_model": "qwen2.5-7b",
                               "revision": "3b1f…", "revision_source": "capability:model_revision"}}
}
```

`prompt_set.provenance` (when the workload has it — trace replay, capacity,
…) is the workload's own account of what it rendered: for a trace, the
source, the record count after subsetting, `send: text|token_ids`, the
`namespace` / `unique_tail` / `block_size` knobs, the token whitelist and
how many rows each `max_*` rule dropped. Two bundles with the same
`prompt_set.hash` have the same prompts; two with the same trace file but
different `provenance.dropped` do not.

How each field is found:

- **`config_hash`** — `sha256` of the canonical JSON of the resolved config
  (`meta.source_config`). Two YAMLs that differ only in env-substitution
  syntax but resolve to the same thing hash the same; two that differ in a
  real knob do not.
- **`prompt_set.hash`** — the workload's `content_hash()` when it has one
  (`JsonlWorkload`, `HFDatasetWorkload`, trace and capacity workloads do),
  plus `item_count()`. "Both runs say `sharegpt`" becomes checkable.
- **`model.revision`** — the explicit `model_revision` kwarg of the
  workload-type if given (`revision_source: "config"`), otherwise a
  revision-looking field of the capability snapshot (`model_revision`,
  `revision`, `commit_hash`, `git_commit`, `model_path`, `checkpoint`,
  `build`; `source: "capability:<key>"`). Values like `main`, `latest`,
  `HEAD` are *not* treated as a revision — two builds can both say `main`.
- **`host.job_id`** — the first of `SLURM_JOB_ID` (or `SLURM_ARRAY_JOB_ID` +
  `SLURM_ARRAY_TASK_ID`), `PBS_JOBID`, `LSB_JOBID`, `KUBERNETES_JOB_ID`,
  `RILL_JOB_ID` that is set; the hostname otherwise (`source: "hostname"`).
- **`env`** — variables matching `record_env` (default:
  `CUDA_VISIBLE_DEVICES`, `CUDA_DEVICE_ORDER`, `VLLM_*`, `SGLANG_*`,
  `NCCL_*IB*`) plus any `env_overrides` passed to `write_bundle` (multi-arm
  runs pass the arm's `env:` there). Secrets never match these globs; if
  you add your own patterns, keep it that way.

Pass `record_env: [MY_SERVER_*]` in YAML to record more.

## What uses it

| Consumer                          | Uses                                                               |
| --------------------------------- | ------------------------------------------------------------------ |
| `benchmaker collect` / `compare`  | exclude void bundles; show `_VOID` rows                            |
| `benchmaker arms` same-job check  | `host.job_id` / `host.hostname` must agree across non-warm-up reps |
| `benchmaker arms` TTFT parity     | `meta.source_config.workload_type.ttft_token`                      |
| `benchmaker arms` void exclusion  | `meta.void` / `meta.void_violations` — incl. lifecycle checks (#9):   |
|                                   | `meta.hooks.{start,ready_check,stop,post_stop_check}` records which  |
|                                   | hooks ran around the rep and whether they passed                    |
| You, before trusting a table      | same `prompt_set.hash`, same `model.revision`, same `host`         |

A difference in `prompt_set.hash` between two bundles that are supposed to
compare the same prompts is the most common silent error in benchmarking;
check it first.

### KV allocation

`manifest.json` records device KV tokens, host-tier bytes and tokens, the
host allocation's bytes per token, sharing, and fleet host bytes under
`allocation`. With several endpoints, `endpoints.<model>.allocation` records
each engine's capacity, and the arm's `allocation.fleet_host_total_bytes`
counts each shared pool once. An uncapped shared pool has an arm total and
unknown per-engine capacity. Each reported value has a `sources` entry with
its origin and input unit. Missing values are written as `null`.

For endpoints serving Rill, set their capability URL to `/get_server_info`
to capture `kv_capacity_tokens`. Point `allocation.kvd_log` at the daemon's
startup log to capture its RAM pool and any per-model limit in GiB. The
daemon socket identifies the shared group. For private deployments, set
`sharing: private` in each endpoint's allocation options.

For vLLM, LMCache, and SGLang, use an engine startup log that contains the
capacity lines. With one endpoint, Benchmaker uses the managed arm's launch
log by default. Set `allocation.startup_log` for an external engine or a
launcher that redirects engine output. With several endpoints, configure
one log per endpoint. Within an arm definition, put these options under
`config.allocation`.

```yaml
allocation:
  endpoints:
    model-a:
      startup_log: logs/model-a.log
      bytes_per_token: 147456
    model-b:
      startup_log: logs/model-b.log
      bytes_per_token: 147456
```

Set `bytes_per_token` to the host KV layout's actual size to convert between
bytes and tokens when one is missing. vLLM `kv_offloading_size` and LMCache
`max_local_cpu_size` use GiB. SGLang's HiCache log reports tokens and can also
report rounded decimal GB. The manifest marks that bytes value and conversions
derived from it as approximate. Supply logs scoped to an engine with a
single capacity report for each cache tier.

`benchmaker compare` shows the capacities alongside the benchmark metrics
and warns when two arms' known fleet totals differ by more than 5% of the
larger total.
