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

## Void rules

```yaml
void_rules:
  - name: served-model           # required, unique
    path: models.0               # dotted path into the (normalized) snapshot; ints index lists
    op: eq                       # empty | eq | ne | lt | le | gt | ge | in | not_in | contains | not_contains
    value: qwen2.5-32b
    at: both                     # start | end | both (default end)
    ignore_missing: false        # a missing path is not a violation
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
  "prompt_set": {"name": "jsonl:sharegpt_v3.jsonl", "hash": "sha256:…", "items": 400},
  "model": {
    "id": "qwen2.5-32b",                 // workload_type.model, or the server's
    "revision": "3b1f…",                 // from config (model_revision) or the capability snapshot
    "revision_source": "capability:model_revision",
    "capability_model": "qwen2.5-32b"
  },
  "host": {"hostname": "gpu-07", "platform": "…", "python": "3.12.2", "pid": 4242, "cpu_count": 128,
           "job_id": "123456", "source": "SLURM_JOB_ID"},
  "env": {"CUDA_VISIBLE_DEVICES": "0,1,2,3", "VLLM_USE_V1": "1", "RILL_KV_CACHE_TIER": "host"},
  "labels": {"arm": "rill", "rep": 1, "gpus": "8xh100"}
}
```

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
| You, before trusting a table      | same `prompt_set.hash`, same `model.revision`, same `host`         |

A difference in `prompt_set.hash` between two bundles that are supposed to
compare the same prompts is the most common silent error in benchmarking;
check it first.
