# Changelog

## Unreleased

- **Decode stalls and pooled ITL (#1):** every streamed request's raw
  inter-token gaps are pooled across requests into
  `workload_metrics.itl_ms_pooled` (`count/mean/min/max/p50/p90/p99/p995/p999`),
  and gaps above `stall_ms` (default 12; `stall_ms: none` disables) are
  accounted in `workload_metrics.stalls` (per-request stall counts and time,
  stall-length percentiles, `total_time_ms`, `excess_ms`, a log2 histogram).
  `dump_series: true` writes each sample's `series` into `samples.jsonl`.
- **Cache classes and TTFT decomposition (#2):** `OpenAIChatWorkloadType`
  tags every response with `meta.cache_class` (`device_hit` / `restore` /
  `partial` / `miss`, from `usage` + headers; override with
  `cache_classifier:`), the new `group_by: [meta key, ...]` config key breaks
  the summary — and every lane/phase block — down by any `Sample.meta` key,
  and `benchmaker.core.compare.decompose_ttft` splits a TTFT difference
  between two runs into an exact mix (hit-rate) and latency (per-class) term.
  TTFT parity: `extra.ttft_token_index` and the `ttft_token` setting are
  compared across arms and warned about when they differ.
- **Capability, manifest, void runs (#4):** `capability: {url}` fetches the
  server's self-description before and after the run
  (`capability.start/end.json`); `void_rules:` assert on it (`eq/ne/lt/…/in/
  contains` over dotted paths, at `start|end|both`) and mark the run void
  (`summary.void`, `meta.void`, excluded from `collect`/`compare`,
  `benchmaker run` exits 3). Every bundle now carries `manifest.json`
  (config hash, prompt-set content hash, model id/revision, host + Slurm/
  PBS/LSF/K8s job id, filtered env; `model_revision:` / `record_env:` keys).
- **Phases, warm-up, capacity-revisit (#5):** `phases:` runs timed stages
  back-to-back, each with its own lanes (own load model starting at the
  phase's t=0, optional per-lane `model`/`meta`, `drain`), tags samples with
  `meta.phase` and summarizes each phase (`summary.phases`, with `lanes` and
  `groups` inside). `warmup_requests:` / `warmup_s:` exclude the first
  requests from every summary while still recording them. New
  `capacity` workload: a warm pass over N long documents (real JSONL or
  synthesized, token-exact with `tokenizer:`) then K revisit passes (sequential
  or seeded random), each visit tagged `visit`/`pass_index`/`doc_id`, with
  `stage: warm|revisit` views for phased recipes.
- **Multi-arm comparisons (#3):** new `benchmaker arms` command (YAML
  `arms:` block; `reps`, `order: abba|rep-major|random`, `warmup_reps`,
  `reference`, per-arm `config` deep-merge / `env` / `launch` hook /
  `relaunch`, `metrics`) runs one bundle per (arm, rep) and writes
  `comparison.md/.json`; `benchmaker compare` rebuilds the table from
  finished bundles. Per-rep medians with min–max, ratios vs the reference,
  same-job/same-host enforcement (exit 2 unless `--allow-mixed-jobs`), void
  exclusion, TTFT parity warnings and the TTFT decomposition. Library:
  `benchmaker.core.arms`, `benchmaker.core.compare`.
- Docs: `itl-cache-metrics.md`, `arms.md`, `capability-manifest.md`,
  `phases-capacity.md`, `trace-replay.md`; `cli-and-yaml.md`, `load-models.md`,
  `metrics.md`, `workloads.md` updated.
- Added the `agentx` recipe and workload (`benchmaker agentx`, YAML `type:
  agentx`): the SemiAnalysis AgentX v1.0 agentic-coding benchmark
  (https://inferencex.semianalysis.com/agentx). Replays the public WEKA-format
  Claude Code trace corpus (`semianalysisai/cc-traces-weka-062126[-256k]`, or
  a local JSONL) with block-hash synthetic prompts that preserve the
  recording's byte-exact prefix-cache structure, session-tree DAG dispatch
  (main turns + concurrent subagent streams with recorded spawn/join/gap
  timing and a system-idle cap), seeded mid-session start points, a per-lane
  primer + warmup pass excluded from metrics, per-play cache-bust markers,
  recorded decode lengths (`ignore_eos`), and a default one-hour profiling
  window. Includes `tools/agentx/prepare.py` for corpus download/subsetting
  and `docs/agentx.md`.
- **Serving-trace replay (#6):** `mooncake-trace` / `azure-trace` /
  `trace-replay` workloads (JSONL/JSONL.gz/parquet, CSV/CSV.gz/.zip;
  `max_records`, `max_duration_s`, `max_input/output_tokens`) keep every
  request's arrival time — via `load: {type: trace, speed}`
  (`TraceArrivalsLoad`, not re-sorted) — and Mooncake's `hash_ids` prefix
  structure (every block id rendered deterministically, so the served prefix
  cache sees the trace's sharing graph). Trace fields ride on `sample.meta`
  and numeric ones on `sample.extra` (`trace_input_tokens`, …). Workload
  item `meta` dicts are now carried onto `Request.meta` instead of being sent
  in the body.
- **Breaking:** renamed the `trajectory-replay` workload to `agentic`, since it
  represents multi-turn *agentic* workloads. The CLI recipe is now
  `benchmaker agentic` (was `trajectory-replay`); the YAML workload `type:
  trajectory` is now `type: agentic`; and `TrajectoryReplayWorkload` is now
  `AgenticWorkload` (in `benchmaker.workloads.agentic`). The underlying
  recorded-data concept (a "trajectory" of agent turns) and its flags
  (`--max-trajectories`, `--max-turns-per-trajectory`, `expand_trajectory`)
  are unchanged. The separate SWE-bench `trajectory` module and
  `swebench-replay` recipe are unaffected.
- Fixed `OpenAIChatWorkloadType` silently ignoring `delta.reasoning_content`
  for thinking models (GLM-4.x, DeepSeek-R1, Qwen3-thinking, gpt-5 reasoning,
  …), which made `ttft_s`, `itl_ms_*`, and (when `usage` was absent)
  `tokens_out` wrong by the entire reasoning phase. Reasoning tokens are now
  counted the same as content tokens for TTFT/ITL, the
  `usage.completion_tokens_details.reasoning_tokens` breakdown is surfaced
  (`reasoning_tokens` / `content_tokens`), and a new `ttft_token`
  (`"any"` | `"content"`, default `"any"`) knob selects whether the headline
  `ttft_s` measures the first token of any kind or the first visible content
  token. The first-content time is always available separately as
  `content_ttft_s` (#14).

## 0.1.4 — 2026-06-24

- Fixed silent loss of streamed `usage` (real `prompt_tokens`, `cached_tokens`)
  and truncated eval text when an SSE event was split across HTTP chunk
  boundaries — common on the `mix` path under high concurrency with large
  prompts, which zeroed prefix-cache hit rates. SSE parsing now reassembles
  lines across chunks (shared by the `openai-chat`, `sglang`, and eval paths),
  and the `openai-chat` workload warns once when `include_usage` was requested
  but no usage block was parsed (#13).
- Added `DeepRAGWorkload` and a HotpotQA distractor JSONL preparation tool for
  prefill-heavy retrieval benchmarks.
- Added concurrent named workload lanes with independent load models, durable
  `meta.lane` tags, and per-lane metric summaries.
- Added a phase-swinging DeepRAG + ShareGPT configuration example.
