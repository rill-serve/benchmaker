# Replaying published serving traces (Mooncake, Azure)

Issue #6. A synthetic benchmark decides its own request sizes and arrival
pattern, which makes it easy to read too much into: a prefix cache that
looks great on 400 ShareGPT prompts with 50 % shared system prompt may be
mediocre on production traffic whose sharing is a session tree with a long
shared history and a unique tail. Benchmaker can replay the two public
serving traces that carry enough information to reproduce this:

| Trace | Where | Fields used | Prefix structure |
| ----- | ----- | ----------- | ---------------- |
| **Mooncake** (FAST '25) | `kvcache-ai/Mooncake` — `mooncake_trace.jsonl` (or the parquet mirror) | `timestamp` (ms), `input_length`, `output_length`, `hash_ids` | **exact** — see below |
| **Azure LLM inference trace** | `Azure/AzurePublicDataset` — `AzureLLMInferenceTrace_conv.csv`, `…_code.csv` (+ `.zip`/`.gz`) | `TIMESTAMP`, `ContextTokens`, `GeneratedTokens` | none — independent prompts of the recorded size |

Replay keeps every request's **own arrival time** (open-loop; late tickets
fire immediately), its **own input/output length**, and — for Mooncake —
its **own prefix-sharing structure**.

## Why `hash_ids` matter

Mooncake's `hash_ids` are chained hashes of 512-token blocks mapped to
global ids: a request with `hash_ids = [a, b, c]` consists of block `a`,
then `b`, then `c`, and any other request that also starts with `[a, b]`
shares those two blocks as a prefix. Benchmaker renders every id
deterministically as its own block of synthetic text, so the server's prefix
cache sees exactly the trace's sharing graph (common system prompt → shared
conversation history → unique tail), not a hit rate "somewhere in between".

```
req 1: hash_ids [a, b, c]   →  [block a][block b][block c]
req 2: hash_ids [a, b, d]   →  [block a][block b][block d]   ← 1024 cached tokens
req 3: hash_ids [e]         →  [block e]                      ← cold
```

A request whose `input_length` is not a multiple of 512 is the last block
*truncated* to the remainder; `input_length` larger than its blocks is
padded with a request-unique tail. The published data's timestamps are
milliseconds relative to the first request.

## Getting the files

```bash
# Mooncake (JSONL, ~1.1 GB uncompressed; gzip is read transparently)
wget https://huggingface.co/datasets/kvcache-ai/Mooncake/resolve/main/mooncake_trace.jsonl.gz

# Azure (CSV inside a zip, or the HF mirror's csv.gz)
wget https://azurepublicdatasettraces.blob.core.windows.net/azurepublicdatasetv1/azure_public_conv.csv.gz
# → AzureLLMInferenceTrace_conv.csv
```

## YAML

```yaml
workload_type:
  type: openai
  url: http://localhost:8000/v1/chat/completions
  model: qwen2.5-32b
  stream: true
  ignore_eos: true            # optional: make decode length == output_length exactly

workload:
  type: mooncake-trace          # mooncake | mooncake-trace | azure | azure-trace | trace-replay (auto-detect)
  path: .local/mooncake_trace.jsonl.gz
  # --- subsetting (applied before anything else; records keep their original times) ---
  max_records: 20000            # first N records by arrival
  max_duration_s: 600           # requests arriving in the first 10 minutes
  max_input_tokens: 16384
  max_output_tokens: 2048
  # --- rendering ---
  tokenizer: Qwen/Qwen2.5-7B    # token-exact prompts (needs `pip install -e .[hf]`); default: char mode
  chars_per_token: 4.0          # char mode only
  block_size: 512               # tokens per hash block (Mooncake: 512)
  match_output_tokens: true     # max_tokens = output_length (false: a fixed max_tokens)
  max_tokens: 256               # used when match_output_tokens is false, and as a cap
  loop: false                   # restart the trace when exhausted

load:
  type: trace                   # one ticket per record at its own arrival time (TraceArrivalsLoad)
  speed: 1.0                    # 2.0 = replay at twice the rate
  max_requests: 10000

group_by: [cache_class]
warmup_requests: 8
```

`load: {type: trace}` (aliases `trace-arrivals`, `arrivals`) is required
for a faithful replay: unlike `TracePacedLoad` it does **not** re-sort the
times, because ticket *i* belongs to record *i* — so every request gets its
own prompt *and* its own arrival. Any other load model (`poisson:8`, …)
also works when you only want the prompt population; benchmaker refuses
`type: trace` with a workload that has no arrival times.

Workload types `mooncake-trace` / `azure-trace` set the source explicitly;
`trace-replay` sniffs the file (`.parquet` → column names; CSV header with
`TIMESTAMP` + `ContextTokens` → azure; otherwise mooncake).

## What is recorded

Each item carries the trace fields in `meta`, which reach `samples.jsonl`
as `meta` and (for the numeric ones) as `extra` metrics summarized in
`workload_metrics`:

| Field                      | Where            | Meaning                                   |
| -------------------------- | ---------------- | ----------------------------------------- |
| `trace_index`              | `meta`           | record number in the (sorted) trace       |
| `trace_t_s`                | `meta`, `extra`  | original arrival time (s)                 |
| `trace_input_tokens`       | `meta`, `extra`  | recorded input length                     |
| `trace_output_tokens`      | `meta`, `extra`  | recorded output length                    |
| `trace_blocks`             | `meta`, `extra`  | number of hash blocks (0 for Azure)       |
| `prompt_tokens_hint`       | `extra`          | rendered prompt size (tokens)             |
| `cache_class`              | `meta`           | `device_hit` / `restore` / `partial` / `miss` |

So a run reports the trace's token distribution (`trace_input_tokens.p50`,
`…max`) next to what the server measured (`prompt_tokens`, `cached_tokens`),
and `group_by: [cache_class]` gives the hit mix the trace's structure
produced on this server. Compare `trace_input_tokens` with `prompt_tokens`
to see how far char-mode rendering is from the real tokenizer.

## CLI

```bash
benchmaker run mooncake.yaml --out-dir ./runs --run-id mooncake-vllm --label trace=mooncake
benchmaker run azure.yaml     --out-dir ./runs --run-id azure-rill
```

There is no dedicated recipe subcommand; `benchmaker run` with the YAML
above is the whole thing. A multi-arm replay (same trace, two servers) is
the usual [arms](arms.md) config with the workload block from above as the
shared base.

## Python

```python
from benchmaker.workloads.traces import load_trace, TraceReplayWorkload, MooncakeTraceWorkload

records = load_trace(".local/mooncake_trace.jsonl.gz", max_records=20000, max_duration_s=600)
wl = TraceReplayWorkload(records, tokenizer="Qwen/Qwen2.5-7B", match_output_tokens=True)
item = await wl.next_item()          # {"messages": [...], "max_tokens": 87, "meta": {"trace_index": 0, ...}}
wl.arrival_times                     # → TraceArrivalsLoad(times, speed=...)
```

`load_mooncake_trace(path, timestamp_unit="ms"|"s"|"auto", block_size=512, …)`
and `load_azure_trace(path, …)` are the per-source loaders; both accept
plain, `.gz` and (Azure) `.zip` files, drop malformed rows with a warning,
sort by arrival and subtract the first timestamp so `t_s` starts at 0.

## Caveats

- **Prompts are synthetic text**, not the original conversations (those are
  not in the traces). Only sizes and sharing structure are faithful; if a
  model's tokenization makes the synthetic blocks denser than real prose,
  use `tokenizer:` to be token-exact.
- **Azure has no prefix information**, so every request is a cold prompt of
  its recorded size: it measures prefill/decode throughput under the
  production arrival process, not caching. Wrap several records behind a
  shared system prompt with a custom workload if you need a hit mix.
- The published Mooncake trace spans many hours at high rate; subset with
  `max_duration_s` / `max_records` first and watch `summary.errors` for
  timeouts — an open-loop replay that overruns the server is a legitimate
  result (that is what production did to it), but a client-side
  `max_in_flight` cap would silently turn it into closed loop.
