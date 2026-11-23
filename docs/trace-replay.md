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

Text rendering has two sources of drift, though: the server's tokenizer may
merge characters across a block boundary (so block *b* does not start on a
fresh token), and `/v1/chat/completions` prepends the chat template, which
shifts every block off the 512-token grid the trace was recorded on. For a
prefix-cache experiment that grid *is* the point — see
[`send: token_ids`](#send-token_ids-bypass-the-tokenizer-11).

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
  max_seq: 32768                # input + output > max_seq -> row dropped (issue #11)
  # --- rendering ---
  tokenizer: Qwen/Qwen2.5-7B    # token-exact prompts (needs `pip install -e .[hf]`); default: char mode
  chars_per_token: 4.0          # char mode only
  block_size: 512               # tokens per hash block (Mooncake: 512)
  match_output_tokens: true     # max_tokens = output_length (false: a fixed max_tokens)
  max_tokens: 256               # used when match_output_tokens is false, and as a cap
  loop: false                   # restart the trace when exhausted
  # --- issue #11 ---
  send: text                    # text | token_ids (see below)
  namespace: null               # null | <string> | model — salt block content per served model
  unique_tail: 0                # append N request-unique tokens after the last block

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

Rows removed by `max_input_tokens` / `max_output_tokens` / `max_seq` are
counted per reason and land in `manifest.json` under
`prompt_set.provenance.dropped` (together with the replay parameters), so a
reader can see that the "same" trace gave two runs different populations.

### `send: token_ids` — bypass the tokenizer (issue #11)

With `send: token_ids` every block is rendered as **token ids**, not text,
and sent as `prompt: [ids]` to `/v1/completions`:

```yaml
workload_type:
  type: openai-completions      # POST /v1/completions; `prompt` may be a list of ints
  url: http://localhost:8000/v1/completions
  model: qwen2.5-32b
  stream: true
  ignore_eos: true
workload:
  type: mooncake-trace
  path: .local/mooncake_trace.jsonl.gz
  send: token_ids
  tokenizer: Qwen/Qwen2.5-7B    # defines the whitelist of sendable ids (special/added tokens removed)
  # vocab_size: 151936          # …or, without a local tokenizer: [0, vocab_size) minus …
  # exclude_ids: [151643, 151644]
  # exclude_last: 64            # … the reserved ids / the last N ids
  namespace: model              # per-model block content when several models share one trace
  unique_tail: 16               # whole-prompt duplicates stop being byte-identical
```

Why ids: nothing is tokenized server-side, so no block boundary can merge
and no chat template shifts the alignment. Every hash block is exactly
`block_size` ids seeded from its hash id, a truncated last block is the true
prefix of the full one, and equal hash ids give **byte-identical,
page-aligned** prefixes — the sharing graph the cache sees is exactly the
one Mooncake recorded. `prompt_tokens` from the server equals the recorded
`input_length` (+ `unique_tail`).

| Knob         | Effect                                                                                                      |
| ------------ | ----------------------------------------------------------------------------------------------------------- |
| `tokenizer`  | HF id or loaded tokenizer: the whitelist is its vocab minus `all_special_ids` and added tokens (BOS/EOS/PAD, chat control tokens — ids that could end generation early or trip a parser). |
| `vocab_size` | Without a tokenizer: ids are drawn from `[0, vocab_size)` minus `exclude_ids` minus the last `exclude_last` ids. State your tokenizer's reserved range. |
| `namespace`  | Salt mixed into every block seed, so runs of several models from one trace get disjoint block content (a cache warmed by another model's rehearsal is a fake hit). `namespace: model` resolves to the workload-type's (or lane's) `model`; unset keeps the pre-#11 seeds and prompt-set hash. |
| `unique_tail`| Append N request-unique ids after the last block. Requests with identical `hash_ids` — whole-prompt duplicates in the trace — keep the shared prefix but stop being the same prompt. `prompt_tokens_hint` reflects the real size. |

`send: token_ids` with `type: openai` is refused (chat completions cannot
take a list of ids). The `openai-completions` workload-type parses the
`/v1/completions` SSE stream (`choices[*].text`) and otherwise behaves like
the chat type (`passthrough_meta`, `extra_body`, `ignore_eos`, …); its
samples carry `meta.prompt_kind = text|token_ids` and `meta.prompt_len`.

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
| `prompt_kind`              | `meta`           | `text` or `token_ids` (`openai-completions`) |
| `trace_unique_tail`        | `meta`, `extra`  | `unique_tail` in effect                    |
| `cache_class`              | `meta`           | `device_hit` / `restore` / `partial` / `miss` |

`manifest.json` → `prompt_set` records the replay next to the content hash:
`provenance: {source, path, records, send, namespace, unique_tail,
block_size, tokenizer, token_whitelist: {size, source}, max_seq, dropped:
{<reason>: n}}`.

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

- **Prompts are synthetic**, not the original conversations (those are not
  in the traces). Only sizes and sharing structure are faithful; if a
  model's tokenization makes the synthetic blocks denser than real prose,
  use `tokenizer:` to be token-exact — or `send: token_ids`, which is exact
  by construction (the ids are then random whitelisted tokens, not words;
  fine for cache/throughput experiments, meaningless for quality).
- **Token-id prompts need a server that accepts `prompt: [ids]`** on
  `/v1/completions` (vLLM, SGLang, TGI do) and a whitelist matching its
  tokenizer; ids outside the server's vocab are rejected.
- **Azure has no prefix information**, so every request is a cold prompt of
  its recorded size: it measures prefill/decode throughput under the
  production arrival process, not caching. Wrap several records behind a
  shared system prompt with a custom workload if you need a hit mix.
- The published Mooncake trace spans many hours at high rate; subset with
  `max_duration_s` / `max_records` first and watch `summary.errors` for
  timeouts — an open-loop replay that overruns the server is a legitimate
  result (that is what production did to it), but a client-side
  `max_in_flight` cap would silently turn it into closed loop.
