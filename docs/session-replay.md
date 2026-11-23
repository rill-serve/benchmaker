# Session replay (`session-replay`)

Runs a **prepared session file** in which each multi-turn session's turns must go out in order: turn *k+1* is sent
only after turn *k* has completed, plus a think time. Generic JSONL workloads fire every row independently and so
lose this dependency; the `agentic` recipe interleaves sessions, but it expands whole trajectories itself and has
no prompt-set hash and no recorded-output pinning (issue #10).

## File format

JSONL. The first line is a manifest, then one line per request, grouped by session in send order:

```json
{"kind": "manifest", "prompt_set_sha256": "41f30be2...", "think": {"median_s": 10, "sigma": 1.4},
 "probe": {"a": {"messages": [...], "prompt_tokens": 2843}}, "source": {...}, "seed": 1}
{"id": "chat-00012.t03", "session": "chat-00012", "turn": 3, "model": "a",
 "messages": [...], "max_tokens": 180, "think_z": 0.41, "tag": "measure", "expect": {"input_tokens": 5321}}
```

| field | meaning |
|---|---|
| `messages` | the full request, including the **recorded** earlier turns: every arm sees byte-identical prompts |
| `max_tokens` | recorded output length; sent with `min_tokens = max_tokens` and `ignore_eos` |
| `think_z` | the pause before this turn is `median * exp(sigma * think_z)` (lognormal) |
| `after.think_s` | alternatively, a fixed pause in seconds (wins over `think_z`) |
| `at_s` | first turn only: the session's start offset for `--trace-scale` |
| `tag` | `warmup` rows carry `meta.warmup` and are excluded from summaries |
| `model` | a label; `--model-label` replays only those sessions (one endpoint per model) |

`prompt_set_sha256` is the sha256 over the canonical JSON (`sort_keys`, compact separators, UTF-8) of every request
row, each followed by `\n`. The file is re-hashed on load and refused on mismatch.

## Running

```bash
benchmaker session-replay --replay-file chat.jsonl \
  --url http://host:8000/v1/chat/completions --model Qwen/Qwen3-8B \
  --concurrent-sessions 8 --think-median-s 10 --out-dir runs/
```

Session starts, exactly one of:

- `--concurrent-sessions N` (default 8): closed loop, N sessions active; the next starts when one finishes.
- `--session-rate R`: open loop, Poisson session starts at R per second (`--seed`).
- `--trace-scale S`: each session starts at its first row's `at_s` times S.

`--think-median-s` overrides the manifest's median, so one prompt set (one hash) serves a whole think-time sweep.
Turns within a session always keep their dependency.

`--template-check` (default on when the manifest has a `probe`) sends the probe conversation once with
`max_tokens: 1` and refuses to run if the server's `usage.prompt_tokens` differs from the count prepared offline:
the server renders a different chat template or thinking mode than the file was prepared with.

YAML:

```yaml
workload: {type: session-replay, path: chat.jsonl, concurrent_sessions: 8, think_median_s: 10}
workload_type: {type: openai, url: http://host:8000/v1/chat/completions, model: m, stream: true,
                passthrough_meta: true}
load: closed:8
```

The workload's completion hook (which releases each session's next turn) is installed automatically. Rows with
`prompt_ids` instead of `messages` need a completions workload type and are refused.
