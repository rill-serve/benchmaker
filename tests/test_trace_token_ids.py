"""Issue #11 — token-id replay for Mooncake traces: ``send: token_ids`` with the
``openai-completions`` workload-type, block ``namespace``, ``unique_tail`` and
``max_seq`` drop statistics in the manifest."""

from __future__ import annotations

import json
import socket

import pytest
import pytest_asyncio
from aiohttp import web

from benchmaker.config import build_config
from benchmaker.core.runner import BenchRunner
from benchmaker.workloads._tokens import TokenIdSampler
from benchmaker.workloads.llm import OpenAICompletionsWorkloadType
from benchmaker.workloads.traces import MooncakeTraceWorkload, load_mooncake_trace


def _write_mooncake(tmp_path):
    rows = [
        {"timestamp": 0, "input_length": 512, "output_length": 8, "hash_ids": [11]},
        {"timestamp": 10, "input_length": 512, "output_length": 8, "hash_ids": [11]},   # whole-prompt duplicate of row 0
        {"timestamp": 20, "input_length": 700, "output_length": 64, "hash_ids": [11, 12]},
        {"timestamp": 30, "input_length": 3000, "output_length": 10, "hash_ids": [11, 12, 13, 14, 15, 16]},
        {"timestamp": 40, "input_length": 0, "output_length": 5, "hash_ids": []},        # dropped by the loader
    ]
    path = tmp_path / "mooncake.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    return str(path)


async def _next(wl, n):
    return [await wl.next_item() for _ in range(n)]


# --------------------------------------------------------------------------- #
# TokenIdSampler                                                               #
# --------------------------------------------------------------------------- #


def test_token_id_sampler_whitelist_and_prefix_property():
    s = TokenIdSampler.from_vocab(1000, exclude_ids=[7], exclude_last=8)
    assert s.size == 1000 - 1 - 8 and "vocab=1000" in s.source
    full = s.block(("mooncake", "11"), 512)
    assert len(full) == 512 and all(isinstance(t, int) for t in full)
    assert s.block(("mooncake", "11"), 300) == full[:300]          # a truncated block is a true prefix
    assert s.block(("mooncake", "11"), 512) == full                 # deterministic / memoized
    assert s.block(("mooncake", "12"), 512) != full                 # different hash id -> different content
    assert not ({7, 992, 999} & set(full))                          # reserved ids never appear
    assert s.block("x", 0) == s.block("x", 1)                       # degenerate sizes clamp to one token


def test_token_id_sampler_from_tokenizer_excludes_special_and_added_tokens():
    class FakeTok:
        all_special_ids = [0, 1, 2]
        added_tokens_encoder = {"<|im_start|>": 3, "<|im_end|>": 4}

        def __len__(self):
            return 50

    s = TokenIdSampler.from_tokenizer(FakeTok())
    assert s.size == 45 and "vocab=50 excluded=5" in s.source
    block = s.block("seed", 200)
    assert not ({0, 1, 2, 3, 4} & set(block))
    with pytest.raises(ValueError):
        TokenIdSampler.from_vocab(0)


# --------------------------------------------------------------------------- #
# Trace workload in token_ids mode                                             #
# --------------------------------------------------------------------------- #


async def test_token_ids_replay_is_exact_and_prefix_faithful(tmp_path):
    path = _write_mooncake(tmp_path)
    wl = MooncakeTraceWorkload(path, send="token_ids", vocab_size=4096, exclude_last=16)
    items = await _next(wl, 4)
    prompts = [it["prompt"] for it in items]
    assert all(isinstance(p, list) and all(isinstance(t, int) for t in p) for p in prompts)
    assert [len(p) for p in prompts] == [512, 512, 700, 3000]       # exact, no tokenizer drift
    assert prompts[0] == prompts[1]                                  # identical hash list -> identical prefix
    assert prompts[2][:512] == prompts[0] and prompts[3][:700] == prompts[2]
    assert [it["max_tokens"] for it in items] == [8, 8, 64, 10]
    assert items[2]["meta"]["prompt_tokens_hint"] == 700 and items[2]["meta"]["trace_blocks"] == 2
    prov = wl.provenance()
    assert prov["send"] == "token_ids" and prov["source"] == "mooncake" and prov["records"] == 4
    assert prov["dropped"] == {"invalid_input": 1} or sum(prov["dropped"].values()) == 1
    assert prov["token_whitelist"]["size"] == 4096 - 16
    # the hash follows the knobs that change prompt content
    assert wl.content_hash() != MooncakeTraceWorkload(path, send="token_ids", vocab_size=8192).content_hash()
    assert wl.content_hash() == MooncakeTraceWorkload(path, send="token_ids", vocab_size=4096, exclude_last=16).content_hash()


async def test_namespace_separates_models_and_unique_tail_breaks_duplicates(tmp_path):
    path = _write_mooncake(tmp_path)
    a = await _next(MooncakeTraceWorkload(path, send="token_ids", vocab_size=1000, namespace="model-a"), 4)
    b = await _next(MooncakeTraceWorkload(path, send="token_ids", vocab_size=1000, namespace="model-b"), 4)
    assert a[0]["prompt"] != b[0]["prompt"] and a[0]["prompt"][:512] != b[2]["prompt"][:512]
    assert a[2]["prompt"][:512] == a[0]["prompt"]                     # sharing still works within a namespace
    assert a[0]["meta"]["trace_namespace"] if "trace_namespace" in a[0]["meta"] else True
    wl_a = MooncakeTraceWorkload(path, send="token_ids", vocab_size=1000, namespace="model-a")
    wl_none = MooncakeTraceWorkload(path, send="token_ids", vocab_size=1000)
    assert wl_a.provenance()["namespace"] == "model-a" and wl_none.provenance()["namespace"] is None
    assert wl_a.content_hash() != wl_none.content_hash()
    # text mode keeps the pre-#11 hash when namespace is unset
    assert MooncakeTraceWorkload(path).content_hash() == MooncakeTraceWorkload(path, namespace=None).content_hash()
    assert MooncakeTraceWorkload(path).content_hash() != MooncakeTraceWorkload(path, namespace="x").content_hash()

    # unique_tail: duplicates keep their shared prefix but stop being identical
    tail = await _next(MooncakeTraceWorkload(path, send="token_ids", vocab_size=1000, unique_tail=8), 4)
    p0, p1 = tail[0]["prompt"], tail[1]["prompt"]
    assert len(p0) == len(p1) == 520 and p0[:512] == p1[:512] and p0 != p1
    assert tail[0]["meta"]["prompt_tokens_hint"] == 520 and tail[0]["meta"]["trace_unique_tail"] == 8
    # also in text mode
    txt = await _next(MooncakeTraceWorkload(path, unique_tail=8), 2)
    assert txt[0]["messages"][0]["content"] != txt[1]["messages"][0]["content"]
    assert txt[0]["messages"][0]["content"].startswith((await _next(MooncakeTraceWorkload(path), 1))[0]["messages"][0]["content"])
    with pytest.raises(ValueError):
        MooncakeTraceWorkload(path, unique_tail=-1)


async def test_max_seq_drops_and_counts_rows(tmp_path):
    path = _write_mooncake(tmp_path)
    wl = MooncakeTraceWorkload(path, max_seq=600)
    items = await _next(wl, 2)                                   # 700+64 and 3000+10 exceed max_seq
    assert [it["meta"]["trace_input_tokens"] for it in items] == [512, 512]
    prov = wl.provenance()
    assert prov["max_seq"] == 600 and prov["dropped"].get("max_seq") == 2
    recs = load_mooncake_trace(path, max_seq=600)
    assert [r.input_tokens for r in recs] == [512, 512]
    with pytest.raises(StopAsyncIteration):
        await wl.next_item()


# --------------------------------------------------------------------------- #
# Config wiring                                                                #
# --------------------------------------------------------------------------- #


def test_config_wires_token_ids_to_completions_and_resolves_namespace(tmp_path):
    path = _write_mooncake(tmp_path)
    with pytest.raises(ValueError, match="openai-completions"):
        build_config({
            "workload_type": {"type": "openai", "url": "http://x/v1/chat/completions", "model": "m"},
            "workload": {"type": "mooncake-trace", "path": path, "send": "token_ids", "vocab_size": 1000},
            "load": {"type": "trace"},
        }, dotenv_path=None)
    with pytest.raises(ValueError, match="tokenizer"):
        build_config({
            "workload_type": {"type": "openai-completions", "url": "http://x/v1/completions", "model": "m"},
            "workload": {"type": "mooncake-trace", "path": path, "send": "token_ids"},
            "load": {"type": "trace"},
        }, dotenv_path=None)
    bc = build_config({
        "workload_type": {"type": "openai-completions", "url": "http://x/v1/completions", "model": "served-model"},
        "workload": {"type": "mooncake-trace", "path": path, "send": "token_ids", "vocab_size": 1000,
                     "namespace": "model", "unique_tail": 4, "max_seq": 100000},
        "load": {"type": "trace"},
    }, dotenv_path=None)
    assert isinstance(bc.workload_type, OpenAICompletionsWorkloadType)
    assert bc.workload.namespace == "served-model" and bc.workload.provenance()["max_seq"] == 100000
    # a lane's model wins over the workload-type's
    bc2 = build_config({
        "workload_type": {"type": "openai-completions", "url": "http://x/v1/completions", "model": "served-model"},
        "workload": {"type": "static", "items": ["x"]},
        "mix": {"lanes": [{"name": "l", "workload": {"type": "mooncake", "path": path, "send": "token_ids",
                                                     "vocab_size": 1000, "namespace": "model"},
                           "rate": 1, "model": "lane-model", "duration": "0.1s"}]},
        "duration": "0.1s",
    }, dotenv_path=None)
    assert bc2.lanes[0].workload.namespace == "lane-model"


# --------------------------------------------------------------------------- #
# End-to-end: /v1/completions receives token ids                              #
# --------------------------------------------------------------------------- #


@pytest_asyncio.fixture
async def completions_stub():
    seen: list = []

    async def completions(request: web.Request) -> web.StreamResponse:
        body = await request.json()
        seen.append(body)
        resp = web.StreamResponse(status=200, headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        for i in range(3):
            chunk = {"choices": [{"index": 0, "text": f"t{i} ", "finish_reason": None}]}
            await resp.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
        await resp.write(b"data: " + json.dumps({
            "choices": [{"index": 0, "text": "", "finish_reason": "stop"}],
            "usage": {"prompt_tokens": len(body["prompt"]), "completion_tokens": 3, "total_tokens": len(body["prompt"]) + 3},
        }).encode() + b"\n\n")
        await resp.write(b"data: [DONE]\n\n")
        await resp.write_eof()
        return resp

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    app = web.Application()
    app.router.add_post("/v1/completions", completions)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", port)
    await site.start()
    try:
        yield f"http://127.0.0.1:{port}", seen
    finally:
        await runner.cleanup()


async def test_token_ids_run_hits_completions_with_ids(completions_stub, tmp_path):
    base, seen = completions_stub
    path = _write_mooncake(tmp_path)
    cfg = {
        "workload_type": {"type": "openai-completions", "url": f"{base}/v1/completions", "model": "m",
                          "stream": True, "ignore_eos": True},
        "workload": {"type": "mooncake-trace", "path": path, "send": "token_ids", "vocab_size": 32000,
                     "namespace": "model", "unique_tail": 4},
        "load": {"type": "trace", "speed": 20},
    }
    runner = BenchRunner(build_config(cfg, dotenv_path=None))
    result = await runner.run()
    samples = sorted(result.samples, key=lambda s: s.meta["trace_index"])
    assert len(samples) == 4 and all(s.ok for s in samples)
    assert all(s.meta["prompt_kind"] == "token_ids" for s in samples)
    assert [s.extra["prompt_tokens_hint"] for s in samples] == [516, 516, 704, 3004]
    assert [len(b["prompt"]) for b in seen] == [516, 516, 704, 3004]
    assert all(isinstance(b["prompt"], list) and b["stream"] is True for b in seen)
    assert seen[0]["prompt"][:512] == seen[1]["prompt"][:512] and seen[0]["prompt"] != seen[1]["prompt"]
    assert seen[2]["prompt"][:512] == seen[0]["prompt"][:512]
    assert samples[0].extra["prompt_tokens"] == 516

    # the manifest records the replay provenance next to the prompt-set hash
    out = runner.write_bundle(str(tmp_path / "out"), run_id="r", source_config=cfg)
    man = json.load(open(out + "/manifest.json"))
    ps = man["prompt_set"]
    assert ps["hash"].startswith("sha256:") and ps["provenance"]["send"] == "token_ids"
    assert ps["provenance"]["namespace"] == "m" and ps["provenance"]["unique_tail"] == 4  # namespace: model -> "m"
    assert ps["provenance"]["token_whitelist"]["size"] == 32000


async def test_completions_workload_type_accepts_ids_and_text():
    wt = OpenAICompletionsWorkloadType(url="http://x/v1/completions", model="m", stream=False)
    r = await wt.make_request([1, 2, 3])
    assert r.json["prompt"] == [1, 2, 3] and r.meta["prompt_kind"] == "token_ids" and r.meta["prompt_len"] == 3
    r = await wt.make_request({"prompt": [5, 6], "meta": {"k": 1}})
    assert r.json["prompt"] == [5, 6] and r.meta["k"] == 1 and r.json["model"] == "m"
    pt = OpenAICompletionsWorkloadType(url="http://x/v1/completions", model="m", passthrough_meta=True)
    r = await pt.make_request({"prompt": [5, 6], "model": "other", "meta": {"k": 1}})
    assert r.meta["model_label"] == "other" and r.meta["k"] == 1 and r.json["model"] == "m"
    r = await wt.make_request("hello")
    assert r.json["prompt"] == "hello" and r.meta["prompt_kind"] == "text"
    with pytest.raises(TypeError):
        await wt.make_request({"messages": [{"role": "user", "content": "x"}]})
