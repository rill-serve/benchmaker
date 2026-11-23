"""Runtime capability capture, void rules and the run manifest (issue #4)."""

from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest
import pytest_asyncio
from aiohttp import web

from benchmaker.config import build_config
from benchmaker.core.capability import (
    OPS, VoidRule, check_rule, evaluate_void_rules, extract_model_identity,
    normalize_capability, parse_void_rules,
)
from benchmaker.core.runner import BenchRunner
from benchmaker.io.bundle import (CAPABILITY_END_FILENAME, CAPABILITY_START_FILENAME, MANIFEST_FILENAME,
                                  read_bundle, write_bundle)
from benchmaker.io.collect import collect_table, format_table
from benchmaker.io.manifest import build_manifest, canonical_json, config_hash, filter_env, job_identity
from benchmaker.core.types import Response
from benchmaker.workloads.llm import OpenAIChatWorkloadType


# --------------------------------------------------------------------------- #
# Snapshot helpers                                                             #
# --------------------------------------------------------------------------- #


def test_normalize_models_list_and_passthrough():
    doc = normalize_capability({"object": "list", "data": [{"id": "a"}, {"id": "b"}]})
    assert doc["models"] == ["a", "b"] and doc["models_detail"][0] == {"id": "a"}
    info = {"model": "x", "quantization": "fp8"}
    assert normalize_capability(info) is info
    assert normalize_capability(42) == {"value": 42}


def test_extract_model_identity_prefers_real_revision():
    snap = {"ok": True, "data": {"model": "llama", "model_revision": "main", "version": "0.9"}}
    ident = extract_model_identity(snap)
    assert ident["model"] == "llama" and ident["revision"] is None  # "main" is not a revision
    snap["data"]["model_revision"] = "9f2c1a7"
    ident = extract_model_identity(snap)
    assert ident == {"model": "llama", "revision": "9f2c1a7", "source": "capability:model_revision"}
    # a single-entry /v1/models is enough to name the model
    ident = extract_model_identity({"ok": True, "data": {"object": "list", "data": [{"id": "only-one"}]}})
    assert ident["model"] == "only-one" and ident["revision"] is None
    assert extract_model_identity({"ok": False, "error": "x"}) == {"model": None, "revision": None, "source": None}
    assert extract_model_identity(None)["model"] is None


def test_void_rule_validation_and_ops():
    assert OPS  # the documented op set
    r = VoidRule(name="q", path="quantization", op="eq", value="fp8")
    assert r.at == "end"
    with pytest.raises(ValueError):
        VoidRule(name="q", path="p", op="~~")
    with pytest.raises(ValueError):
        VoidRule(name="q", path="p", op="eq", at="middle")
    with pytest.raises(ValueError):
        VoidRule.from_dict({"name": "q", "path": "p", "bogus": 1})
    rules = parse_void_rules({"name": "single", "path": "a"})
    assert len(rules) == 1 and rules[0].op == "empty"
    with pytest.raises(ValueError):
        parse_void_rules([{"name": "dup", "path": "a"}, {"name": "dup", "path": "b"}])
    with pytest.raises(ValueError):
        parse_void_rules("nope")


def _snap(doc):
    return {"ok": True, "url": "u", "data": doc}


def test_check_rule_semantics():
    snap = _snap({"quantization": "fp8", "fallbacks": [], "dtype": "float16", "max_num_seqs": 32,
                  "models": ["a", "b"], "note": "kv cache fp8", "nested": {"x": [1, 2, 3]}})
    assert check_rule(VoidRule("q", "quantization", "eq", "fp8"), snap) is None
    v = check_rule(VoidRule("q", "quantization", "eq", "int4"), snap)
    assert v["actual"] == "fp8" and v["reason"] == "path missing" or v["rule"] == "q"
    assert check_rule(VoidRule("q", "quantization", "ne", "int4"), snap) is None
    assert check_rule(VoidRule("f", "fallbacks", "empty"), snap) is None
    assert check_rule(VoidRule("f", "models", "empty"), snap) is not None
    assert check_rule(VoidRule("m", "max_num_seqs", "ge", 32), snap) is None
    assert check_rule(VoidRule("m", "max_num_seqs", "gt", 32), snap) is not None
    assert check_rule(VoidRule("m", "models", "contains", "b"), snap) is None
    assert check_rule(VoidRule("m", "models.0", "eq", "a"), snap) is None
    assert check_rule(VoidRule("m", "nested.x.2", "eq", 3), snap) is None
    assert check_rule(VoidRule("m", "models", "in", ["a", "b"]), snap) is None
    assert check_rule(VoidRule("m", "quantization", "not_in", ["int4"]), snap) is None
    assert check_rule(VoidRule("m", "note", "not_contains", "int4"), snap) is None
    # missing path: violation unless ignored; numeric compare against a string is a violation
    assert check_rule(VoidRule("x", "nope", "eq", 1), snap)["reason"] == "path missing"
    assert check_rule(VoidRule("x", "nope", "eq", 1, ignore_missing=True), snap) is None
    assert check_rule(VoidRule("x", "quantization", "lt", 5), snap)["reason"].startswith("ValueError")
    # unavailable snapshot -> violation (a run whose capability cannot be read is not trusted)
    assert check_rule(VoidRule("x", "quantization", "eq", "fp8"), None)["reason"] == "no snapshot"
    assert "unavailable" in check_rule(VoidRule("x", "a", "empty"), {"ok": False, "error": "HTTP 500"})["reason"]


def test_evaluate_void_rules_at_start_end_both():
    start = _snap({"quantization": "fp8", "fallbacks": []})
    end = _snap({"quantization": "fp8", "fallbacks": ["marlin"]})
    rules = [
        VoidRule("q", "quantization", "eq", "fp8", at="both"),
        VoidRule("fb", "fallbacks", "empty", at="end"),
        VoidRule("fb_start", "fallbacks", "empty", at="start"),
    ]
    verdict = evaluate_void_rules(rules, start, end)
    assert verdict.void is True
    assert [v["rule"] for v in verdict.violations] == ["fb"]
    assert verdict.violations[0]["at"] == "end" and verdict.violations[0]["actual"] == ["marlin"]
    assert {(c["rule"], c["at"]) for c in verdict.checks} == {
        ("q", "start"), ("q", "end"), ("fb", "end"), ("fb_start", "start")}
    d = verdict.to_dict()
    assert d["void"] and len(d["checks"]) == 4
    assert evaluate_void_rules(rules[:1], start, end).void is False
    assert evaluate_void_rules([], start, end).void is False


# --------------------------------------------------------------------------- #
# Manifest                                                                     #
# --------------------------------------------------------------------------- #


def test_config_hash_is_order_insensitive_and_stable():
    a = config_hash({"workload_type": {"model": "m", "type": "openai"}, "duration": 10})
    b = config_hash({"duration": 10, "workload_type": {"type": "openai", "model": "m"}})
    assert a == b and a.startswith("sha256:") and len(a) == 7 + 64
    assert config_hash({"duration": 11}) != a
    assert canonical_json({"b": 1, "a": [2, 1.5]}) == '{"a":[2,1.5],"b":1}'


def test_filter_env_globs_and_job_identity(monkeypatch):
    env = {"CUDA_VISIBLE_DEVICES": "0,1", "VLLM_USE_V1": "1", "SGLANG_X": "y", "NCCL_IB_HCA": "h",
           "SECRET_KEY": "s", "PATH": "/bin"}
    out = filter_env(["CUDA_VISIBLE_DEVICES", "VLLM_*", "NCCL_*IB*"], env)
    assert out == {"CUDA_VISIBLE_DEVICES": "0,1", "VLLM_USE_V1": "1", "NCCL_IB_HCA": "h"}
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setenv("SLURM_ARRAY_JOB_ID", "12")
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "3")
    ident = job_identity(env={"SLURM_ARRAY_JOB_ID": "12", "SLURM_ARRAY_TASK_ID": "3"})
    assert ident == {"job_id": "12_3", "source": "SLURM_ARRAY_JOB_ID+SLURM_ARRAY_TASK_ID"}
    assert job_identity(env={}) == {"job_id": socket.gethostname(), "source": "hostname"}


class _WT:
    model = "stub-model"


class _WL:
    name = "static"

    def content_hash(self) -> str:
        return "deadbeef"

    def item_count(self) -> int:
        return 7


def test_build_manifest_fields(monkeypatch):
    monkeypatch.setenv("BENCHMAKER_TEST_KNOB", "on")
    cap = {"start": {"ok": True, "data": {"model": "stub-model", "model_revision": "main"}},
           "end": {"ok": True, "data": {"model": "stub-model", "model_revision": "1a2b3c4d"}}}
    m = build_manifest(
        source_config={"a": 1}, run_id="r1", workload_type=_WT(), workload=_WL(), workload_name="static",
        labels={"arm": "a"}, started_at=1_700_000_000.0, capability=cap,
        env={"BENCHMAKER_TEST_KNOB": "on"}, record_env=["NOPE_*"], benchmaker_version="0.1",
    )
    assert m["manifest_version"] == 1 and m["benchmaker_version"] == "0.1" and m["run_id"] == "r1"
    assert m["config_hash"] == config_hash({"a": 1})
    assert m["prompt_set"] == {"name": "static", "hash": "deadbeef", "items": 7}
    assert m["model"] == {"id": "stub-model", "revision": "1a2b3c4d", "revision_source": "capability:model_revision",
                          "capability_model": "stub-model"}
    assert m["host"]["hostname"] and m["started_at"].startswith("2023-11-14")
    assert m["env"] == {"BENCHMAKER_TEST_KNOB": "on"} and m["labels"] == {"arm": "a"}
    # explicit revision beats the snapshot; placeholder revisions are not recorded
    m2 = build_manifest(source_config={}, run_id="r", model_revision="v1.2", capability=cap)
    assert m2["model"]["revision"] == "v1.2" and m2["model"]["revision_source"] == "config"
    m3 = build_manifest(source_config={}, run_id="r", capability={"end": cap["start"]})
    assert m3["model"]["revision"] is None and m3["model"]["capability_model"] == "stub-model"
    # a workload without content_hash still produces the block
    m4 = build_manifest(source_config={}, run_id="r", workload=object(), workload_name="x")
    assert m4["prompt_set"] == {"name": "x"}


# --------------------------------------------------------------------------- #
# Runner integration: snapshots, void verdict, bundle contents, collect         #
# --------------------------------------------------------------------------- #


class _Info:
    def __init__(self) -> None:
        self.doc = {"model": "stub", "quantization": "fp8", "fallbacks": [], "dtype": "bfloat16"}


async def _chat(request: web.Request) -> web.Response:
    body = await request.json()
    return web.json_response({"choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                              "usage": {"prompt_tokens": 5, "completion_tokens": 1}, "model": body.get("model", "stub")})


@pytest_asyncio.fixture
async def cap_server():
    info = _Info()
    hits: list[str] = []

    async def server_info(request: web.Request) -> web.Response:
        hits.append(request.path)
        return web.json_response(info.doc)

    async def models(request: web.Request) -> web.Response:
        return web.json_response({"object": "list", "data": [{"id": "stub"}]})

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    app = web.Application()
    app.router.add_post("/v1/chat/completions", _chat)
    app.router.add_get("/server_info", server_info)
    app.router.add_get("/v1/models", models)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", port)
    await site.start()
    try:
        yield f"http://127.0.0.1:{port}", info, hits
    finally:
        await runner.cleanup()


def _cfg(base: str, **extra) -> dict:
    return {
        "workload_type": {"type": "openai", "url": f"{base}/v1/chat/completions", "model": "stub", "stream": False},
        "load": "50", "duration": "0.2s",
        "capability": {"url": f"{base}/server_info"},
        **extra,
    }


@pytest.mark.asyncio
async def test_runner_captures_snapshots_and_evaluates_void_rules(cap_server, tmp_path):
    base, info, hits = cap_server
    rules = [
        {"name": "quant", "path": "quantization", "op": "eq", "value": "fp8", "at": "both"},
        {"name": "no_fallback", "path": "fallbacks", "op": "empty", "at": "both"},
    ]
    cfg = _cfg(base, void_rules=rules)
    runner = BenchRunner(build_config(cfg, dotenv_path=None))
    result = await runner.run()
    assert hits == ["/server_info", "/server_info"]
    cap = result.capability
    assert cap["start"]["at"] == "start" and cap["end"]["at"] == "end"
    assert cap["start"]["data"] == info.doc
    assert result.summary["void"] == {"void": False, "violations": [], "checks": [
        {"rule": "quant", "at": "start", "path": "quantization", "op": "eq", "expected": "fp8", "ok": True},
        {"rule": "quant", "at": "end", "path": "quantization", "op": "eq", "expected": "fp8", "ok": True},
        {"rule": "no_fallback", "at": "start", "path": "fallbacks", "op": "empty", "expected": None, "ok": True},
        {"rule": "no_fallback", "at": "end", "path": "fallbacks", "op": "empty", "expected": None, "ok": True},
    ]}
    out = runner.write_bundle(str(tmp_path / "runs"), run_id="ok", source_config=cfg, labels={"arm": "a"})
    assert Path(out, CAPABILITY_START_FILENAME).is_file() and Path(out, CAPABILITY_END_FILENAME).is_file()
    bundle = read_bundle(out)
    assert bundle["capability"]["start"]["ok"] and bundle["meta"]["void"] is False
    assert bundle["summary"]["void"]["void"] is False and bundle["summary"]["capability"]["end"]["ok"]
    man = bundle["manifest"]
    assert man["model"] == {"id": "stub", "revision": None, "revision_source": None, "capability_model": "stub"}
    assert man["config_hash"] == config_hash(cfg) and man["labels"] == {"arm": "a"}
    assert man["prompt_set"]["name"] == "static"

    # Now the server falls back mid-run: the end snapshot shows a fallback and
    # the run is void. meta.json carries it so `collect` can flag the row.
    info.doc = {**info.doc, "fallbacks": ["marlin"]}
    runner = BenchRunner(build_config(_cfg(base, void_rules=rules), dotenv_path=None))
    result = await runner.run()
    v = result.summary["void"]
    assert v["void"] is True and [x["rule"] for x in v["violations"]] == ["no_fallback", "no_fallback"]
    assert [x["at"] for x in v["violations"]] == ["start", "end"]  # rule is checked at both snapshots
    assert v["violations"][0]["actual"] == ["marlin"]
    assert result.summary["capability"]["start"]["data"]["fallbacks"] == ["marlin"]
    out = runner.write_bundle(str(tmp_path / "runs"), run_id="void", source_config=cfg)
    meta = json.loads(Path(out, "meta.json").read_text())
    assert meta["void"] is True and [x["rule"] for x in meta["void_violations"]] == ["no_fallback", "no_fallback"]
    assert json.loads(Path(out, "summary.json").read_text())["void"]["void"] is True


@pytest.mark.asyncio
async def test_capability_failure_is_recorded_not_fatal(cap_server, tmp_path):
    base, info, hits = cap_server
    cfg = _cfg(base, void_rules=[{"name": "q", "path": "quantization", "op": "eq", "value": "fp8"}])
    cfg["capability"] = {"url": f"{base}/missing"}  # 404
    runner = BenchRunner(build_config(cfg, dotenv_path=None))
    result = await runner.run()
    assert result.capability["start"]["ok"] is False and "404" in result.capability["start"]["error"]
    # an unreadable snapshot cannot vouch for the run -> rule violated
    assert result.summary["void"]["void"] is True
    assert result.summary["void"]["violations"][0]["reason"].startswith("snapshot unavailable")
    out = runner.write_bundle(str(tmp_path / "runs"), run_id="r")
    assert read_bundle(out)["capability"]["end"]["ok"] is False
    # without void rules, a failed snapshot is just recorded
    cfg["void_rules"] = []
    runner = BenchRunner(build_config(cfg, dotenv_path=None))
    result = await runner.run()
    assert result.summary["void"]["void"] is False


@pytest.mark.asyncio
async def test_no_capability_config_means_no_snapshots(cap_server):
    base, info, hits = cap_server
    cfg = _cfg(base)
    del cfg["capability"]
    runner = BenchRunner(build_config(cfg, dotenv_path=None))
    result = await runner.run()
    assert result.capability is None and hits == []
    assert "void" not in result.summary and "capability" not in result.summary


@pytest.mark.asyncio
async def test_collect_exposes_manifest_and_void(cap_server, tmp_path):
    base, info, hits = cap_server
    cfg = _cfg(base, void_rules=[{"name": "no_fallback", "path": "fallbacks", "op": "empty"}])
    runner = BenchRunner(build_config(cfg, dotenv_path=None))
    await runner.run()
    good = runner.write_bundle(str(tmp_path / "runs"), run_id="good", source_config=cfg, labels={"arm": "good"})
    info.doc = {**info.doc, "fallbacks": ["marlin"]}
    runner = BenchRunner(build_config(cfg, dotenv_path=None))
    await runner.run()
    bad = runner.write_bundle(str(tmp_path / "runs"), run_id="bad", source_config=cfg, labels={"arm": "bad"})
    rows, cols = collect_table(
        [good, bad],
        extra_metrics=["manifest.model.id", "manifest.config_hash", "capability.end.data.quantization",
                       "summary.void.void", "ttft_s.mean"],
    )
    assert rows[0]["manifest.model.id"] == "stub" and rows[0]["manifest.config_hash"] == config_hash(cfg)
    assert rows[0]["capability.end.data.quantization"] == "fp8" and rows[0]["summary.void.void"] is False
    assert rows[1]["summary.void.void"] is True
    assert "void" in cols and rows[0]["void"] is False and rows[1]["void"] is True
    text = format_table(rows, ["run_id", "void"], "md")
    assert text.count("VOID") == 1 and "good" in text and "bad" in text
    csv_text = format_table(rows, ["run_id", "void"], "csv")
    assert "good,false" in csv_text and "bad,true" in csv_text


def test_read_bundle_tolerates_missing_manifest(tmp_path):
    from benchmaker.core.metrics import MetricsAggregator
    from benchmaker.core.types import Sample
    agg = MetricsAggregator()
    agg.add(Sample(start_ts=0.0, latency_s=0.1, status=200, ok=True))
    agg.finalize()
    d = write_bundle(str(tmp_path), agg, workload_type_name="http", workload_name="static", run_id="old")
    assert not Path(d, MANIFEST_FILENAME).exists()  # pre-#4 bundle: no manifest, no snapshots
    b = read_bundle(d)
    assert b["manifest"] is None and b["capability"] == {"start": None, "end": None} and b["meta"]["void"] is None
    rows, _ = collect_table([d], extra_metrics=["manifest.model.id", "summary.void.void"])
    assert rows[0]["manifest.model.id"] is None and rows[0]["void"] is None


@pytest.mark.asyncio
async def test_model_revision_and_record_env_from_config(cap_server, monkeypatch):
    base, info, hits = cap_server
    monkeypatch.setenv("MY_SERVER_KNOB", "on")
    cfg = _cfg(base, model_revision="v1.2.3", record_env=["MY_SERVER_*"])
    bc = build_config(cfg, dotenv_path=None)
    assert bc.model_revision == "v1.2.3" and bc.record_env == ["MY_SERVER_*"]
    runner = BenchRunner(bc)
    await runner.run()
    out = runner.write_bundle(str("./_tmp_runs_rev"), run_id="rev")
    try:
        man = read_bundle(out)["manifest"]
        assert man["model"]["revision"] == "v1.2.3" and man["model"]["revision_source"] == "config"
        assert man["env"]["MY_SERVER_KNOB"] == "on"
        # explicit kwargs still win over the config
        out2 = runner.write_bundle(str("./_tmp_runs_rev"), run_id="rev2", model_revision="v9")
        assert read_bundle(out2)["manifest"]["model"]["revision"] == "v9"
    finally:
        import shutil
        shutil.rmtree("./_tmp_runs_rev", ignore_errors=True)
    assert runner.is_void() is False


def test_cli_run_exits_3_on_void_run(tmp_path, capsys):
    from click.testing import CliRunner
    from benchmaker.cli import main

    cfg = {
        "workload_type": {"type": "http", "url": "http://127.0.0.1:9/nope", "method": "GET"},
        "load": "1", "max_requests": 1,
        "capability": {"url": "http://127.0.0.1:9/missing", "timeout_s": 1},
        "void_rules": [{"name": "up", "path": "model", "op": "ne", "value": None}],
    }
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text(json.dumps(cfg))
    res = CliRunner().invoke(main, ["run", str(yaml_path), "--out-dir", str(tmp_path / "runs"),
                                    "--dotenv", "", "--quiet"])
    assert res.exit_code == 3, res.output
    assert "VOID" in res.stderr or "VOID" in res.output
    bundles = list((tmp_path / "runs").iterdir())
    assert len(bundles) == 1 and (bundles[0] / "summary.json").is_file()
    assert json.loads((bundles[0] / "summary.json").read_text())["void"]["void"] is True


# --------------------------------------------------------------------------- #
# TTFT token parity (issue #4)                                                 #
# --------------------------------------------------------------------------- #


def _sse(evs):
    raw = b"".join((e + "\n\n").encode() for e, _ in evs)
    chunks = [(e + "\n\n").encode() for e, _ in evs]
    times = [t for _, t in evs]
    return Response(status=200, headers={}, body=raw, elapsed_s=times[-1], ok=True,
                    stream_chunks=chunks, stream_chunk_times=times)


async def test_ttft_index_points_at_reasoning_token_when_configured_any():
    evs = [
        ('data: ' + json.dumps({"choices": [{"index": 0, "delta": {"reasoning_content": "hmm"}}]}), 0.050),
        ('data: ' + json.dumps({"choices": [{"index": 0, "delta": {"reasoning_content": "more"}}]}), 0.060),
        ('data: ' + json.dumps({"choices": [{"index": 0, "delta": {"content": "a"}}]}), 0.200),
        ('data: ' + json.dumps({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}), 0.201),
        ('data: ' + json.dumps({"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 3}}), 0.202),
        ('data: [DONE]', 0.203),
    ]
    wt = OpenAIChatWorkloadType(url="http://x", model="m")
    s_any = await wt.make_sample("hi", await wt.make_request("hi"), _sse(evs), 0.0)
    assert s_any.extra["ttft_token_index"] == 0.0 and s_any.meta["ttft_token_kind"] == "reasoning"
    wt2 = OpenAIChatWorkloadType(url="http://x", model="m", ttft_token="content")
    s_content = await wt2.make_sample("hi", await wt2.make_request("hi"), _sse(evs), 0.0)
    assert s_content.extra["ttft_token_index"] == 2.0 and s_content.meta["ttft_token_kind"] == "content"
    assert s_content.extra["ttft_s"] == pytest.approx(0.200)
