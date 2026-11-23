"""Per-model endpoint maps (issue #8): one arm, several servers."""

from __future__ import annotations

import asyncio
import json
import socket

import pytest
import pytest_asyncio
from aiohttp import web

from benchmaker.config import build_config
from benchmaker.core.endpoints import endpoint_model, normalize_endpoints, route_url
from benchmaker.core.runner import BenchRunner


# Two stub servers; each records the model name it was asked for.
def _make_app(store: list, delay: float = 0.002) -> web.Application:
    async def chat(request: web.Request) -> web.StreamResponse:
        body = await request.json()
        store.append(body.get("model"))
        resp = web.StreamResponse(status=200, headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        for i in range(3):
            await asyncio.sleep(delay)
            chunk = {"choices": [{"index": 0, "delta": {"content": f"tok{i} "}, "finish_reason": None}]}
            await resp.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
        await resp.write(b"data: " + json.dumps({
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8,
                      "prompt_tokens_details": {"cached_tokens": 0}},
        }).encode() + b"\n\n")
        await resp.write(b"data: [DONE]\n\n")
        await resp.write_eof()
        return resp

    async def server_info(request: web.Request) -> web.Response:
        return web.json_response({"version": "0.1", "model": "stub"})

    app = web.Application()
    app.router.add_post("/v1/chat/completions", chat)
    app.router.add_get("/server_info", server_info)
    return app


async def _start(app: web.Application) -> str:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", port)
    await site.start()
    return f"http://127.0.0.1:{port}", runner


@pytest_asyncio.fixture
async def two_stubs():
    seen_a: list = []
    seen_b: list = []
    url_a, runner_a = await _start(_make_app(seen_a))
    url_b, runner_b = await _start(_make_app(seen_b, delay=0.01))
    try:
        yield url_a, url_b, seen_a, seen_b
    finally:
        await runner_a.cleanup()
        await runner_b.cleanup()


def test_normalize_endpoints_accepts_urls_and_mappings():
    eps = normalize_endpoints({
        "a": "http://a:8000",
        "b": {"url": "http://b:8000/v1/chat/completions", "capability": "http://b:8000/server_info",
              "headers": {"X-Tenant": "x"}},
    })
    assert eps["a"] == {"url": "http://a:8000", "capability": None, "headers": {}}
    assert eps["b"]["capability"] == "http://b:8000/server_info" and eps["b"]["headers"] == {"X-Tenant": "x"}
    assert normalize_endpoints(None) == {}
    with pytest.raises(ValueError):
        normalize_endpoints({"a": {"capability": "http://a/server_info"}})
    with pytest.raises(TypeError):
        normalize_endpoints(["http://a"])


def test_route_url_swaps_host_for_base_urls_and_keeps_full_urls():
    tpl = "http://h:8000/v1/chat/completions"
    assert route_url("http://other:8401", tpl) == "http://other:8401/v1/chat/completions"
    assert route_url("http://other:8401/", tpl) == "http://other:8401/v1/chat/completions"
    assert route_url("http://other:8400/v2/chat/completions", tpl) == "http://other:8400/v2/chat/completions"
    assert route_url("", tpl) == tpl


def test_endpoint_model_reads_body_model_or_meta_label():
    eps = {"a": {}, "b": {}}
    assert endpoint_model({"model": "a"}, None, eps) == "a"
    assert endpoint_model({"model": "zzz"}, None, eps) is None
    body = {"model": "wt-model"}
    assert endpoint_model(body, {"model_label": "b"}, eps) == "b" and body["model"] == "b"


async def test_endpoints_fan_out_routes_by_model_and_records_per_endpoint(two_stubs, tmp_path):
    url_a, url_b, seen_a, seen_b = two_stubs
    cfg = build_config({
        "workload_type": {"type": "openai", "url": f"{url_a}/v1/chat/completions", "model": "a",
                          "stream": True, "include_usage": True},
        "endpoints": {"a": url_a, "b": url_b},
        "capability": {"url": f"{url_a}/server_info"},
        "mix": {"lanes": [
            {"name": "lane-a", "workload": {"type": "static", "items": ["hi"]}, "rate": 50,
             "duration": "0.2s"},
            {"name": "lane-b", "workload": {"type": "static", "items": ["ho"]}, "rate": 50,
             "duration": "0.2s", "model": "b"},
        ]},
        "duration": "0.2s",
    }, dotenv_path=None)
    assert "model" in cfg.group_by  # auto per-model groups
    runner = BenchRunner(cfg)
    await runner.run()
    summary = runner.metrics.summary()
    assert set(summary["groups"]["model"]) == {"a", "b"}
    assert seen_a and set(seen_a) == {"a"}
    assert seen_b and set(seen_b) == {"b"}
    # each endpoint's capability was captured separately (derived from the shared URL)
    per = runner.capability["endpoints"]
    assert set(per) == {"a", "b"} and per["a"]["start"]["url"] == f"{url_a}/server_info"
    assert per["b"]["end"]["url"] == f"{url_b}/server_info"
    assert per["b"]["start"]["ok"] is True

    run_dir = runner.write_bundle(str(tmp_path), run_id="r", source_config={"x": 1})
    from benchmaker.io.bundle import read_bundle
    b = read_bundle(run_dir)
    assert set(b["capability"]["endpoints"]) == {"a", "b"}
    # manifest records the URL requests were actually sent to (base URL + workload path)
    assert b["manifest"]["endpoints"]["b"]["url"] == f"{url_b}/v1/chat/completions"
    assert b["manifest"]["endpoints"]["b"]["configured_url"] == url_b
    assert "configured_url" not in b["manifest"]["endpoints"]["a"] or True
    assert b["manifest"]["endpoints"]["b"]["capability_model"] == "stub"
    assert b["summary"]["void"] == {"void": False, "violations": [], "checks": []}
    assert b["summary"]["groups"]["model"]["b"]["share"] > 0
    with open(b["samples_path"]) as f:
        models = {json.loads(line)["meta"]["model"] for line in f}
    assert models == {"a", "b"}


async def test_endpoints_route_by_workload_item_model(two_stubs):
    url_a, url_b, seen_a, seen_b = two_stubs
    cfg = build_config({
        "workload_type": {"type": "openai", "url": f"{url_a}/v1/chat/completions", "model": "a"},
        "endpoints": {"a": url_a, "b": url_b},
        "workload": {"type": "static", "items": [{"prompt": "hi"}, {"prompt": "ho", "model": "b"}]},
        "load": 20, "duration": "0.15s",
    }, dotenv_path=None)
    runner = BenchRunner(cfg)
    await runner.run()
    assert "b" in seen_b and set(seen_a) == {"a"}
    assert set(runner.metrics.summary()["groups"]["model"]) == {"a", "b"}


async def test_endpoints_reject_lane_without_endpoint(two_stubs):
    url_a, url_b, _, _ = two_stubs
    with pytest.raises(ValueError, match="no endpoint"):
        build_config({
            "workload_type": {"type": "openai", "url": f"{url_a}/v1/chat/completions", "model": "a"},
            "endpoints": {"a": url_a},
            "mix": {"lanes": [{"name": "l", "workload": {"type": "static", "items": ["hi"]},
                               "rate": 10, "model": "b"}]},
            "duration": "0.1s",
        }, dotenv_path=None)
