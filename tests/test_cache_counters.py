"""Tests for host KV counter attribution and exact benchmark boundaries."""

import asyncio

import pytest
from aiohttp import web

from benchmaker.config import build_config
from benchmaker.core.cache_counters import counter_restores
from benchmaker.core.compare import RunRef, compare, render_comparison
from benchmaker.core.monitors import parse_prometheus
from benchmaker.core.runner import BenchRunner


def test_legacy_vllm_counts_only_loads_and_uses_one_metric_version():
    def scrape(load, modern, store, prompt):
        return parse_prometheus(
            f'vllm:kv_offload_total_bytes_total{{transfer_type="CPU_to_GPU"}} {load}\n'
            f'vllm:kv_offload_total_bytes_total{{transfer_type="GPU_to_CPU"}} {store}\n'
            f'vllm:kv_offload_load_bytes_total {modern}\n'
            f'vllm:prompt_tokens_total {prompt}\n'
        )

    report = counter_restores([scrape(100, 50, 10, 10), scrape(150, 100, 1000, 30)], bytes_per_token=5)
    assert report["attribution"] == "aggregate only"
    assert report["restored_bytes"] == 50
    assert report["restored_tokens"] == 10
    assert report["restored_token_share"] == 0.5
    assert report["counters"]["restored_bytes"] == "vllm:kv_offload_load_bytes_total"

    report = counter_restores([scrape(100, 50, 10, 10), scrape(150, 20, 1000, 30)])
    assert report["restored_bytes"] == 50
    assert report["restored_token_share"] is None
    assert report["counters"]["restored_bytes"] == 'vllm:kv_offload_total_bytes_total{transfer_type="CPU_to_GPU"}'


def test_sglang_counts_logical_tokens_on_one_rank_and_physical_bytes_on_all_ranks():
    def scrape(factor):
        return parse_prometheus("\n".join(
            "\n".join((
                f'sglang:prefill_effective_tokens_total{{tp_rank="{rank}",mode="host_hit"}} {3 * factor}',
                f'sglang:prefill_effective_tokens_total{{tp_rank="{rank}",mode="input"}} {7 * factor}',
                f'sglang:load_back_bytes_total{{tp_rank="{rank}"}} {50 * factor}',
            )) for rank in (0, 1)
        ))

    report = counter_restores([scrape(1), scrape(2)])
    assert report["restored_tokens"] == 3
    assert report["restored_token_share"] == 0.3
    assert report["restored_bytes"] == 100


def test_lmcache_only_uses_stable_series_and_preserves_measured_zero():
    hits = "lmcache:num_hit_tokens_total"
    prompt = "lmcache:num_prompt_tokens_total"
    assert counter_restores([{prompt: 10}, {hits: 5, prompt: 20}])["attribution"] == "unavailable"
    assert counter_restores([{hits: 4, prompt: 10}, {hits: 3, prompt: 20}])["attribution"] == "unavailable"
    report = counter_restores([{hits: 4, prompt: 10}, {hits: 4, prompt: 20}], bytes_per_token=20)
    assert report["restored_token_share"] == 0.0
    assert report["restored_bytes"] == 0


def test_lmcache_mp_keeps_lookup_hits_and_one_transfer_phase():
    snapshots = []
    for step in (1, 2):
        snapshots.append({
            "lmcache_mp_lookup_hit_tokens_total": 6 * step,
            "lmcache_mp_lookup_prompt_tokens_total": 10 * step,
            'lmcache_mp_transfer_phase_bytes_total{phase="staging",direction="h2d"}': 100 * step,
            'lmcache_mp_transfer_phase_bytes_total{phase="kernel",direction="h2d"}': 80 * step,
            'lmcache_mp_transfer_phase_bytes_total{phase="staging",direction="d2h"}': 20 * step,
        })
    report = counter_restores(snapshots)
    assert report["restored_token_share"] == 0.6
    assert report["restored_token_estimate"] == "lookup hits"
    assert report["restored_bytes"] == 100


def test_simple_vllm_can_convert_load_blocks_to_tokens_and_bytes():
    blocks = "vllm:simple_kv_offload_load_blocks_total"
    prompt = "vllm:prompt_tokens_total"
    report = counter_restores(
        [{blocks: 1, prompt: 80}, {blocks: 3, prompt: 144}],
        block_size_tokens=16, bytes_per_token=4096,
    )
    assert report["restored_blocks"] == 2
    assert report["restored_token_share"] == 0.5
    assert report["restored_bytes"] == 32 * 4096


def test_compare_preserves_aggregate_scope_and_missing_arm():
    run = RunRef("host", 0, ".", {"cache_restores": {
        "engine": {"restored_token_share": 0.1, "restored_bytes": 100, "attribution": "aggregate only"},
    }})
    result = compare([run, RunRef("unmonitored", 0, ".", {})])
    rendered = render_comparison(result)
    assert "| engine | — | 10.00% | 100 | aggregate only |" in rendered
    assert "| unmonitored | 0 | — | — | unavailable | unavailable | unavailable (no restore telemetry) |" in rendered


@pytest.mark.asyncio
async def test_runner_scrapes_before_admission_and_after_drain(unused_tcp_port):
    events = []

    async def request(_):
        events.append("request")
        return web.Response(text="OK")

    async def metrics(_):
        await asyncio.sleep(0.02)
        count = events.count("request")
        events.append("scrape")
        return web.Response(text=(
            f"lmcache:num_hit_tokens_total {100 + count * 5}\n"
            f"lmcache:num_prompt_tokens_total {200 + count * 10}\n"
        ))

    app = web.Application()
    app.router.add_get("/request", request)
    app.router.add_get("/metrics", metrics)
    server = web.AppRunner(app)
    await server.setup()
    await web.TCPSite(server, "127.0.0.1", unused_tcp_port).start()
    root = f"http://127.0.0.1:{unused_tcp_port}"
    try:
        result = await BenchRunner(build_config({
            "workload_type": {"type": "http", "url": f"{root}/request"},
            "load": "closed:1",
            "max_requests": 1,
            "duration": "1s",
            "monitors": [{
                "type": "prometheus", "url": f"{root}/metrics", "name": "lmcache",
                "cache_restore": True, "metric_names": ["custom_metric"], "bytes_per_token": 4,
            }],
        })).run()
    finally:
        await server.cleanup()
    assert events == ["scrape", "request", "scrape"]
    report = result.summary["cache_restores"]["lmcache"]
    assert report["restored_bytes"] == 20
    assert report["restored_token_share"] == 0.5
