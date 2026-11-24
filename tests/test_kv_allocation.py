"""KV budget manifests and comparison for host cache backends."""

import pytest

from benchmaker.config import build_config
from benchmaker.core.allocation import GB, GIB
from benchmaker.core.compare import RunRef, compare, render_comparison
from benchmaker.core.runner import BenchRunner
from benchmaker.io.bundle import read_bundle
from benchmaker.io.manifest import build_manifest


def _manifest(allocation=None, *, capability=None, endpoints=None, launch_log=None):
    return build_manifest(
        source_config={}, run_id="test", allocation=allocation, capability=capability,
        endpoints=endpoints, launch_log=launch_log,
    )


def _snap(doc):
    return {"start": {"ok": True, "data": doc, "url": "http://localhost:8000/get_server_info"}}


@pytest.mark.parametrize(("line", "bytes_budget", "source_name"), [
    ("non-default args: {'kv_offloading_size': 8.0}", 8 * GIB, "kv_offloading_size"),
    ("LMCache Configuration: {'local_cpu': True, 'max_local_cpu_size': 8.0}", 8 * GIB, "max_local_cpu_size"),
    ("LMCache Configuration: {'local_cpu': False, 'max_local_cpu_size': 8.0}", 0, "local_cpu"),
])
def test_vllm_lmcache_log(line, bytes_budget, source_name, tmp_path):
    log = tmp_path / "serve.log"
    log.write_text(f"{line}\nGPU KV cache size: 58,255 tokens\n")
    allocation = _manifest({"bytes_per_token": 147456}, launch_log=str(log))["allocation"]
    assert allocation["device_kv_tokens"] == 58255
    assert allocation["host_tier_bytes"] == bytes_budget
    assert allocation["host_tier_tokens"] == bytes_budget // 147456
    assert allocation["fleet_host_total_bytes"] == bytes_budget
    assert allocation["sharing"] == "private"
    source = allocation["sources"]["host_tier_bytes"]
    assert source == {"source": f"log:{log}:1:{source_name}", "unit": "GiB", "value": bytes_budget / GIB}


def test_sglang_decimal_host_memory_and_token_counts(tmp_path):
    log = tmp_path / "sgl.log"
    log.write_text(
        "KV Cache is allocated. dtype: torch.bfloat16, #tokens: 10,800, KV size: 5.80 GB\n"
        "Allocating kv hierarchical KV host pool: 54254 tokens, 8.00 GB host memory.\n"
    )
    allocation = _manifest({"startup_log": str(log)})["allocation"]
    assert allocation["device_kv_tokens"] == 10800
    assert allocation["fleet_host_total_bytes"] == 8 * GB
    assert allocation["host_tier_tokens"] == 54254
    assert allocation["bytes_per_token"] == pytest.approx(8 * GB / 54254)
    assert allocation["sources"]["host_tier_bytes"]["unit"] == "GB"
    assert allocation["sources"]["bytes_per_token"]["approximate"]


def test_rill_shared_fleet_is_counted_once_and_quota_is_per_engine(tmp_path):
    daemon = tmp_path / "kvd.log"
    daemon.write_text("rill-kvd: serving /tmp/kvd.sock (ram_gb=16, ram_quota_per_model_gb=8, shm=off)\n")
    endpoints = {"a": "http://a", "b": "http://b"}
    capability = {"endpoints": {
        "a": _snap({"kv_capacity_tokens": 10000}), "b": _snap({"kv_capacity_tokens": 20000}),
    }}
    manifest = _manifest({"kvd_log": str(daemon), "bytes_per_token": 147456},
                         endpoints=endpoints, capability=capability)
    assert manifest["allocation"]["fleet_host_total_bytes"] == 16 * GIB
    assert manifest["allocation"]["sharing"] == "shared:/tmp/kvd.sock"
    a, b = (manifest["endpoints"][name]["allocation"] for name in endpoints)
    assert (a["device_kv_tokens"], b["device_kv_tokens"]) == (10000, 20000)
    assert a["host_tier_bytes"] == b["host_tier_bytes"] == 8 * GIB
    assert a["host_tier_tokens"] == b["host_tier_tokens"] == 58254
    assert a["sources"]["device_kv_tokens"]["source"].endswith("#kv_capacity_tokens")
    daemon.write_text("rill-kvd: serving /tmp/kvd.sock (ram_gb=16, ram_quota_per_model_gb=off)\n")
    uncapped = _manifest({"kvd_log": str(daemon)}, endpoints=endpoints, capability=capability)
    assert uncapped["allocation"]["fleet_host_total_bytes"] == 16 * GIB
    assert uncapped["endpoints"]["a"]["allocation"]["host_tier_bytes"] is None


def test_compare_endpoint_budgets_warn_on_fleet_difference(tmp_path):
    endpoints = {"a": "http://a", "b": "http://b"}
    left = tmp_path / "left.log"
    right = tmp_path / "right.log"
    runs = []
    for name, host in [("vllm", "kv_offloading_size=8"), ("sglang", "--hicache-size 8")]:
        left.write_text(f"GPU KV cache size: 100 tokens\n{host}\n")
        right.write_text(f"GPU KV cache size: 200 tokens\n{host}\n")
        manifest = _manifest(
            {"endpoints": {"a": {"startup_log": str(left)}, "b": {"startup_log": str(right)}}},
            endpoints=endpoints,
        )
        runs.append(RunRef(name, 0, "", {"throughput_rps": 10}, manifest=manifest))
    runs.append(RunRef("legacy", 0, "", {"throughput_rps": 15}))
    result = compare(runs, ["rps"])
    assert result["allocation_warnings"] == [
        "Fleet host KV budgets differ by more than 5% (sglang=16,000,000,000, vllm=17,179,869,184 bytes)."
    ]
    assert result["arms"]["sglang"]["allocations"][0]["endpoints"]["a"]["device_kv_tokens"] == 100
    markdown = render_comparison(result)
    assert "device KV tokens" in markdown and "a 100<br>b 200" in markdown
    assert "fleet host bytes" in markdown and "16,000,000,000" in markdown
    assert result["allocation_warnings"][0] in markdown


def test_compare_does_not_mistake_unknown_or_void_budget_for_zero():
    limit = _manifest(capability=_snap({"kv_offloading_size": 8}))
    unknown = _manifest(capability=_snap({"kv_cache_info": {"max_total_num_tokens": 5000}}))
    assert unknown["allocation"]["fleet_host_total_bytes"] is None
    assert unknown["allocation"]["device_kv_tokens"] == 5000
    zero = _manifest(capability=_snap({"kv_offloading_size": 0}))
    runs = [RunRef("a", 0, "", {}, manifest=limit), RunRef("b", 0, "", {}, manifest=unknown),
            RunRef("a", 1, "", {}, manifest=zero, meta={"void": True})]
    assert not compare(runs)["allocation_warnings"]


def test_runner_writes_allocation_from_config_and_launch_hook(tmp_path):
    log = tmp_path / "launch.log"
    log.write_text("GPU KV cache size: 22 tokens\nkv_offloading_size=1\n")
    config = build_config({
        "workload_type": {"type": "http", "url": "http://localhost:8000"},
        "load": 10,
        "duration": "1s",
        "allocation": {"bytes_per_token": 2**17},
    }, dotenv_path=None)
    runner = BenchRunner(config)
    out = runner.write_bundle(str(tmp_path), run_id="test", hooks={"start": {"log": str(log)}})
    allocation = read_bundle(out)["manifest"]["allocation"]
    assert allocation["device_kv_tokens"] == 22
    assert allocation["host_tier_tokens"] == 8192
