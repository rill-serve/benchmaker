"""benchmaker CLI."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys

import click
import yaml

from benchmaker.config import build_config
from benchmaker.core.runner import BenchRunner
from benchmaker.recipes import all_recipes
from benchmaker.recipes._cli_shared import (
    output_options as _output_options,
    parse_headers as _parse_headers,
    write_bundle_if_requested as _write_bundle_if_requested,
)
from benchmaker.recipes._factory import make_command


# ---------------------------------------------------------------- main


@click.group()
@click.option("--log-level", default="INFO",
              type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
              help="Logging level (default: INFO).")
def main(log_level: str) -> None:
    """[benchmaker]: async HTTP benchmarking with pluggable workloads."""
    level = log_level.upper()
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    # Chatty third-party loggers (one INFO line per HTTP request / hub fetch)
    # drown out our own output. Pin them to WARNING unless DEBUG was requested.
    if level != "DEBUG":
        for noisy in ("httpx", "httpx2", "httpcore", "httpcore2", "urllib3",
                      "huggingface_hub", "filelock", "fsspec", "datasets",
                      "aiohttp"):
            logging.getLogger(noisy).setLevel(logging.WARNING)


@main.command()
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False))
@_output_options
@click.option("--dotenv", type=click.Path(), default=".env",
              help="Path to .env file to load (default: .env). "
                   "Use --dotenv '' to disable.")
@click.option("--record", "record_path", type=click.Path(), default=None,
              help="Write a JSONL request trace (with relative timestamps) to "
                   "this path. A later run can replay it deterministically via "
                   "a 'replay:' config block. Overrides any 'record:' in YAML.")
@click.option("--replay", "replay_path", type=click.Path(exists=True, dir_okay=False),
              default=None,
              help="Replay a previously recorded trace at the same relative "
                   "timings. Overrides 'workload_type' / 'workload' / 'load' "
                   "(and any 'replay:' in YAML).")
@click.option("--replay-speed", type=float, default=None,
              help="Speed multiplier for --replay (default 1.0).")
@click.option("--quiet", is_flag=True, help="Suppress progress output.")
def run(config_path: str, out_dir: str | None, run_id: str | None,
        labels: tuple[str, ...], notes: str, dotenv: str,
        record_path: str | None, replay_path: str | None,
        replay_speed: float | None, quiet: bool) -> None:
    """Run a benchmark from a YAML config file.

    Environment variables (loaded from `.env` by default) are interpolated
    into the YAML using `${VAR}` or `${VAR:-default}` syntax.
    """
    with open(config_path) as f:
        raw_cfg = yaml.safe_load(f)

    if record_path is not None:
        raw_cfg = {**raw_cfg, "record": {"path": record_path}}
    if replay_path is not None:
        replay_cfg: dict = {"path": replay_path}
        if replay_speed is not None:
            replay_cfg["speed"] = replay_speed
        raw_cfg = {**raw_cfg, "replay": replay_cfg}

    bench_cfg = build_config(raw_cfg, dotenv_path=(dotenv or None))
    if quiet:
        bench_cfg.progress_every_s = 0.0

    runner = BenchRunner(bench_cfg)
    asyncio.run(runner.run())
    runner.metrics.render(sys.stdout)
    _write_bundle_if_requested(runner, raw_cfg, out_dir, run_id, labels, notes)


@main.command()
@click.option("--url", required=True, help="Target URL.")
@click.option("--method", default="GET")
@click.option("--header", "-H", multiple=True, help="Header 'Name: value'.")
@click.option("--json-body", default=None, help="JSON body string.")
@click.option("--data", default=None, help="Raw body string.")
@click.option("--rate", default="10", help="Load spec, e.g. '100', 'poisson:100', "
              "'closed:32', 'ramp:10..500:30s'.")
@click.option("--duration", default="10s", help="Run duration (e.g. '30s', '2m').")
@click.option("--max-requests", type=int, default=None)
@click.option("--timeout", "timeout_s", default=60.0, type=float)
@click.option("--connection-limit", default=1000, type=int)
@_output_options
@click.option("--quiet", is_flag=True)
def quick(url: str, method: str, header: tuple[str, ...], json_body: str | None,
          data: str | None, rate: str, duration: str, max_requests: int | None,
          timeout_s: float, connection_limit: int,
          out_dir: str | None, run_id: str | None,
          labels: tuple[str, ...], notes: str, quiet: bool) -> None:
    """[deprecated] One-liner HTTP benchmark — use `benchmaker http` instead."""
    sys.stderr.write(
        "[benchmaker] 'quick' is deprecated; use 'benchmaker http'.\n"
    )
    cfg: dict = {
        "workload_type": {
            "type": "http",
            "url": url,
            "method": method,
            "headers": _parse_headers(header),
            "timeout_s": timeout_s,
        },
        "load": rate,
        "duration": duration,
        "max_requests": max_requests,
        "timeout_s": timeout_s,
        "connection_limit": connection_limit,
    }
    if json_body is not None:
        cfg["workload"] = {"type": "static", "items": [json.loads(json_body)]}
    elif data is not None:
        cfg["workload"] = {"type": "static", "items": [data.encode("utf-8")]}

    bench_cfg = build_config(cfg)
    if quiet:
        bench_cfg.progress_every_s = 0.0

    runner = BenchRunner(bench_cfg)
    asyncio.run(runner.run())
    runner.metrics.render(sys.stdout)
    _write_bundle_if_requested(runner, cfg, out_dir, run_id, labels, notes)


# ---------------------------------------------------------------- arms


@main.command("arms")
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--out-dir", type=click.Path(file_okay=False), required=True,
              help="Parent directory for the run. Each (arm, rep) bundle goes to "
                   "<out-dir>/<run-id>/arms/<arm>/rep-<k>/; the comparison table to "
                   "<out-dir>/<run-id>/comparison.md.")
@click.option("--run-id", default=None, help="Explicit run id (default: UTC timestamp).")
@click.option("--label", "labels", multiple=True, help="'key=value' tag for every bundle. Repeatable.")
@click.option("--notes", default="", help="Notes stored in every bundle's meta.json.")
@click.option("--dotenv", type=click.Path(), default=".env", show_default=True,
              help="Path to .env file to load. Use --dotenv '' to disable.")
@click.option("--allow-mixed-jobs", is_flag=True, default=False,
              help="Do not fail when reps come from different jobs/hosts (issue #3).")
@click.option("--reps", type=int, default=None, help="Override arms.reps.")
@click.option("--dry-run", "dry_run", is_flag=True, help="Print the execution plan and exit.")
@click.option("--quiet", is_flag=True, help="Suppress progress output.")
def arms(config_path: str, out_dir: str, run_id: str | None, labels: tuple[str, ...],
         notes: str, dotenv: str, allow_mixed_jobs: bool, reps: int | None,
         dry_run: bool, quiet: bool) -> None:
    """Run a multi-arm comparison (ABBA / rep-major / random) from a YAML config.

    The YAML holds the shared benchmark config plus an ``arms:`` block; see
    ``benchmaker/core/arms.py``. Exit code 2 when reps came from different
    jobs/hosts and --allow-mixed-jobs was not given.
    """
    from benchmaker.core.arms import ArmsRunner, plan_order
    from benchmaker.core.compare import render_comparison, render_decomposition
    from benchmaker.recipes._cli_shared import parse_labels

    with open(config_path) as f:
        raw_cfg = yaml.safe_load(f)
    if reps is not None:
        raw_cfg = {**raw_cfg, "arms": {**(raw_cfg.get("arms") or {}), "reps": reps}}

    runner = ArmsRunner(raw_cfg, out_dir=out_dir, run_id=run_id,
                        labels=parse_labels(labels), notes=notes,
                        dotenv_path=(dotenv or None), quiet=quiet,
                        allow_mixed_jobs=True if allow_mixed_jobs else None,
                        progress=not quiet)
    if dry_run:
        from benchmaker.core.arms import live_overlaps, stops_after
        s = runner.settings
        plan = plan_order([a.name for a in s.arms], s.reps, s.order, s.seed)
        by_name = {a.name: a for a in s.arms}
        for i, (rep, arm) in enumerate(plan):
            spec = by_name[arm]
            stop = "  + stop" if spec.managed and stops_after(spec, plan, i) else ""
            sys.stdout.write(f"{arm} rep-{rep}{' (warmup)' if rep < s.warmup_reps else ''}{stop}\n")
        for a, b in live_overlaps(plan, by_name):
            sys.stdout.write(f"⚠ {a} and {b} are both live at some point (relaunch "
                             f"{by_name[a].relaunch}/{by_name[b].relaunch}); use relaunch: visit or "
                             "every_rep if they share GPUs\n")
        return

    result = runner.run()
    sys.stdout.write("\n" + render_comparison(result) + "\n")
    for w in result.get("ttft_parity_warnings", []):
        sys.stdout.write(f"\n⚠ TTFT parity: {w}\n")
    for dec in result.get("decomposition", []):
        sys.stdout.write("\n" + render_decomposition(dec, result["reference"], dec["arm"]) + "\n")
    for v in result.get("same_job_violations", []):
        sys.stderr.write(f"\n[benchmaker] same-job check: {v}\n")
    if result.get("same_job_violations") and not runner.settings.allow_mixed_jobs:
        sys.stderr.write("[benchmaker] refusing to merge results across jobs/hosts; "
                         "re-run with --allow-mixed-jobs to accept.\n")
        sys.exit(runner.exit_code(result))


@main.command("compare")
@click.argument("paths", nargs=-1, required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--reference", default=None, help="Reference arm (default: first).")
@click.option("--metric", "metrics", multiple=True, help="Metric alias or dotted summary path. Repeatable.")
@click.option("--warmup-reps", type=int, default=None, help="Override the number of warm-up reps to exclude.")
@click.option("--allow-mixed-jobs", is_flag=True, default=False)
@click.option("--format", "fmt", type=click.Choice(["md", "json"]), default="md", show_default=True)
def compare(paths: tuple[str, ...], reference: str | None, metrics: tuple[str, ...],
            warmup_reps: int | None, allow_mixed_jobs: bool, fmt: str) -> None:
    """Compare finished arms: per-rep medians, min–max, ratios, same-job check,
    TTFT parity and the TTFT mix/latency decomposition.

    Each PATH is either an arms run root (``<run-id>/arms/<arm>/rep-<k>/``) or
    a directory of such roots. Re-runs the comparison without new requests.
    """
    import json as _json

    from benchmaker.core.arms import (
        ArmsSettings, build_comparison, discover_arm_dirs, runs_from_arm_dirs,
    )
    from benchmaker.core.compare import render_comparison, render_decomposition

    arm_dirs: dict[str, list[str]] = {}
    roots: list[str] = []
    for p in paths:
        if os.path.isdir(os.path.join(p, "arms")):
            roots.append(p)
        else:
            for sub in sorted(os.listdir(p)):
                if os.path.isdir(os.path.join(p, sub, "arms")):
                    roots.append(os.path.join(p, sub))
    for root in roots:
        for arm, dirs in discover_arm_dirs(root).items():
            arm_dirs.setdefault(arm, []).extend(dirs)
    if not arm_dirs:
        raise click.UsageError("no arms found; expected <path>/arms/<arm>/rep-<k>/ bundles")
    warm = warmup_reps if warmup_reps is not None else _detect_warmup(roots)
    runs = runs_from_arm_dirs(arm_dirs, warmup_reps=warm)
    settings = ArmsSettings(reference=reference, metrics=list(metrics) or None,
                            allow_mixed_jobs=allow_mixed_jobs,
                            warmup_reps=warm, reps=max(r.rep for r in runs) + 1)
    result = build_comparison(runs, settings)
    if fmt == "json":
        sys.stdout.write(_json.dumps(result, indent=2, default=str) + "\n")
        return
    sys.stdout.write(render_comparison(result) + "\n")
    for w in result.get("ttft_parity_warnings", []):
        sys.stdout.write(f"\n⚠ TTFT parity: {w}\n")
    for dec in result.get("decomposition", []):
        sys.stdout.write("\n" + render_decomposition(dec, result["reference"], dec["arm"]) + "\n")
    if result.get("same_job_violations"):
        for v in result["same_job_violations"]:
            sys.stderr.write(f"\n[benchmaker] same-job check: {v}\n")
        if not allow_mixed_jobs:
            sys.exit(2)


def _detect_warmup(roots: list[str]) -> int:
    for root in roots:
        try:
            with open(os.path.join(root, "plan.json")) as f:
                return int(json.load(f).get("warmup_reps", 0))
        except (OSError, ValueError):
            continue
    return 0


# ---------------------------------------------------------------- collect


@main.command()
@click.argument("paths", nargs=-1, required=True,
                type=click.Path(exists=True, file_okay=False))
@click.option("--format", "fmt", type=click.Choice(["md", "csv", "json"]),
              default="md", show_default=True,
              help="Output format. 'md' is a Markdown table, 'csv' is comma-separated, "
                   "'json' is a JSON array of row dicts.")
@click.option("--metric", "metrics", multiple=True,
              help="Extra dotted-path metric to add as a column "
                   "(e.g. 'workload_metrics.ttft_s.p50'). Repeatable.")
@click.option("--columns", default=None,
              help="Comma-separated list of column names to keep (after metrics are added). "
                   "Overrides the default column set.")
@click.option("--sort-by", default=None,
              help="Column name to sort rows by (ascending).")
@click.option("--label", "label_keys", multiple=True,
              help="Promote a meta.labels[<key>] entry into its own column. Repeatable.")
@click.option("--recursive/--no-recursive", default=True,
              help="When a path is a directory of run-dirs, descend one level to find them.")
def collect(paths: tuple[str, ...], fmt: str, metrics: tuple[str, ...],
            columns: str | None, sort_by: str | None,
            label_keys: tuple[str, ...], recursive: bool) -> None:
    """Collect summaries from one or more run-dirs into a table.

    Each PATH may be a run directory (containing meta.json + summary.json) or a
    directory of such run-dirs. With --recursive (default), a non-bundle
    directory is scanned for immediate subdirectories that are bundles.
    """
    from benchmaker.io.collect import collect_table, format_table, find_bundles

    bundle_dirs: list[str] = []
    for p in paths:
        bundle_dirs.extend(find_bundles(p, recursive=recursive))
    if not bundle_dirs:
        raise click.UsageError(
            f"No run bundles found under: {', '.join(paths)}. "
            "Run bundles must contain meta.json and summary.json."
        )

    rows, column_names = collect_table(
        bundle_dirs,
        extra_metrics=list(metrics),
        label_keys=list(label_keys),
    )
    if columns:
        column_names = [c.strip() for c in columns.split(",") if c.strip()]
    if sort_by:
        rows.sort(key=lambda r: (r.get(sort_by) is None, r.get(sort_by)))

    sys.stdout.write(format_table(rows, column_names, fmt))
    if fmt != "json":
        sys.stdout.write("\n")


# ---------------------------------------------------------------- recipes
#
# Each registered recipe (http, llm, sandbox, swebench, ...) is exposed as a
# `benchmaker <recipe> --args` subcommand, built from the recipe's options plus
# the shared load/output options. See benchmaker/recipes/.

for _recipe in all_recipes():
    main.add_command(make_command(_recipe))


if __name__ == "__main__":
    main()
