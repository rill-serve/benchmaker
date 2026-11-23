"""``session-replay`` recipe — run a prepared session replay file (issue #10).

Each session's turns go out in order, turn k+1 after turn k completed plus a think time, so multi-turn prefix
reuse (and its eviction between turns) is reproduced the same way for every arm. See
:mod:`benchmaker.workloads.session_replay` for the file format.

Session starts: ``--concurrent-sessions N`` (closed loop, the default 8), ``--session-rate R`` (Poisson session
starts) or ``--trace-scale S`` (each session at its recorded ``at_s`` times S). ``--think-median-s`` overrides the
manifest's think-time median. ``--template-check`` (default on when the manifest has a probe) refuses to run when
the server's prompt-token count for the probe conversation differs from the prepared count.
"""

from __future__ import annotations

from typing import Any

import click

from benchmaker.recipes import register
from benchmaker.recipes._cli_shared import parse_headers
from benchmaker.recipes.base import BuildResult, Recipe, SharedOpts


class ReplayRecipe(Recipe):
    name = "session-replay"
    help = ("Run a prepared session replay file (per-session turn dependencies, prompt-set hash, "
            "recorded-output pinning) against an OpenAI-compatible chat endpoint.")

    def options(self) -> list:
        return [
            click.option("--replay-file", "replay_file", required=True,
                         type=click.Path(exists=True, dir_okay=False), help="Replay JSONL (manifest + rows)."),
            click.option("--url", default=None,
                         help="Chat-completions URL. Falls back to $OPENAI_API_BASE_URL/$OPENAI_BASE_URL."),
            click.option("--model", default=None,
                         help="Served model name. Falls back to $OPENAI_COMPATIBLE_MODEL/$OPENAI_MODEL."),
            click.option("--api-key", "api_key", default=None, help="API key. Falls back to $OPENAI_API_KEY."),
            click.option("--header", "-H", "header", multiple=True, help="Extra header 'Name: value'. Repeatable."),
            click.option("--model-label", "model_label", default=None,
                         help="Replay only the sessions whose row 'model' equals this label (multi-model files)."),
            click.option("--concurrent-sessions", "concurrent_sessions", type=int, default=None,
                         help="Closed loop: N sessions active (default 8 when no other mode is given)."),
            click.option("--session-rate", "session_rate", type=float, default=None,
                         help="Open loop: Poisson session starts per second."),
            click.option("--trace-scale", "trace_scale", type=float, default=None,
                         help="Trace timing: each session starts at its first row's at_s times this."),
            click.option("--think-median-s", "think_median_s", type=float, default=None,
                         help="Override the manifest's think-time median (seconds)."),
            click.option("--max-in-flight", "max_in_flight", type=int, default=None,
                         help="Request slots for open/trace modes (default 256)."),
            click.option("--template-check/--no-template-check", "template_check", default=True,
                         show_default=True,
                         help="Refuse to run when the probe conversation's prompt_tokens differ from the prepared "
                              "count (needs a 'probe' in the manifest)."),
            click.option("--seed", type=int, default=0, show_default=True, help="Seed for Poisson session starts."),
        ]

    def build(self, shared: SharedOpts, *, replay_file, url, model, api_key, header, model_label=None,
              concurrent_sessions=None, session_rate=None, trace_scale=None, think_median_s=None,
              max_in_flight=None, template_check=True, seed=0) -> BuildResult:
        from benchmaker.workloads.llm import OpenAIChatWorkloadType
        from benchmaker.workloads.session_replay import ReplayWorkload
        from benchmaker.workloads.session_replay import template_check as probe_check

        if concurrent_sessions is None and session_rate is None and trace_scale is None:
            concurrent_sessions = 8
        headers = parse_headers(header)
        wt_kwargs: dict[str, Any] = {"timeout_s": shared.timeout_s, "headers": headers, "passthrough_meta": True}
        wt = OpenAIChatWorkloadType.from_env(url=url, model=model, api_key=api_key, dotenv_path=shared.dotenv,
                                             **wt_kwargs)
        try:
            workload = ReplayWorkload(replay_file, concurrent_sessions=concurrent_sessions,
                                      session_rate=session_rate, trace_scale=trace_scale,
                                      think_median_s=think_median_s, seed=seed, model_label=model_label)
        except ValueError as e:
            raise click.UsageError(str(e)) from e

        probe = (workload.manifest.get("probe") or {})
        probe = probe.get(model_label) if model_label else next(iter(probe.values()), None)
        check: dict[str, Any] = {"enabled": bool(template_check and probe)}
        if check["enabled"]:
            got, want = probe_check(wt._url, wt._model, probe, headers=headers, timeout_s=shared.timeout_s)
            check.update(server=got, prepared=want)
            if got != want:
                raise click.UsageError(
                    f"chat template check failed: the server counts {got} prompt tokens for the probe, the replay "
                    f"file was prepared with {want}; the server renders a different template or thinking mode "
                    f"(--no-template-check to run anyway)")

        source_config = {
            "workload_type": {"type": "openai-chat", "url": wt._url, "model": wt._model,
                              **{k: v for k, v in wt_kwargs.items() if k != "headers"}},
            "workload": {"type": "session-replay", "path": replay_file, "model_label": model_label,
                         "concurrent_sessions": concurrent_sessions, "session_rate": session_rate,
                         "trace_scale": trace_scale, "think_median_s": workload.think_median_s,
                         "think_sigma": workload.think_sigma, "seed": seed,
                         "prompt_set_sha256": workload.content_hash(), "template_check": check},
        }
        slots = concurrent_sessions if concurrent_sessions is not None else (max_in_flight or 256)
        return BuildResult(workload_type=wt, workload=workload, source_config=source_config,
                           post_hooks=[workload.completion_hook()],
                           default_rate=f"closed:{slots}", default_duration="24h")


register(ReplayRecipe())
