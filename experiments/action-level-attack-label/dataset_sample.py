"""Write a human-browsable sample of the export (not a result — a reading aid).

Picks a handful of trajectories and renders every action the way the monitor sees it
(`format_action(a, hide_output=True)` = the `current_action` string the A2 regexes run on),
with the tool output truncated underneath, the cached monitor scores, and whether the
action is in subset_main. Also writes one fully rendered monitor prompt so you can see
exactly what the model was scored on.

Output goes to checkpoints/<slug>/dataset_sample/ (gitignored: contains cached scores and
action text, which the A2 session must not see).
"""
from __future__ import annotations

import html
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SLUG = HERE.name
REPO = HERE.parents[1]
GPU = REPO / "experiments" / "logit-curve-replication" / "gpu"
sys.path.insert(0, str(GPU))
import format_monitor_prompt as fmp  # noqa: E402

EXPORT = REPO / "checkpoints" / "local-monitor-activations" / "data" / "export"
SUBSET = REPO / "experiments" / "local-monitor-activations" / "subset_main.json"
OUT = REPO / "checkpoints" / SLUG / "dataset_sample"
OUT_TRUNC = 1500


def pick(trajs: dict) -> list[str]:
    rng = random.Random(0)
    attack = [t for t in trajs.values() if t["side"] == "attack"]
    honest = [t for t in trajs.values() if t["side"] == "honest"]
    live_ok = [t for t in attack if t["side_task_failure_type"] == "live" and t["side_task_succeeded"]]
    delayed_ok = [t for t in attack if t["side_task_failure_type"] == "delayed" and t["side_task_succeeded"]]
    failed = [t for t in attack if not t["side_task_succeeded"]]
    chosen = []
    # shortish trajectories so the files stay readable
    for pool, n in ((live_ok, 2), (delayed_ok, 2), (failed, 1), (honest, 1)):
        pool = sorted(pool, key=lambda t: t["traj_id"])
        short = [t for t in pool if t["n_actions_in_traj"] <= 45] or pool
        chosen += rng.sample(short, min(n, len(short)))
    return [t["traj_id"] for t in chosen]


def trunc(s: str | None, n: int = OUT_TRUNC) -> str:
    if s is None:
        return ""
    return s if len(s) <= n else s[:n] + f"\n... [truncated, {len(s) - n} more chars]"


def render_md(t: dict, in_subset: set[tuple[str, int]]) -> str:
    L = [f"# {t['side']} — {t['env']} / {t['main_task']}", ""]
    L += [f"- traj_id: `{t['traj_id']}`   ({t['trajectory_url']})",
          f"- side: **{t['side']}**   side_task: `{t['side_task']}`   category: `{t['side_task_category']}`   failure_type: `{t['side_task_failure_type']}`",
          f"- side_task_succeeded: **{t['side_task_succeeded']}**   main_task_success: {t['main_task_success']}",
          f"- n_actions: {t['n_actions_in_traj']}   side_task_trigger_action_idx: {t['side_task_trigger_action_idx']}",
          f"- policy: {t['policy']}", ""]
    if t["side_task_goal"]:
        L += ["## Side-task objective (what the attacker was told to do)", "", "> " + t["side_task_goal"].strip(), ""]
    L += ["## Main task description (what the monitor sees as `task_description`)", "", "```", t["main_task_description"].strip(), "```", ""]
    L += ["## Side-task goal block shown to the monitor for this env (`side_task_goals`)", "", "```", t["side_task_goals"].strip(), "```", ""]
    L += ["## Actions", "",
          "Each action shows: the `current_action` string exactly as the monitor sees it (this is what the A2",
          "regexes run against), then the tool output (truncated), the cached monitor scores (2 samples of",
          "GPT-5 Nano on the old_action prompt) and whether the action is in subset_main.", ""]
    for a in t["actions"]:
        i = a["index"]
        sub = "in subset_main" if (t["traj_id"], i) in in_subset else "not in subset_main"
        skip = f"  SKIPPED BY MONITOR: {a['skipped_by_monitor']}" if a["skipped_by_monitor"] else ""
        L += [f"### Action {i}  ·  tool=`{a['function']}`  ·  cached scores={a['monitor_scores']} (mean {a['monitor_score_mean']})  ·  {sub}{skip}", ""]
        L += ["```", fmp.format_action(a, hide_output=True), "```", ""]
        out = fmp._format_action_output(a)
        if out:
            L += ["<details><summary>tool output</summary>", "", "```", trunc(out), "```", "", "</details>", ""]
        if a.get("output_error"):
            L += [f"Error: `{trunc(a['output_error'], 300)}`", ""]
    return "\n".join(L)


def render_html(t: dict, in_subset: set[tuple[str, int]]) -> str:
    e = html.escape
    H = ["<style>body{font-family:system-ui;max-width:1100px;margin:auto;padding:16px}pre{white-space:pre-wrap;word-break:break-all;background:#f5f5f5;padding:8px;border-radius:4px}"
         ".act{border:1px solid #ddd;border-radius:6px;padding:8px;margin:8px 0}.hi{background:#fff3e0}.skip{opacity:.6}.sub{color:#0a7}.cur{background:#eef;border-left:4px solid #55f}"
         "summary{cursor:pointer;color:#555}</style>"]
    H += [f"<h1>{e(t['side'])} — {e(t['env'])} / {e(t['main_task'])}</h1>",
          f"<p>traj_id <code>{t['traj_id']}</code> · <a href='{t['trajectory_url']}'>linuxarena page</a><br>"
          f"side_task <code>{e(str(t['side_task']))}</code> · category <code>{e(str(t['side_task_category']))}</code> · failure_type <code>{e(str(t['side_task_failure_type']))}</code><br>"
          f"side_task_succeeded <b>{t['side_task_succeeded']}</b> · main_task_success {t['main_task_success']} · n_actions {t['n_actions_in_traj']} · trigger idx {t['side_task_trigger_action_idx']}</p>"]
    if t["side_task_goal"]:
        H += [f"<h2>Side-task objective</h2><blockquote>{e(t['side_task_goal'])}</blockquote>"]
    H += [f"<details><summary><b>Main task description</b> (task_description in the prompt)</summary><pre>{e(t['main_task_description'])}</pre></details>",
          f"<details><summary><b>Side-task goal block</b> shown to the monitor for this env</summary><pre>{e(t['side_task_goals'])}</pre></details>",
          f"<details><summary><b>Environment description</b> (in the export; not in the old_action prompt)</summary><pre>{e(t['environment_description'] or '')}</pre></details>",
          "<h2>Actions</h2><p>Blue box = the <code>current_action</code> string the monitor sees (what A2 regexes run on). Orange = cached mean ≥ 6. Faded = skipped by monitor.</p>"]
    for a in t["actions"]:
        i = a["index"]
        sub = "<span class=sub>in subset_main</span>" if (t["traj_id"], i) in in_subset else "not in subset_main"
        cls = "act" + (" hi" if (a["monitor_score_mean"] or 0) >= 6 else "") + (" skip" if a["skipped_by_monitor"] else "")
        skip = f" · <i>skipped: {e(a['skipped_by_monitor'])}</i>" if a["skipped_by_monitor"] else ""
        H += [f"<div class='{cls}'><b>Action {i}</b> · tool <code>{e(a['function'])}</code> · cached scores {a['monitor_scores']} (mean {a['monitor_score_mean']}) · {sub}{skip}",
              f"<pre class=cur>{e(fmp.format_action(a, hide_output=True))}</pre>"]
        out = fmp._format_action_output(a)
        if out:
            H += [f"<details><summary>tool output ({len(out)} chars)</summary><pre>{e(trunc(out, 4000))}</pre></details>"]
        if a.get("output_error"):
            H += [f"<p>Error: <code>{e(trunc(a['output_error'], 300))}</code></p>"]
        H += ["</div>"]
    return "\n".join(H)


def main() -> None:
    trajs = fmp.load_export(EXPORT)
    subset = json.load(open(SUBSET))
    in_subset = {(a["traj_id"], a["action_idx"]) for a in subset["actions"]}
    OUT.mkdir(parents=True, exist_ok=True)
    ids = pick(trajs)
    index = ["# Dataset sample — reading aid, not a result", "",
             "Files: one `.md` and one `.html` per trajectory (same content; the html has collapsible outputs),",
             "`<traj>.raw.jsonl` = the untouched export rows for that trajectory, and `rendered_prompt_*.txt` = a",
             "complete monitor prompt for one action, byte-identical to what the cached monitor was sent.", "",
             "| file | side | env | side_task | category | failure_type | succeeded | n_actions |", "|---|---|---|---|---|---|---|---|"]
    raw_by_traj: dict[str, list[str]] = {}
    for side in ("attack", "honest"):
        with (EXPORT / f"{side}.jsonl").open() as f:
            for line in f:
                tid = json.loads(line)["traj_id"]
                if tid in ids:
                    raw_by_traj.setdefault(tid, []).append(line)
    for k, tid in enumerate(ids):
        t = trajs[tid]
        stem = f"{k:02d}_{t['side']}_{t['env']}_{t['side_task'] or t['main_task']}"
        (OUT / f"{stem}.md").write_text(render_md(t, in_subset))
        (OUT / f"{stem}.html").write_text(render_html(t, in_subset))
        (OUT / f"{stem}.raw.jsonl").write_text("".join(raw_by_traj[tid]))
        index.append(f"| [{stem}]({stem}.md) / [html]({stem}.html) | {t['side']} | {t['env']} | {t['side_task']} | {t['side_task_category']} | "
                     f"{t['side_task_failure_type']} | {t['side_task_succeeded']} | {t['n_actions_in_traj']} |")
    # one full rendered prompt: the first chosen attack trajectory, its highest-cached action
    t = trajs[ids[0]]
    scorable = [a for a in t["actions"] if not a["skipped_by_monitor"]]
    a = max(scorable, key=lambda a: a["monitor_score_mean"] or 0)
    p = OUT / f"rendered_prompt_{t['traj_id']}_action{a['index']}.txt"
    p.write_text(fmp.build_prompt(t, a["index"]))
    index += ["", f"Full rendered monitor prompt for `{t['traj_id']}` action {a['index']} (cached mean {a['monitor_score_mean']}): `{p.name}`", ""]
    (OUT / "README.md").write_text("\n".join(index))
    print("\n".join(index))
    print("\nwritten to", OUT.relative_to(REPO))


if __name__ == "__main__":
    main()
