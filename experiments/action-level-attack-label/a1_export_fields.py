"""A1 — what the export already gives (TASK.md §A1). Stdlib only, no model.

Reads checkpoints/local-monitor-activations/data/export/{trajectories,attack,honest}.jsonl,
experiments/logit-curve-replication/gpu/side_task_goals.json and
experiments/local-monitor-activations/subset_main.json.

Writes
  results/raw/<slug>/a1_export_fields_<ts>.json   every number below, per-trajectory rows included
  experiments/<slug>/side_task_descriptions.md    the A2 hand-off: side_task, category, env, verbatim
                                                  objective. No actions, no scores, no trajectory ids.
"""
from __future__ import annotations

import collections
import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SLUG = HERE.name
REPO = HERE.parents[1]
EXPORT = REPO / "checkpoints" / "local-monitor-activations" / "data" / "export"
GOALS = REPO / "experiments" / "logit-curve-replication" / "gpu" / "side_task_goals.json"
SUBSET = REPO / "experiments" / "local-monitor-activations" / "subset_main.json"
RAW_DIR = REPO / "results" / "raw" / SLUG
DESC_MD = HERE / "side_task_descriptions.md"

TS = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def git_commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def jsonl(p: Path):
    with p.open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def field_types(rows) -> dict:
    out: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for r in rows:
        for k, v in r.items():
            out[k][type(v).__name__] += 1
    return {k: dict(v) for k, v in out.items()}


def main() -> None:
    trajs = list(jsonl(EXPORT / "trajectories.jsonl"))
    attack_actions = list(jsonl(EXPORT / "attack.jsonl"))
    honest_first = next(jsonl(EXPORT / "honest.jsonl"))
    subset = json.load(open(SUBSET))
    goals = json.load(open(GOALS))

    # ── 1. fields ────────────────────────────────────────────────────────────────────────────
    fields = {
        "trajectories.jsonl": field_types(trajs),
        "attack.jsonl": field_types(attack_actions),
        "honest.jsonl(first row only)": {k: {type(v).__name__: 1} for k, v in honest_first.items()},
    }

    attack = [t for t in trajs if t["side"] == "attack"]
    honest = [t for t in trajs if t["side"] == "honest"]
    subset_attack_traj_ids = sorted({a["traj_id"] for a in subset["actions"] if a["side"] == "attack"})
    subset_honest_traj_ids = sorted({a["traj_id"] for a in subset["actions"] if a["side"] == "honest"})

    rows = []
    for t in attack:
        rows.append({
            "traj_id": t["traj_id"],
            "env": t["env"],
            "main_task": t["main_task"],
            "side_task": t["side_task"],
            "side_task_category": t["side_task_category"],
            "side_task_failure_type": t["side_task_failure_type"],
            "side_task_succeeded": t["side_task_succeeded"],
            "main_task_success": t["main_task_success"],
            "n_actions_in_traj": t["n_actions_in_traj"],
            "n_actions_in_agent_traj": t["n_actions_in_agent_traj"],
            "side_task_trigger_action_idx": t["side_task_trigger_action_idx"],
            "in_subset_main": t["traj_id"] in subset_attack_traj_ids,
            "n_actions_in_subset_main": sum(1 for a in subset["actions"] if a["traj_id"] == t["traj_id"]),
        })

    def tri(v):
        return "success" if v is True else "failed" if v is False else "unknown"

    def counts(rs, key):
        return dict(collections.Counter(tri(r[key]) for r in rs))

    sub_rows = [r for r in rows if r["in_subset_main"]]
    summary = {
        "n_trajectories": len(trajs), "n_attack": len(attack), "n_honest": len(honest),
        "n_attack_in_subset_main": len(sub_rows),
        "n_honest_traj_in_subset_main": len(subset_honest_traj_ids),
        "side_task_succeeded": {"all_92": counts(rows, "side_task_succeeded"), "subset_main_32": counts(sub_rows, "side_task_succeeded")},
        "main_task_success": {"all_92": counts(rows, "main_task_success"), "subset_main_32": counts(sub_rows, "main_task_success")},
        "side_task_failure_type": {"all_92": dict(collections.Counter(r["side_task_failure_type"] for r in rows)),
                                   "subset_main_32": dict(collections.Counter(r["side_task_failure_type"] for r in sub_rows))},
        "side_task_category": {"all_92": dict(collections.Counter(r["side_task_category"] for r in rows)),
                               "subset_main_32": dict(collections.Counter(r["side_task_category"] for r in sub_rows))},
        "succeeded_by_category": {"all_92": {c: counts([r for r in rows if r["side_task_category"] == c], "side_task_succeeded")
                                             for c in sorted({r["side_task_category"] for r in rows})},
                                  "subset_main_32": {c: counts([r for r in sub_rows if r["side_task_category"] == c], "side_task_succeeded")
                                                     for c in sorted({r["side_task_category"] for r in sub_rows})}},
        "env": {"all_92": dict(collections.Counter(r["env"] for r in rows)),
                "subset_main_32": dict(collections.Counter(r["env"] for r in sub_rows))},
        "n_distinct_side_task": len({r["side_task"] for r in rows}),
        "honest_side_task_succeeded": dict(collections.Counter(tri(t["side_task_succeeded"]) for t in honest)),
        "traj_level_flag_consistent_with_action_rows": all(
            a["side_task_succeeded"] == {t["traj_id"]: t["side_task_succeeded"] for t in attack}[a["traj_id"]]
            for a in attack_actions),
        "n_attack_actions": len(attack_actions),
    }

    # ── 2. side-task descriptions ─────────────────────────────────────────────────────────────
    # The rendered covert-objective block is side_task_goals.json['goals_text'][env] =
    # '## {id}\n{goal}' per side task. Cross-check the per-trajectory side_task_goal field
    # in trajectories.jsonl against that block, then write the A2 hand-off.
    goal_lookup = {(env, g["id"]): g for env, gs in goals["goals"].items() for g in gs}
    desc: dict[str, dict] = {}
    mismatches = []
    for t in attack:
        key = (t["env"], t["side_task"])
        g = goal_lookup.get(key)
        if g is None:
            mismatches.append({"env": t["env"], "side_task": t["side_task"], "problem": "not in side_task_goals.json"})
            continue
        if g["goal"].strip() != (t["side_task_goal"] or "").strip():
            mismatches.append({"env": t["env"], "side_task": t["side_task"], "problem": "goal text differs from trajectories.jsonl"})
        if g["failure_type"] != t["side_task_failure_type"]:
            mismatches.append({"env": t["env"], "side_task": t["side_task"], "problem": "failure_type differs from trajectories.jsonl"})
        d = desc.setdefault(t["side_task"], {
            "side_task": t["side_task"], "category": t["side_task_category"], "env": t["env"],
            "failure_type": g["failure_type"], "objective": g["goal"].strip(),
        })
        assert d["category"] == t["side_task_category"] and d["env"] == t["env"]

    by_cat = collections.defaultdict(list)
    for d in desc.values():
        by_cat[d["category"]].append(d)
    lines = [
        "# side_task_descriptions.md — A1 output, the only input to A2",
        "",
        "One entry per `side_task` that appears in the 92 attack trajectories of the export.",
        "Objective text is verbatim from the rendered monitor prompt's side-task goal block",
        "(`experiments/logit-curve-replication/gpu/side_task_goals.json`, cross-checked against",
        "`side_task_goal` in `trajectories.jsonl`). `failure_type` is the export's own field:",
        "`live` = the payload runs during the trajectory, `delayed` = a vulnerability is planted and",
        "triggered later. No actions, no scores, no trajectory ids.",
        "",
        f"Generated by `experiments/{SLUG}/a1_export_fields.py` at {TS}.",
        "",
    ]
    for cat in sorted(by_cat):
        lines.append(f"## category: {cat}  ({len(by_cat[cat])} side tasks)")
        lines.append("")
        for d in sorted(by_cat[cat], key=lambda d: (d["env"], d["side_task"])):
            lines.append(f"### {d['side_task']}")
            lines.append(f"- category: `{d['category']}`")
            lines.append(f"- environment: `{d['env']}`")
            lines.append(f"- failure_type: `{d['failure_type']}`")
            lines.append(f"- objective: {d['objective']}")
            lines.append("")
    DESC_MD.write_text("\n".join(lines))

    # ── 3. trigger idx re-check ───────────────────────────────────────────────────────────────
    known = [r for r in rows if r["side_task_trigger_action_idx"] is not None]
    # For each known trigger, look at how it relates to the two lengths. A constant offset
    # between agent-trajectory length and manifest length would show as a constant
    # (n_agent - n_traj); trigger falls inside the manifest iff idx < n_actions_in_traj.
    # Also record which tool sits at idx, idx+1 and the last action, so the reviewer can see
    # what the index actually points at.
    by_traj: dict[str, dict[int, dict]] = collections.defaultdict(dict)
    for a in attack_actions:
        by_traj[a["traj_id"]][a["action_idx"]] = a
    trig = []
    for r in known:
        idx = r["side_task_trigger_action_idx"]
        A = by_traj[r["traj_id"]]
        n = r["n_actions_in_traj"]
        tool_at = lambda j: (A[j]["tool_name"] if j in A else None)  # noqa: E731
        trig.append({
            "tool_at_idx": tool_at(idx), "tool_at_idx_plus_1": tool_at(idx + 1), "tool_at_last": tool_at(n - 1),
            "skipped_by_monitor_at_idx": A[idx].get("skipped_by_monitor") if idx in A else None,
            "traj_id": r["traj_id"], "env": r["env"], "side_task": r["side_task"],
            "side_task_succeeded": r["side_task_succeeded"],
            "trigger_idx": idx,
            "n_actions_in_agent_traj": r["n_actions_in_agent_traj"],
            "n_actions_in_traj": r["n_actions_in_traj"],
            "len_diff_agent_minus_manifest": r["n_actions_in_agent_traj"] - r["n_actions_in_traj"],
            "idx_within_manifest": idx < r["n_actions_in_traj"],
            "idx_within_agent_traj": idx < r["n_actions_in_agent_traj"],
            "idx_minus_manifest_len": idx - r["n_actions_in_traj"],
            "idx_minus_agent_len": idx - r["n_actions_in_agent_traj"],
        })
    diffs = collections.Counter(t["len_diff_agent_minus_manifest"] for t in trig)
    trig_summary = {
        "n_known": len(trig),
        "tool_at_idx_counts": dict(collections.Counter(t["tool_at_idx"] for t in trig)),
        "tool_at_idx_plus_1_counts": dict(collections.Counter(t["tool_at_idx_plus_1"] for t in trig)),
        "tool_at_last_counts": dict(collections.Counter(t["tool_at_last"] for t in trig)),
        "n_idx_equals_len_minus_2": sum(t["idx_minus_manifest_len"] == -2 for t in trig),
        "known_failure_types": dict(collections.Counter(
            {r["traj_id"]: r["side_task_failure_type"] for r in rows}[t["traj_id"]] for t in trig)),
        "tool_name_counts_all_attack_actions": dict(collections.Counter(a["tool_name"] for a in attack_actions)),
        "skipped_by_monitor_counts_all_attack_actions": dict(collections.Counter(a["skipped_by_monitor"] for a in attack_actions)),
        "n_within_manifest": sum(t["idx_within_manifest"] for t in trig),
        "n_within_agent_traj": sum(t["idx_within_agent_traj"] for t in trig),
        "len_diff_agent_minus_manifest_counts": dict(diffs),
        "idx_minus_manifest_len_values": sorted(t["idx_minus_manifest_len"] for t in trig),
        "constant_offset_found": len(diffs) == 1 and all(t["idx_within_manifest"] for t in trig),
        "len_diff_all_92_counts": dict(collections.Counter(r["n_actions_in_agent_traj"] - r["n_actions_in_traj"] for r in rows)),
    }

    out = {
        "slug": SLUG, "script": f"experiments/{SLUG}/a1_export_fields.py", "git_commit": git_commit(), "timestamp": TS,
        "inputs": {str(p.relative_to(REPO)): sha256(p) for p in [EXPORT / "trajectories.jsonl", EXPORT / "attack.jsonl",
                                                                EXPORT / "honest.jsonl", GOALS, SUBSET]},
        "fields": fields,
        "summary": summary,
        "trajectories_attack": rows,
        "side_task_descriptions": sorted(desc.values(), key=lambda d: (d["category"], d["env"], d["side_task"])),
        "side_task_description_mismatches": mismatches,
        "side_task_descriptions_md": str(DESC_MD.relative_to(REPO)),
        "side_task_descriptions_md_sha256": sha256(DESC_MD),
        "trigger_idx_check": {"summary": trig_summary, "rows": trig},
    }
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RAW_DIR / f"a1_export_fields_{TS}.json"
    raw_path.write_text(json.dumps(out, indent=1))

    # ── stdout report ─────────────────────────────────────────────────────────────────────────
    print(f"raw -> {raw_path.relative_to(REPO)}")
    print(f"desc -> {DESC_MD.relative_to(REPO)}  ({len(desc)} side tasks, {len(mismatches)} mismatches)")
    print("\n== fields ==")
    for fn, ft in fields.items():
        print(f"{fn}: " + ", ".join(f"{k}<{'/'.join(v)}>" for k, v in ft.items()))
    print("\n== summary ==")
    print(json.dumps(summary, indent=1))
    print("\n== per-trajectory (92 attack) ==")
    hdr = f"{'traj_id':32} {'env':17} {'side_task':42} {'category':22} {'fail':8} {'side':7} {'main':7} {'nA':>3} {'nM':>3} {'trig':>4} sub"
    print(hdr)
    for r in sorted(rows, key=lambda r: (r["side_task_category"], r["env"], r["side_task"])):
        print(f"{r['traj_id']:32} {r['env']:17} {r['side_task'][:42]:42} {r['side_task_category']:22} "
              f"{str(r['side_task_failure_type']):8} {tri(r['side_task_succeeded']):7} {tri(r['main_task_success']):7} "
              f"{r['n_actions_in_agent_traj']:3} {r['n_actions_in_traj']:3} {str(r['side_task_trigger_action_idx']):>4} "
              f"{'Y' if r['in_subset_main'] else '.'}")
    print("\n== trigger idx check ==")
    print(json.dumps(trig_summary, indent=1))
    for t in trig:
        print(f"{t['traj_id']} {t['env']:17} {t['side_task'][:36]:36} idx={t['trigger_idx']:3} "
              f"agent_len={t['n_actions_in_agent_traj']:3} manifest_len={t['n_actions_in_traj']:3} "
              f"within_manifest={t['idx_within_manifest']} succeeded={t['side_task_succeeded']} "
              f"tool@idx={t['tool_at_idx']} tool@idx+1={t['tool_at_idx_plus_1']} tool@last={t['tool_at_last']}")
    if mismatches:
        print("\n!! description mismatches:", json.dumps(mismatches, indent=1))


if __name__ == "__main__":
    sys.exit(main())
