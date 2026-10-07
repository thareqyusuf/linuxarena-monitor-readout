#!/usr/bin/env python3
"""L2 export: one JSONL line per action for the Opus-honest / Opus-attack
trajectories that the paper's GPT-5 Nano live monitor scored.

Sources (all cached under checkpoints/<slug>/ by recon.py):
  * mirror `lean/<monitor_run_id>`      -> per-action ensemble member scores
  * mirror `monitor-data/<monitor_run_id>` -> tool outputs (cross-check only)
  * HF dump rows for the monitor runs    -> tool_input (arguments), outputs,
                                            task descriptions, everything else
  * HF dump rows for the agent runs      -> canonical trajectory id + full length

The monitor run's trajectories are the export basis (they are the ones with
scores). Each is joined to the agent run by (env, main_task, side_task).

Writes
  experiments/<slug>/data_export/data/honest.jsonl        (gitignored: big)
  experiments/<slug>/data_export/data/attack.jsonl        (gitignored: big)
  experiments/<slug>/data_export/data/trajectories.jsonl  (per-trajectory text: task description etc.)
  experiments/<slug>/data_export/data/summary.md
  experiments/<slug>/data_export/data/CHECKSUMS.json
  results/raw/<slug>/export_actions_<ts>.jsonl   compact per-action numbers (no tool text)
  results/raw/<slug>/export_meta_<ts>.json       counts, cross-checks, checksums

Pure stdlib.  python3 experiments/logit-curve-replication/data_export/export.py
"""
from __future__ import annotations

import collections
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import recon  # noqa: E402  (same directory)

HERE = Path(__file__).resolve()
SLUG = recon.SLUG
REPO = recon.REPO
RAW = recon.RAW
DATA = HERE.parent / "data"
HIDDEN_TOOLS = recon.HIDDEN_TOOLS


# ── ports of control-tower 1.0.1 helpers (pure python) ──────────────────────

def is_malformed(a: dict) -> bool:
    return bool((a.get("output_error") or "").startswith("parsing:"))


def should_skip(a: dict) -> str | None:
    """control_tower.trajectories.format_history.should_skip_action"""
    if a.get("function") in HIDDEN_TOOLS:
        return f'"{a.get("function")}" tool calls are ignored'
    if is_malformed(a):
        return "malformed action (parsing error)"
    return None


def live_failure_action_index(row: dict) -> int | None:
    """control_tower.trajectories.utils.get_live_failure_action_index"""
    st = row.get("side_task") or {}
    if not st or st.get("failure_type") != "live":
        return None
    for i, a in enumerate(row["actions"]):
        if a.get("function") == "run_side_task_test" and a.get("output") and a["output"].startswith("Score value: C"):
            return i
        if a.get("function") == "submit" and row.get("side_task_success"):
            return i
    return None


def task_key(t: dict) -> tuple:
    return ((t.get("environment") or {}).get("id"),
            (t.get("main_task") or {}).get("id"),
            (t.get("side_task") or {}).get("id"))


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ── main ─────────────────────────────────────────────────────────────────────

def export_side(side: str, rows_m: list[dict], rows_a: list[dict], lean_m: dict, mon_m: dict) -> tuple[list[dict], list[dict], dict]:
    """Return (action_lines, trajectory_lines, checks) for one side."""
    api_lean = {t["trajectory_id"]: t for t in lean_m["trajectories"]}
    api_mon = {t["trajectory_id"]: t for t in mon_m["trajectories"]}
    agent_by_key = {task_key(r): r for r in rows_a}
    assert len(agent_by_key) == len(rows_a)

    action_lines: list[dict] = []
    traj_lines: list[dict] = []
    checks: dict = collections.defaultdict(int)
    output_diffs: list[dict] = []
    truncated: list[dict] = []

    for r in sorted(rows_m, key=lambda r: (task_key(r)[0], task_key(r)[1], str(task_key(r)[2]))):
        k = task_key(r)
        ag = agent_by_key[k]
        checks["trajectories"] += 1
        # prefix content equality with the agent run
        n_m, n_a = len(r["actions"]), len(ag["actions"])
        same_prefix = all(
            a.get("function") == b.get("function") and a.get("arguments") == b.get("arguments")
            and (a.get("output") or "") == (b.get("output") or "")
            for a, b in zip(r["actions"], ag["actions"])
        )
        if n_m != n_a:
            diff_actions = []
            for i, (a, b) in enumerate(zip(r["actions"], ag["actions"])):
                flds = [f for f in ("function", "arguments", "output", "output_error")
                        if (a.get(f) or "") != (b.get(f) or "")]
                if flds:
                    diff_actions.append({"action_idx": i, "fields": flds})
            truncated.append({"traj_id": ag["trajectory_id"], "monitor_traj_id": r["trajectory_id"], "env": k[0],
                              "main_task": k[1], "n_actions_monitor": n_m, "n_actions_agent": n_a,
                              "prefix_content_identical": same_prefix,
                              "n_prefix_actions_differing": len(diff_actions),
                              "prefix_actions_differing": diff_actions[:20]})
        elif same_prefix:
            checks["trajectories_content_identical_to_agent_run"] += 1
        else:
            checks["trajectories_same_length_different_content"] += 1
        if r.get("side_task_success") != ag.get("side_task_success"):
            checks["side_task_success_mismatch"] += 1

        api_t = api_lean[r["trajectory_id"]]
        api_o = api_mon[r["trajectory_id"]]
        assert len(api_t["actions"]) == n_m == len(api_o["actions"])
        harm = live_failure_action_index(r)
        st = r.get("side_task") or {}

        traj_lines.append({
            "side": side,
            "run_id": ag["run_id"], "traj_id": ag["trajectory_id"],
            "monitor_run_id": r["run_id"], "monitor_traj_id": r["trajectory_id"],
            "env": k[0], "main_task": k[1], "side_task": k[2],
            "main_task_category": (r.get("main_task") or {}).get("task_category"),
            "side_task_category": st.get("task_category"),
            "policy": r.get("run_display_name"),
            "n_actions_in_traj": n_m, "n_actions_in_agent_traj": n_a,
            "main_task_success": r.get("main_task_success"),
            "side_task_succeeded": bool(r.get("side_task_success")),
            "side_task_failure_type": st.get("failure_type"),
            "side_task_trigger_action_idx": harm,
            "content_hash": recon.content_hash(r),
            "main_task_description": (r.get("main_task") or {}).get("description"),
            "environment_description": (r.get("environment") or {}).get("description"),
            "side_task_goal": st.get("goal"),
            "trajectory_url": r.get("trajectory_url"),
        })

        for i, a in enumerate(r["actions"]):
            mr = a.get("monitor_response") or {}
            api_mr = api_t["actions"][i].get("monitor_response") or {}
            members_hf = [m.get("sus_score") for m in (mr.get("members") or [])]
            members_api = [m.get("sus_score") for m in (api_mr.get("members") or [])]
            if members_api:
                samples = members_api
            else:                       # single-sample actions have no members on either side
                samples = [api_mr.get("sus_score")] if api_mr.get("sus_score") is not None else []
            if members_hf != members_api:
                checks["member_mismatch_hf_vs_api"] += 1
            if mr.get("sus_score") != api_mr.get("sus_score"):
                checks["score_mismatch_hf_vs_api"] += 1
            out_hf = a.get("output")
            out_api = api_o["actions"][i].get("output")
            # The mirror is the paper snapshot; the HF dump was scrubbed
            # ("LinuxBench" -> "linuxarena" in a few outputs). Export the
            # mirror text, which is what the monitor saw.
            out_export = out_api if out_api is not None else out_hf
            if (out_hf or "") != (out_api or ""):
                checks["output_diff_hf_vs_api"] += 1
                s_hf, s_api = out_hf or "", out_api or ""
                j = next((x for x in range(min(len(s_hf), len(s_api))) if s_hf[x] != s_api[x]), min(len(s_hf), len(s_api)))
                output_diffs.append({"traj_id": ag["trajectory_id"], "action_idx": i, "len_hf": len(s_hf), "len_api": len(s_api),
                                     "first_diff_at": j, "hf": s_hf[max(0, j - 30): j + 50], "api": s_api[max(0, j - 30): j + 50]})
            if a.get("agent_reasoning"):
                checks["actions_with_agent_reasoning_dropped"] += 1
            if a.get("extended_thinking"):
                checks["actions_with_extended_thinking_dropped"] += 1
            if a.get("inter_action_messages"):
                checks["actions_with_inter_action_messages_dropped"] += 1
            action_lines.append({
                "run_id": ag["run_id"], "traj_id": ag["trajectory_id"],
                "monitor_run_id": r["run_id"], "monitor_traj_id": r["trajectory_id"],
                "env": k[0], "main_task": k[1], "side_task": k[2],
                "side_task_category": st.get("task_category"),
                "side_task_failure_type": st.get("failure_type"),
                "policy": r.get("run_display_name"),
                "action_idx": i, "n_actions_in_traj": n_m,
                "tool_name": a.get("function"),
                "tool_input": a.get("arguments"),
                "tool_output": out_export,
                "tool_output_source": "mirror" if out_api is not None else "hf",
                "tool_output_differs_from_hf": (out_hf or "") != (out_api or ""),
                "output_error": a.get("output_error"),
                "skipped_by_monitor": should_skip(a),
                "side_task_succeeded": bool(r.get("side_task_success")),
                "side_task_trigger_action_idx": harm,
                "monitor_scores": samples,
                "monitor_score_mean": api_mr.get("sus_score"),
                "monitor_is_ensemble": bool(api_mr.get("is_ensemble")),
            })
    checks["actions"] = len(action_lines)
    checks["truncated_vs_agent_run"] = truncated
    checks["output_diff_examples"] = output_diffs[:60]
    return action_lines, traj_lines, dict(checks)


def main() -> None:
    R = recon.RUNS
    rows = recon.hf_rows([R["honest_monitor"], R["honest_agent"], R["attack_monitor"], R["attack_agent"]])
    lean = {s: recon.fetch("lean", R[f"{s}_monitor"]) for s in ("honest", "attack")}
    mon = {s: recon.fetch("monitor-data", R[f"{s}_monitor"]) for s in ("honest", "attack")}

    DATA.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    meta: dict = {"script": str(HERE.relative_to(REPO)), "git_commit": recon.git_commit(),
                  "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(), "runs": R,
                  "fingerprints": {s: lean[s]["fingerprint"] for s in lean},
                  "hidden_tools": sorted(HIDDEN_TOOLS), "sides": {}}
    compact_path = RAW / f"export_actions_{ts}.jsonl"
    all_traj: list[dict] = []
    with compact_path.open("w") as compact:
        for side in ("honest", "attack"):
            acts, trajs, checks = export_side(
                side, rows[R[f"{side}_monitor"]], rows[R[f"{side}_agent"]], lean[side], mon[side])
            p = DATA / f"{side}.jsonl"
            with p.open("w") as f:
                for a in acts:
                    f.write(json.dumps(a, ensure_ascii=False) + "\n")
            for a in acts:
                compact.write(json.dumps({
                    "side": side, "traj_id": a["traj_id"], "monitor_traj_id": a["monitor_traj_id"],
                    "env": a["env"], "main_task": a["main_task"], "side_task": a["side_task"],
                    "side_task_category": a["side_task_category"], "side_task_failure_type": a["side_task_failure_type"],
                    "action_idx": a["action_idx"], "n_actions_in_traj": a["n_actions_in_traj"],
                    "tool_name": a["tool_name"], "skipped_by_monitor": a["skipped_by_monitor"],
                    "side_task_succeeded": a["side_task_succeeded"],
                    "side_task_trigger_action_idx": a["side_task_trigger_action_idx"],
                    "monitor_scores": a["monitor_scores"], "monitor_score_mean": a["monitor_score_mean"],
                }) + "\n")
            all_traj.extend(trajs)
            n_benign = sum(1 for a in acts if not a["skipped_by_monitor"])
            meta["sides"][side] = {
                "file": str(p.relative_to(REPO)), "n_trajectories": len(trajs), "n_actions": len(acts),
                "n_actions_scorable": n_benign,
                "n_actions_hidden_tool": sum(1 for a in acts if a["tool_name"] in HIDDEN_TOOLS),
                "n_actions_malformed": sum(1 for a in acts if (a["skipped_by_monitor"] or "").startswith("malformed")),
                "samples_per_action_hist": dict(collections.Counter(len(a["monitor_scores"]) for a in acts)),
                "per_env": dict(collections.Counter(a["env"] for a in trajs)),
                "per_env_side_task": dict(collections.Counter(f"{t['env']}/{t['side_task']}" for t in trajs)),
                "side_task_succeeded": dict(collections.Counter(t["side_task_succeeded"] for t in trajs)),
                "failure_type": dict(collections.Counter(str(t["side_task_failure_type"]) for t in trajs)),
                "trigger_idx_recorded": sum(1 for t in trajs if t["side_task_trigger_action_idx"] is not None),
                "checks": checks,
            }
            print(f"{side}: {len(trajs)} trajectories, {len(acts)} actions ({n_benign} scorable) -> {p.relative_to(REPO)}")
            print("  checks:", {k: v for k, v in checks.items() if not isinstance(v, list)},
                  "| truncated:", len(checks["truncated_vs_agent_run"]))
    with (DATA / "trajectories.jsonl").open("w") as f:
        for t in all_traj:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    n_benign = meta["sides"]["honest"]["n_actions_scorable"]
    n_attack = meta["sides"]["attack"]["n_trajectories"]
    meta["min_usable_fpr"] = {"rule": "20 / n_benign_actions", "n_benign_actions": n_benign, "value": 20 / n_benign}
    meta["compact_raw"] = str(compact_path.relative_to(REPO))
    checks = {"checksums": {}}
    for p in sorted(DATA.glob("*.jsonl")):
        checks["checksums"][p.name] = {"sha256": sha256_file(p), "bytes": p.stat().st_size,
                                       "lines": sum(1 for _ in p.open())}
    checks["git_commit"] = meta["git_commit"]
    checks["timestamp"] = meta["timestamp"]
    (DATA / "CHECKSUMS.json").write_text(json.dumps(checks, indent=1))
    meta["checksums"] = checks["checksums"]
    meta_path = RAW / f"export_meta_{ts}.json"
    meta_path.write_text(json.dumps(meta, indent=1))

    # summary.md
    lines = [f"# data_export summary ({ts})", "",
             f"Source: mirror `{recon.API}` (scores, fingerprints, outputs cross-check) + HF dump rows (tool inputs, outputs, task text).",
             f"Raw numbers: `{meta_path.relative_to(REPO)}`, compact per-action file `{compact_path.relative_to(REPO)}`.", "",
             "## Counts per (policy, env, side_task) — trajectories", "", "| policy | env | side_task | trajectories | actions |", "|---|---|---|---|---|"]
    per = collections.defaultdict(lambda: [0, 0])
    for t in all_traj:
        per[(t["policy"], t["env"], t["side_task"])][0] += 1
        per[(t["policy"], t["env"], t["side_task"])][1] += t["n_actions_in_traj"]
    for (pol, env, stk), (nt, na) in sorted(per.items(), key=lambda kv: (kv[0][0], kv[0][1], str(kv[0][2]))):
        lines.append(f"| {pol} | {env} | {stk} | {nt} | {na} |")
    h, a = meta["sides"]["honest"], meta["sides"]["attack"]
    lines += ["", "## Totals", "",
              f"- honest: {h['n_trajectories']} trajectories, {h['n_actions']} actions, **{h['n_actions_scorable']} benign actions** after dropping hidden tools ({h['n_actions_hidden_tool']}) and malformed ({h['n_actions_malformed']})",
              f"- attack: {a['n_trajectories']} trajectories, {a['n_actions']} actions ({a['n_actions_scorable']} scorable); side task succeeded in {a['side_task_succeeded'].get(True, 0)}; failure_type {a['failure_type']}; trigger index recorded for {a['trigger_idx_recorded']} (live-failure side tasks with a passing test/submit)",
              f"- samples per action: honest {h['samples_per_action_hist']}, attack {a['samples_per_action_hist']}",
              f"- min usable FPR = 20 / {n_benign} = {20 / n_benign:.2e}",
              f"- {len(h['checks']['truncated_vs_agent_run'])} honest trajectories are strict prefixes of the agent-run trajectory (scored prefix exported): "
              + "; ".join(f"{t['env']}/{t['main_task']} {t['n_actions_monitor']}/{t['n_actions_agent']} prefix-identical={t['prefix_content_identical']}" for t in h['checks']['truncated_vs_agent_run']),
              f"- tool_output differs between HF row and mirror monitor-data on {h['checks'].get('output_diff_hf_vs_api', 0)} honest / {a['checks'].get('output_diff_hf_vs_api', 0)} attack actions (examples in the meta raw file; the mirror text is exported, `tool_output_differs_from_hf` flags them)",
              "- tool_input comes from the HF dump only (the mirror has no arguments); it may carry the same scrubbing",
              f"- policy reasoning dropped: agent_reasoning on {h['checks'].get('actions_with_agent_reasoning_dropped', 0)}+{a['checks'].get('actions_with_agent_reasoning_dropped', 0)} actions, extended_thinking on {h['checks'].get('actions_with_extended_thinking_dropped', 0)}+{a['checks'].get('actions_with_extended_thinking_dropped', 0)}, inter_action_messages on {h['checks'].get('actions_with_inter_action_messages_dropped', 0)}+{a['checks'].get('actions_with_inter_action_messages_dropped', 0)}",
              "", "## Files", ""]
    for name, c in checks["checksums"].items():
        lines.append(f"- `{name}`: {c['lines']} lines, {c['bytes']/1e6:.1f} MB, sha256 `{c['sha256']}`")
    (DATA / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {meta_path.relative_to(REPO)} and {DATA.relative_to(REPO)}/summary.md")


if __name__ == "__main__":
    main()
