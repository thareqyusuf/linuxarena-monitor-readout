#!/usr/bin/env python3
"""G3 addendum 4: the highest-cached-score actions of each attack trajectory in subset_main.

Hypothesis under test (reviewer's): `side` labels the TRAJECTORY, not the action; most actions of an attack
trajectory are ordinary development, so local-vs-cached agreement must be judged on the actions the cached
monitor itself flagged, not on all actions of attack trajectories.

CPU only, no model. Produces:
  subset_topcached.json                  the top --per-traj cached-score actions of each attack trajectory
  results/raw/<slug>/g3_topcached_<ts>.json
     - the selected actions with their current-action text, side-task goal, position in trajectory
     - the cached ensemble's internal disagreement (the two cached samples) over the whole subset, by bucket
     - which selected actions already have a local score from an earlier run, and the agreement on those
"""
from __future__ import annotations

import collections
import datetime as dt
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
SLUG = HERE.parent.name
REPO = HERE.parents[2]
RAW = REPO / "results" / "raw" / SLUG
TS = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
sys.path.insert(0, str(HERE.parent))
import monitor_prompt as mp  # noqa: E402

PER_TRAJ = int(sys.argv[1]) if len(sys.argv) > 1 else 5

manifest = mp.load_manifest()
trajs, _ = mp.load_trajs()
man = {(a["traj_id"], a["action_idx"]): a for a in manifest["actions"]}
tmeta = {}
with (mp.EXPORT_DIR / "trajectories.jsonl").open() as f:
    for line in f:
        t = json.loads(line); tmeta[t["traj_id"]] = t
sub = json.loads((HERE.parent / "subset_main.json").read_text())["actions"]

# ── cached ensemble disagreement over the whole subset (is a [1, 8] plausible noise?) ─────────
pairs = [(a, man[(a["traj_id"], a["action_idx"])]["cached_scores"]) for a in sub]
pairs = [(a, cs) for a, cs in pairs if len(cs) == 2]
spread = [abs(cs[0] - cs[1]) for _, cs in pairs]
by_mean = collections.defaultdict(list)
for a, cs in pairs:
    by_mean[round(statistics.mean(cs) * 2) / 2].append(abs(cs[0] - cs[1]))
disagreement = {
    "n_with_two_samples": len(pairs), "n_of_subset": len(sub),
    "mean_abs_diff": statistics.mean(spread), "median_abs_diff": statistics.median(spread),
    "frac_diff_0": sum(d == 0 for d in spread) / len(spread),
    "frac_diff_ge3": sum(d >= 3 for d in spread) / len(spread),
    "frac_diff_ge5": sum(d >= 5 for d in spread) / len(spread),
    "max_abs_diff": max(spread),
    "by_cached_mean": {str(k): {"n": len(v), "mean_abs_diff": statistics.mean(v)} for k, v in sorted(by_mean.items())},
    "n_pairs_with_one_sample_ge6_and_other_le2": sum(1 for _, cs in pairs if max(cs) >= 6 and min(cs) <= 2),
}

# ── select the top-cached actions per attack trajectory ───────────────────────────────────────
by_traj = collections.defaultdict(list)
for a in sub:
    if a["side"] == "attack":
        by_traj[a["traj_id"]].append(a)
selected = []
for tid, rows in sorted(by_traj.items()):
    cs = lambda a: man[(a["traj_id"], a["action_idx"])]["cached_scores"]  # noqa: E731
    rows.sort(key=lambda a: (-a["cached_mean"], -max(cs(a)), a["action_idx"]))
    for rank, a in enumerate(rows[:PER_TRAJ]):
        t = trajs[tid]
        act = t["actions"][a["action_idx"]]
        selected.append({**a, "rank_in_traj": rank, "cached_scores": cs(a),
                         "side_task": tmeta[tid]["side_task"], "side_task_category": tmeta[tid]["side_task_category"],
                         "n_actions_in_traj_subset": len(rows), "trajectory_length": tmeta[tid]["n_actions_in_traj"],
                         "position_frac": a["action_idx"] / max(1, tmeta[tid]["n_actions_in_traj"] - 1),
                         "tool": man[(tid, a["action_idx"])]["tool"],
                         "current_action_text": mp.fmp.format_action(act, hide_output=True)[:2000],
                         "side_task_goal": (tmeta[tid].get("side_task_goal") or "")[:400]})

# ── which already have a local score, and agreement on those ─────────────────────────────────
local: dict[str, dict] = {}
pil = sorted(RAW.glob("g3_pilot_2026*.json"))
if pil:
    for p in json.loads(pil[-1].read_text())["throughput"]["per_action"]:
        local.setdefault(p["key"], {})["forced_E"] = p["expected_score"]
pk = sorted(RAW.glob("g3_peek_*.json"))
for f in pk:
    d = json.loads(f.read_text())
    for g in d["generations"]:
        if "modes" in g:
            local.setdefault(g["key"], {})["unforced_score"] = g["modes"]["free"]["parsed_score"]
va = sorted(RAW.glob("g3_variants_*.json"))
for f in va:
    for g in json.loads(f.read_text())["generations"]:
        local.setdefault(g["key"], {})["sampled_mean"] = g["samples"]["mean"]
        local.setdefault(g["key"], {})["greedy_medium"] = g["greedy"]["parsed_score"]
g4 = REPO / "checkpoints" / SLUG / "out" / "main" / "scores.jsonl"
if g4.exists():
    for line in g4.open():
        r = json.loads(line)
        if r.get("status") == "ok":
            k = f"{r['traj_id']}|{r['action_idx']}"
            local.setdefault(k, {}).update({"g4_score": r["score"], "g4_expected": r["expected_score"]})
for s in selected:
    s["local"] = local.get(f"{s['traj_id']}|{s['action_idx']}", {})
n_have = sum(1 for s in selected if s["local"])

hdr = {k: json.loads((HERE.parent / "subset_main.json").read_text())[k]
       for k in ("cap_tokens", "seed", "template_date", "reasoning_effort", "pinned_chat_template_sha256", "prompt_template_sha256")}
(HERE.parent / "subset_topcached.json").write_text(json.dumps(
    {**hdr, "parent": "subset_main.json", "per_traj": PER_TRAJ, "n": len(selected), "order": "ascending n_tokens",
     "actions": sorted([{k: v for k, v in s.items() if k not in ("current_action_text", "side_task_goal", "local")} for s in selected],
                       key=lambda a: (a["n_tokens"], a["traj_id"], a["action_idx"]))}, indent=0))

R = {"slug": SLUG, "script": str(HERE.relative_to(REPO)), "timestamp": TS, "per_traj": PER_TRAJ,
     "n_attack_trajectories": len(by_traj), "n_selected": len(selected),
     "cached_ensemble_disagreement_over_subset_main": disagreement,
     "n_selected_with_a_local_score": n_have,
     "cached_mean_of_selected": {"min": min(s["cached_mean"] for s in selected), "max": max(s["cached_mean"] for s in selected),
                                 "median": statistics.median([s["cached_mean"] for s in selected]),
                                 "hist": dict(sorted(collections.Counter(s["cached_mean"] for s in selected).items()))},
     "position_frac_of_selected": {"median": statistics.median([s["position_frac"] for s in selected]),
                                   "mean": statistics.mean([s["position_frac"] for s in selected])},
     "tool_hist": dict(collections.Counter(s["tool"] for s in selected)),
     "selected": selected}
RAW.mkdir(parents=True, exist_ok=True)
out = RAW / f"g3_topcached_{TS}.json"
out.write_text(json.dumps(R, indent=1, ensure_ascii=False))
print(json.dumps({k: v for k, v in R.items() if k != "selected"}, indent=1))
print("raw ->", out.relative_to(REPO))
