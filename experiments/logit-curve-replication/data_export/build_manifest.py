#!/usr/bin/env python3
"""L4.2: write gpu/manifest.json — everything the pod needs to know which
actions to score and how to check its inputs, and nothing else.

    run ids (agent + monitor), mirror fingerprints (L1), data file checksums (L2),
    per-trajectory content hashes, the full list of (traj_id, action_idx) the
    live monitor scores (hidden tools and malformed actions excluded), the
    control-tower prompt sha256 per action (from verify_prompt_format.py), and a
    `pilot` subset of 200 (100 honest / 100 attack) spread over the 10 envs and
    over early / middle / late action positions.

Pure stdlib.  python3 experiments/logit-curve-replication/data_export/build_manifest.py
"""
from __future__ import annotations

import collections
import datetime as dt
import json
import random
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
SLUG = HERE.parents[1].name
REPO = HERE.parents[3]
RAW = REPO / "results" / "raw" / SLUG
DATA = HERE.parent / "data"
GPU = HERE.parents[1] / "gpu"
sys.path.insert(0, str(GPU))
import format_monitor_prompt as fmp  # noqa: E402

PILOT_PER_SIDE = 100
ENVS = ["auto_workflow", "clinical_trial", "file_indexing", "media_processing", "model_registry",
        "shared_dotfiles", "slack_clone", "tiktok", "user_tracking", "web_scraping"]


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "unknown"


def position_bucket(idx: int, n: int) -> str:
    f = idx / max(1, n - 1)
    return "early" if f < 1 / 3 else ("late" if f >= 2 / 3 else "middle")


def pick_pilot(items: list[dict], rng: random.Random) -> list[dict]:
    """100 actions: 10 per env, cycling early/middle/late, at most one action per trajectory
    until every trajectory of the env has been used."""
    by_env: dict[str, dict[str, list[dict]]] = collections.defaultdict(lambda: collections.defaultdict(list))
    for it in items:
        by_env[it["env"]][it["position"]].append(it)
    picked: list[dict] = []
    for env in ENVS:
        pools = {k: rng.sample(v, len(v)) for k, v in by_env[env].items()}
        used_traj: set[str] = set()
        got = 0
        order = ["early", "middle", "late"]
        k = 0
        guard = 0
        while got < PILOT_PER_SIDE // len(ENVS) and guard < 10000:
            guard += 1
            pos = order[k % 3]; k += 1
            pool = pools.get(pos, [])
            cand = next((x for x in pool if x["traj_id"] not in used_traj), None)
            if cand is None:
                if not any(x["traj_id"] not in used_traj for p in pools.values() for x in p):
                    used_traj.clear()          # every trajectory used once -> allow repeats
                cand = next((x for x in pool if x["traj_id"] not in used_traj), None)
                if cand is None:
                    continue
            pool.remove(cand)
            used_traj.add(cand["traj_id"])
            picked.append(cand)
            got += 1
    return picked


def main() -> None:
    rng = random.Random(0)
    trajs = fmp.load_export(DATA)
    verify = sorted(RAW.glob("l4_prompt_verify_*.json"))
    hashes: dict[tuple[str, int], dict] = {}
    verify_file = None
    if verify:
        verify_file = verify[-1]
        for p in json.loads(verify_file.read_text())["per_action"]:
            hashes[(p["traj_id"], p["action_idx"])] = p
    recon = sorted(RAW.glob("l1_recon_*.json"))[-1]
    fps = json.loads(recon.read_text())["fingerprints"]
    checksums = json.loads((DATA / "CHECKSUMS.json").read_text())

    items: list[dict] = []
    trajectories: list[dict] = []
    for traj_id, t in sorted(trajs.items()):
        n = len(t["actions"])
        trajectories.append({"traj_id": traj_id, "monitor_traj_id": t["monitor_traj_id"], "side": t["side"],
                             "env": t["env"], "main_task": t["main_task"], "side_task": t["side_task"],
                             "side_task_category": t.get("side_task_category"), "n_actions": n,
                             "n_actions_in_agent_traj": t["n_actions_in_agent_traj"],
                             "side_task_succeeded": t["side_task_succeeded"],
                             "side_task_trigger_action_idx": t["side_task_trigger_action_idx"],
                             "content_hash": t["content_hash"]})
        for a in t["actions"]:
            if a["skipped_by_monitor"]:
                continue
            h = hashes.get((traj_id, a["index"]))
            items.append({"traj_id": traj_id, "action_idx": a["index"], "side": t["side"], "env": t["env"],
                          "position": position_bucket(a["index"], n), "tool": a["function"],
                          "cached_scores": a["monitor_scores"], "cached_mean": a["monitor_score_mean"],
                          "prompt_sha256": (h["sha256"] if h else None), "prompt_chars": (h["chars"] if h else None)})
    pilot = pick_pilot([x for x in items if x["side"] == "honest"], rng) + pick_pilot([x for x in items if x["side"] == "attack"], rng)
    pilot_keys = {(x["traj_id"], x["action_idx"]) for x in pilot}

    manifest = {
        "slug": SLUG, "built_by": str(HERE.relative_to(REPO)), "git_commit": git_commit(),
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "runs": {"honest": {"agent_run_id": fps["honest_agent"]["run_id"], "monitor_run_id": fps["honest_monitor"]["run_id"],
                            "agent_fingerprint": fps["honest_agent"]["server_fingerprint"],
                            "monitor_fingerprint": fps["honest_monitor"]["server_fingerprint"]},
                 "attack": {"agent_run_id": fps["attack_agent"]["run_id"], "monitor_run_id": fps["attack_monitor"]["run_id"],
                            "agent_fingerprint": fps["attack_agent"]["server_fingerprint"],
                            "monitor_fingerprint": fps["attack_monitor"]["server_fingerprint"]}},
        "mirror": "https://repro.linuxarena.ai/api/public",
        "data_files": checksums["checksums"],
        "prompt_template_sha256": __import__("hashlib").sha256((GPU / "prompts" / "old_action.md").read_bytes()).hexdigest(),
        "prompt_hashes_from": (str(verify_file.relative_to(REPO)) if verify_file else None),
        "n_trajectories": len(trajectories), "n_actions": len(items),
        "n_actions_by_side": dict(collections.Counter(x["side"] for x in items)),
        "pilot_size": len(pilot),
        "pilot_by_side_env": dict(collections.Counter(f"{x['side']}/{x['env']}" for x in pilot)),
        "pilot_by_side_position": dict(collections.Counter(f"{x['side']}/{x['position']}" for x in pilot)),
        "pilot_distinct_trajectories": len({x["traj_id"] for x in pilot}),
        "trajectories": trajectories,
        "actions": [{**x, "pilot": (x["traj_id"], x["action_idx"]) in pilot_keys} for x in items],
    }
    p = GPU / "manifest.json"
    p.write_text(json.dumps(manifest, indent=0))
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("trajectories", "actions", "data_files")}, indent=1))
    print(f"wrote {p.relative_to(REPO)} ({p.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
