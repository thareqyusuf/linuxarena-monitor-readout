#!/usr/bin/env python3
"""Write run_order.json: the fixed, committed order of every action in every G4' stage (TASK-ver3).

Stages, in chain order:
  low_cached        subset_low_cached.json        original      ascending n_tokens
  consistent_high   subset_consistent_high.json   original      ascending n_tokens (same order as pass 1)
  main              subset_main.json minus the keys already in out/main_pass1 (the halted pass 1),
                                                  original      random.Random(seed).shuffle of the ascending list
  cf_neutralized    subset_counterfactual.json    neutralized   ascending
  cf_random_scrub   subset_counterfactual.json    random_scrub  ascending
  sampled_k4        subset_sampled.json           original      ascending, n_samples=4 (optional stage)
  enrich            subset_payload_enrich.json    original      ascending — only with --enrich-out, only if the file exists

"ascending" = sorted by (n_tokens, traj_id, action_idx), the TASK-ver2 runner's default.
Each stage entry: stage, subset, render, out, order, seed, n_samples, n, actions[{index, key, traj_id, action_idx, n_tokens}].
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import random
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve()
SLUG = HERE.parent.name
REPO = HERE.parents[2]
OUT = REPO / "checkpoints" / SLUG / "out"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def ascending(actions):
    return sorted(actions, key=lambda a: (a["n_tokens"], a["traj_id"], a["action_idx"]))


def entries(actions):
    return [{"index": i, "key": f"{a['traj_id']}|{a['action_idx']}", "traj_id": a["traj_id"], "action_idx": a["action_idx"],
             "n_tokens": a["n_tokens"]} for i, a in enumerate(actions)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=HERE.parent / "run_order.json")
    ap.add_argument("--enrich-out", type=Path, default=None, help="write ONLY the enrich stage to this file (chain, after main)")
    ap.add_argument("--exclude-from", type=Path, default=OUT / "main_pass1" / "acts_index.json")
    args = ap.parse_args()
    git = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    subsets = {}

    def load(name):
        p = HERE.parent / name
        subsets[name] = {"sha256": sha(p), "n": None}
        d = json.loads(p.read_text())
        subsets[name]["n"] = len(d["actions"])
        return d["actions"]

    stages = []
    if args.enrich_out:
        p = HERE.parent / "subset_payload_enrich.json"
        assert p.exists(), "subset_payload_enrich.json does not exist"
        acts = ascending(load("subset_payload_enrich.json"))
        stages.append({"stage": "enrich", "subset": "subset_payload_enrich.json", "render": "original", "out": "out/enrich",
                       "order": "ascending (n_tokens, traj_id, action_idx)", "seed": None, "n_samples": 0, "n": len(acts), "actions": entries(acts)})
        target = args.enrich_out
    else:
        lc = ascending(load("subset_low_cached.json"))
        stages.append({"stage": "low_cached", "subset": "subset_low_cached.json", "render": "original", "out": "out/low_cached",
                       "order": "ascending (n_tokens, traj_id, action_idx)", "seed": None, "n_samples": 0, "n": len(lc), "actions": entries(lc)})
        ch = ascending(load("subset_consistent_high.json"))
        stages.append({"stage": "consistent_high", "subset": "subset_consistent_high.json", "render": "original", "out": "out/consistent_high",
                       "order": "ascending (n_tokens, traj_id, action_idx) — identical to pass 1", "seed": None, "n_samples": 0, "n": len(ch), "actions": entries(ch)})
        main_all = ascending(load("subset_main.json"))
        idx = json.loads(args.exclude_from.read_text())
        excluded = {k.rsplit("|", 1)[0] for keys in idx["shards"].values() for k in keys}       # "traj|idx|original" -> "traj|idx"
        rem = [a for a in main_all if f"{a['traj_id']}|{a['action_idx']}" not in excluded]
        assert len(rem) == len(main_all) - len(excluded), (len(rem), len(main_all), len(excluded))
        random.Random(args.seed).shuffle(rem)
        stages.append({"stage": "main", "subset": "subset_main.json", "render": "original", "out": "out/main",
                       "order": f"random.Random({args.seed}).shuffle applied to the ascending (n_tokens, traj_id, action_idx) list of subset_main minus the pass-1 keys",
                       "seed": args.seed, "n_samples": 0, "n": len(rem), "excluded_keys": sorted(excluded), "excluded_from": str(args.exclude_from.relative_to(REPO)),
                       "excluded_index_sha256": sha(args.exclude_from), "actions": entries(rem)})
        cf = ascending(load("subset_counterfactual.json"))
        for render in ("neutralized", "random_scrub"):
            stages.append({"stage": f"cf_{render}", "subset": "subset_counterfactual.json", "render": render, "out": f"out/cf_{render}",
                           "order": "ascending (n_tokens, traj_id, action_idx)", "seed": None, "n_samples": 0, "n": len(cf), "actions": entries(cf)})
        sp = ascending(load("subset_sampled.json"))
        stages.append({"stage": "sampled_k4", "subset": "subset_sampled.json", "render": "original", "out": "out/sampled_k4",
                       "order": "ascending (n_tokens, traj_id, action_idx)", "seed": None, "n_samples": 4, "optional": True, "n": len(sp), "actions": entries(sp)})
        target = args.out
    doc = {"slug": SLUG, "script": str(HERE.relative_to(REPO)), "git_commit": git, "created": dt.datetime.now(dt.timezone.utc).isoformat(),
           "seed": args.seed, "subsets": subsets, "chain_order": [s["stage"] for s in stages],
           "enrich_note": "stage 'enrich' is added by the chain via --enrich-out only if subset_payload_enrich.json exists when main finishes",
           "stages": stages}
    target.write_text(json.dumps(doc, indent=0))
    for s in stages:
        first = [e["key"][:8] + "#" + str(e["action_idx"]) for e in s["actions"][:5]]
        print(f"{s['stage']:16s} {s['render']:13s} n={s['n']:5d}  first5={first}  ->", s["out"])
    print("wrote", target.relative_to(REPO), f"{target.stat().st_size / 1024:.0f} KiB", "sha256", sha(target)[:16])


if __name__ == "__main__":
    main()
