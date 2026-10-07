"""P1 — lock the trajectory split before any activation is loaded.

Test = 1/3 of the 59 subset_main trajectories, stratified by side x side_task_category (honest is one
stratum), random.Random(0): within each stratum shuffle and take round(n/3). Train = the rest; 5 grouped-CV
folds over train trajectories, same stratification, round-robin after an in-stratum shuffle.
Output experiments/<slug>/split.json (committed) + results/raw/<slug>/p1_split_<ts>.json (same content + counts).
"""
from __future__ import annotations

import collections, json, random, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

SPLIT = C.HERE / "split.json"


def main():
    stamp = C.ts(); rows, meta = C.build_table()
    traj = {}
    for r in rows:
        traj.setdefault(r["traj_id"], {"side": r["side"], "category": r["side_task_category"], "env": r["env"], "n_actions_in_subset": 0, "n_level2": 0, "n_usable": 0})
        traj[r["traj_id"]]["n_actions_in_subset"] += 1
        if r["score"] is not None and not r["placeholder_current"]:
            traj[r["traj_id"]]["n_usable"] += 1; traj[r["traj_id"]]["n_level2"] += int(r["is_pos"])
    strata = collections.defaultdict(list)
    for t in sorted(traj):
        strata[f"{traj[t]['side']}|{traj[t]['category']}"].append(t)
    rng = random.Random(0); test, train = [], []
    for s in sorted(strata):
        ids = sorted(strata[s]); rng.shuffle(ids); k = round(len(ids) / 3)
        test += ids[:k]; train += ids[k:]
    folds = {i: [] for i in range(5)}
    fold_of = {}
    for s in sorted(strata):
        ids = sorted(t for t in strata[s] if t in set(train)); rng.shuffle(ids)
        for j, t in enumerate(ids):
            folds[j % 5].append(t); fold_of[t] = j % 5
    def counts(ids):
        return {"n_traj": len(ids), "n_honest_traj": sum(traj[t]["side"] == "honest" for t in ids), "n_attack_traj": sum(traj[t]["side"] == "attack" for t in ids),
                "n_pos_traj": sum(traj[t]["n_level2"] > 0 for t in ids), "n_level2": sum(traj[t]["n_level2"] for t in ids),
                "n_usable_honest": sum(traj[t]["n_usable"] for t in ids if traj[t]["side"] == "honest"), "envs": sorted({traj[t]["env"] for t in ids})}
    split = {"slug": C.SLUG, "script": f"experiments/{C.SLUG}/p1_split.py", "timestamp": stamp, "git_commit": C.git_commit(), "seed": 0,
             "rule": "test = round(n/3) per stratum (side x side_task_category), random.Random(0); 5 grouped folds on train, round-robin within stratum",
             "strata": {s: sorted(v) for s, v in strata.items()}, "test": sorted(test), "train": sorted(train), "folds": {str(i): sorted(v) for i, v in folds.items()},
             "fold_of": fold_of, "trajectories": traj,
             "counts": {"test": counts(test), "train": counts(train), **{f"fold{i}": counts(v) for i, v in folds.items()}}}
    SPLIT.write_text(json.dumps(split, indent=1)); print("split ->", SPLIT.relative_to(C.REPO))
    C.write_raw("p1_split", split | {"split_json_sha256": C.sha256(SPLIT)}, stamp)
    for k, v in split["counts"].items():
        print(f"  {k:6} {v}")


if __name__ == "__main__":
    main()
