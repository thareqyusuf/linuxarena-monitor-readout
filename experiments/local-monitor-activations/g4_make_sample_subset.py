#!/usr/bin/env python3
"""Build subset_sampled.json: ~150 actions from subset_main.json for the OPTIONAL post-run k-sample step.

Stratified over side x cached-score bucket ([0,1], (1,3], (3,6], (6,10]), seed 0, shortest-first within a
cell so the extra decode cost is bounded. Run later as:

  python experiments/<slug>/run_monitor.py ... --subset experiments/<slug>/subset_sampled.json \
      --n-samples 4 --temperature 1.0 --gate-at 0 --out checkpoints/<slug>/out/sampled

CPU only, no model. Writes results/raw/<slug>/g4_sample_subset_<ts>.json alongside.
"""
from __future__ import annotations

import collections
import datetime as dt
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve()
SLUG = HERE.parent.name
REPO = HERE.parents[2]
RAW = REPO / "results" / "raw" / SLUG
TS = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
N_WANT = 150
SEED = 0
BUCKETS = [(0, 1), (1, 3), (3, 6), (6, 10)]


def bucket(x: float) -> str:
    for lo, hi in BUCKETS:
        if (x >= lo if lo == 0 else x > lo) and x <= hi:
            return f"{lo}-{hi}"
    return "?"


main = json.loads((HERE.parent / "subset_main.json").read_text())
acts = list(main["actions"])
rng = random.Random(SEED)
cells: dict[tuple, list] = collections.defaultdict(list)
for a in acts:
    cells[(a["side"], bucket(a["cached_mean"]))].append(a)
for v in cells.values():
    v.sort(key=lambda a: (a["n_tokens"], a["traj_id"], a["action_idx"]))
    rng.shuffle(v)
keys = sorted(cells)
per = N_WANT // len(keys)
chosen: list = []
for k in keys:
    chosen += cells[k][:per]
    cells[k] = cells[k][per:]
left = [a for k in keys for a in cells[k]]
rng.shuffle(left)
chosen += left[: max(0, N_WANT - len(chosen))]
chosen.sort(key=lambda a: (a["n_tokens"], a["traj_id"], a["action_idx"]))

hdr = {k: main[k] for k in ("cap_tokens", "seed", "template_date", "reasoning_effort", "pinned_chat_template_sha256", "prompt_template_sha256")}
hdr.update({"parent": "subset_main.json", "purpose": "optional post-run k-sample step", "n": len(chosen), "order": "ascending n_tokens"})
(HERE.parent / "subset_sampled.json").write_text(json.dumps({**hdr, "actions": chosen}, indent=0))
summary = {"slug": SLUG, "script": str(HERE.relative_to(REPO)), "timestamp": TS, "n": len(chosen), "seed": SEED, "buckets": BUCKETS,
           "cells": {f"{s}/{b}": n for (s, b), n in sorted(collections.Counter((a["side"], bucket(a["cached_mean"])) for a in chosen).items())},
           "by_side": dict(collections.Counter(a["side"] for a in chosen)),
           "n_placeholder": sum(a["placeholder"] for a in chosen),
           "n_trajectories": len({a["traj_id"] for a in chosen}),
           "n_tokens": {"min": min(a["n_tokens"] for a in chosen), "median": sorted(a["n_tokens"] for a in chosen)[len(chosen) // 2],
                        "max": max(a["n_tokens"] for a in chosen), "sum": sum(a["n_tokens"] for a in chosen)},
           "keys": [f"{a['traj_id']}|{a['action_idx']}" for a in chosen]}
RAW.mkdir(parents=True, exist_ok=True)
(RAW / f"g4_sample_subset_{TS}.json").write_text(json.dumps(summary, indent=1))
print(json.dumps({k: v for k, v in summary.items() if k != "keys"}, indent=1))
