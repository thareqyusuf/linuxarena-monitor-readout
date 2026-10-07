#!/usr/bin/env python3
"""G3 prep: pick the 30-action pilot from subset_main.json (CPU, no model).

  subset_pilot.json        30 actions: 3 per length decile of subset_main, alternating sides (15/15),
                           preferring, in each decile, one action whose current action has a neutralization
                           event, so that >= 10 of the 30 can be re-rendered `neutralized` with a visible change.
  subset_pilot_noise.json  20 of those 30 (2 per decile) for the chunk-size noise-floor re-run.
  subset_pilot_neut.json   10 of the 30 with >= 1 neutralization event (action stratum first) for the
                           `neutralized` render check.
Deterministic (seed 0). Writes the choice and its composition into the raw file
results/raw/<slug>/g3_pilot_subset_<ts>.json.
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

main = json.loads((HERE.parent / "subset_main.json").read_text())
g2 = json.loads((RAW / main["raw"]).read_text()) if (RAW / main["raw"]).exists() else None
raw_files = sorted(RAW.glob("g2_prep_*.json"))
g2 = json.loads(raw_files[-1].read_text())
pa = g2["per_action"]
neut = {(t, i): (h, a) for t, i, h, a in zip(pa["traj_id"], pa["action_idx"], pa["neut_hist_events"], pa["neut_action_events"])}

acts = sorted(main["actions"], key=lambda a: (a["n_tokens"], a["traj_id"], a["action_idx"]))
for a in acts:
    a["neut_hist_events"], a["neut_action_events"] = neut[(a["traj_id"], a["action_idx"])]
n = len(acts)
rng = random.Random(0)
pilot = []
for d in range(10):
    dec = acts[d * n // 10:(d + 1) * n // 10]
    want = ["attack", "honest", "attack"] if d % 2 == 0 else ["honest", "attack", "honest"]
    chosen = []
    for k, side in enumerate(want):
        pool = [a for a in dec if a["side"] == side and a not in chosen]
        if k == 0:
            pref = [a for a in pool if a["neut_action_events"] > 0]
            pool = pref or pool
        chosen.append(rng.choice(pool))
    pilot += chosen
for a in pilot:
    a["decile"] = next(d for d in range(10) if a in acts[d * n // 10:(d + 1) * n // 10])
noise = [a for d in range(10) for a in [x for x in pilot if x["decile"] == d][:2]]
neut_first = sorted(pilot, key=lambda a: (-(a["neut_action_events"] > 0), -(a["neut_hist_events"] + a["neut_action_events"]), a["n_tokens"]))
neut10 = [a for a in neut_first if a["neut_hist_events"] + a["neut_action_events"] > 0][:10]

hdr = {k: main[k] for k in ("cap_tokens", "seed", "template_date", "reasoning_effort", "pinned_chat_template_sha256", "prompt_template_sha256")}
hdr["parent"] = "subset_main.json"; hdr["g2_raw"] = raw_files[-1].name
for name, rows in (("subset_pilot.json", pilot), ("subset_pilot_noise.json", noise), ("subset_pilot_neut.json", neut10)):
    (HERE.parent / name).write_text(json.dumps({**hdr, "n": len(rows), "actions": rows}, indent=0))
summary = {"slug": SLUG, "script": str(HERE.relative_to(REPO)), "timestamp": TS, "g2_raw": raw_files[-1].name,
           "pilot": {"n": len(pilot), "by_side": dict(collections.Counter(a["side"] for a in pilot)),
                     "n_tokens": [a["n_tokens"] for a in pilot], "n_with_any_neut_event": sum(a["neut_hist_events"] + a["neut_action_events"] > 0 for a in pilot),
                     "n_with_action_neut_event": sum(a["neut_action_events"] > 0 for a in pilot),
                     "n_placeholder": sum(a["placeholder"] for a in pilot), "n_trajectories": len({a["traj_id"] for a in pilot}),
                     "actions": [{k: a[k] for k in ("traj_id", "action_idx", "side", "env", "n_tokens", "cached_mean", "decile", "neut_action_events", "neut_hist_events")} for a in pilot]},
           "noise": {"n": len(noise), "n_tokens": [a["n_tokens"] for a in noise]},
           "neut": {"n": len(neut10), "n_with_action_event": sum(a["neut_action_events"] > 0 for a in neut10), "keys": [f"{a['traj_id']}|{a['action_idx']}" for a in neut10]}}
RAW.mkdir(parents=True, exist_ok=True)
(RAW / f"g3_pilot_subset_{TS}.json").write_text(json.dumps(summary, indent=1))
print(json.dumps({k: v for k, v in summary["pilot"].items() if k != "actions"}, indent=0))
print(json.dumps(summary["noise"]), json.dumps(summary["neut"]))
