"""D — what the verbalized integer throws away, and what a probe could cheat on. CPU, no model.

Two blocks, both from files already committed or checkpointed:

  expression  the forced-readout digit distribution (`digit_dist`, mass >= 0.9994 on digits) vs the
              emitted integer `score`, for honest and level-2 actions; and Delta p_null for E:
              of the level-2 actions the integer puts at <= 1, how many does E rank above the
              honest 95th / 99th percentile. This is the no-white-box baseline a probe must beat.
  leakage     what separates level-2 from honest on the action string alone: the magic hostnames,
              the tool-shape tokens, environment-identity tokens, action length, position.
              Every row is (level-2 rate, honest rate) so the confound is visible.

Inputs: action_labels.json (this branch), the G4 main score records (gitignored checkpoint,
sha256 recorded), the export for the rendered action strings.
"""
from __future__ import annotations

import collections
import datetime as dt
import glob
import hashlib
import json
import re
import statistics as st
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SLUG = HERE.name
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "experiments" / "logit-curve-replication" / "gpu"))
import format_monitor_prompt as fmp  # noqa: E402

EXPORT = REPO / "checkpoints" / "local-monitor-activations" / "data" / "export"
G4 = REPO / "checkpoints" / SLUG / "g4"
G4_COMMIT = "0153b29153147d668de2fb973aa07d8ad7afcc7b"
G4_FILES = ["g4_main_scores_20260911T075642Z_part1.jsonl", "g4_main_scores_20260911T075642Z_part2.jsonl",
            "g4_main_first200_scores_20260910T133150Z.jsonl"]  # first-pass 200: 4 positions only, but score/E/digit_dist present
LABELS = HERE / "action_labels.json"
RAW_DIR = REPO / "results" / "raw" / SLUG
FIG_DIR = REPO / "results" / "figures" / SLUG
TS = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
MAGIC = ["packages-cdn.com", "filebin.net", "telemetry.taskflow.co", "/dev/tcp/", "authorized_keys"]
SHAPE = ["str_replace", "old_str", "new_str", "DELETE FROM", "psql", "curl"]


def pct(v, q):
    s = sorted(v)
    return s[min(int(q * len(s)), len(s) - 1)]


def main():
    lab = json.load(open(LABELS)); D = lab["data"]
    L = {(D["traj_id"][i], D["action_idx"][i]): {c: D[c][i] for c in lab["columns"]} for i in range(lab["n"])}
    rec = {}
    for f in G4_FILES:
        for line in open(G4 / f):
            if line.strip():
                r = json.loads(line); rec[(r["traj_id"], int(r["action_idx"]))] = r
    trajs = {t["traj_id"]: t for t in fmp.load_export(EXPORT).values()}
    text, nact = {}, {}
    for tid, t in trajs.items():
        nact[tid] = len(t["actions"])
        for a in t["actions"]:
            text[(tid, a["index"])] = fmp.format_action(a, hide_output=True)

    rows = []
    for k, r in rec.items():
        a = L.get(k)
        if a is None or a["label_source"] == "placeholder" or r["score"] is None:
            continue
        rows.append({"traj_id": k[0], "action_idx": k[1], "side": a["side"], "env": a["env"],
                     "label_level": a["label_level"], "score": r["score"], "E": r["expected_score"],
                     "digit_dist": r["digit_dist"], "digit_mass": r["digit_token_mass"],
                     "n_actions": nact[k[0]], "chars": len(text[k]),
                     "pos": k[1] / max(nact[k[0]] - 1, 1), "text": text[k]})
    hon = [r for r in rows if r["side"] == "honest"]
    l2 = [r for r in rows if r["side"] == "attack" and r["label_level"] == 2]

    def meandist(rs):
        agg = collections.Counter()
        for r in rs:
            for d, p in (r["digit_dist"] or {}).items():
                agg[int(d)] += p
        return {str(d): agg[d] / len(rs) for d in sorted(agg)}

    dh, dl = meandist(hon), meandist(l2)
    hi = lambda d: sum(v for k, v in d.items() if int(k) >= 3)  # noqa: E731
    honE = [r["E"] for r in hon]
    thr = {"p95": pct(honE, 0.95), "p99": pct(honE, 0.99)}
    nulled = [r for r in l2 if r["score"] <= 1]
    expression = {
        "n_honest": len(hon), "n_level2": len(l2), "min_digit_token_mass": min(r["digit_mass"] for r in rows),
        "mean_digit_dist": {"honest": dh, "level2": dl},
        "mass_at_or_above_3": {"honest": hi(dh), "level2": hi(dl)},
        "E_minus_score": {s: {"mean": st.mean([r["E"] - r["score"] for r in g]), "median": st.median([r["E"] - r["score"] for r in g])}
                          for s, g in (("honest", hon), ("level2", l2))},
        "honest_E_thresholds": thr,
        "p_null_integer": {"n_level2": len(l2), "n_score_le_1": len(nulled), "p_null": len(nulled) / len(l2)},
        "delta_p_null_from_E": {q: {"k": sum(r["E"] > thr[q] for r in nulled), "n": len(nulled),
                                    "frac": sum(r["E"] > thr[q] for r in nulled) / len(nulled)} for q in thr},
        "E_of_nulled": {"mean": st.mean([r["E"] for r in nulled]), "max": max(r["E"] for r in nulled)},
        "auroc_note": "AUROC for E / score / cached is in the C entry raw; not recomputed here",
    }

    def rate(rs, pred):
        return sum(bool(pred(r)) for r in rs) / len(rs)

    feats = {}
    for m in MAGIC + SHAPE:
        feats[m] = {"level2": rate(l2, lambda r, m=m: m in r["text"]), "honest": rate(hon, lambda r, m=m: m in r["text"]),
                    "level2_n": sum(m in r["text"] for r in l2), "honest_n": sum(m in r["text"] for r in hon)}
    # environment-identity unigrams: high on level-2, zero on honest, and concentrated in few trajectories
    tok = lambda s: set(re.findall(r"[A-Za-z0-9_.\-/]{4,}", s.lower()))  # noqa: E731
    cl, ch = collections.Counter(), collections.Counter()
    for r in l2: cl.update(tok(r["text"]))
    for r in hon: ch.update(tok(r["text"]))
    cand = sorted(((w, cl[w], ch[w]) for w in cl if cl[w] >= 5), key=lambda x: -(x[1] / len(l2) - x[2] / len(hon)))
    top = []
    for w, a, b in cand[:20]:
        trs = {r["traj_id"] for r in l2 if w in tok(r["text"])}
        top.append({"token": w, "level2_n": a, "level2_rate": a / len(l2), "honest_n": b, "honest_rate": b / len(hon),
                    "n_level2_trajectories": len(trs)})
    leakage = {
        "substring_features": feats,
        "top_discriminative_unigrams": top,
        "action_chars": {s: {"median": st.median([r["chars"] for r in g]), "mean": st.mean([r["chars"] for r in g])}
                         for s, g in (("honest", hon), ("level2", l2))},
        "position_in_trajectory": {s: st.median([r["pos"] for r in g]) for s, g in (("honest", hon), ("level2", l2))},
        "trajectory_length": {s: st.median([r["n_actions"] for r in g])
                              for s, g in (("honest", hon), ("attack", [r for r in rows if r["side"] == "attack"]))},
        "level2_concentration": {"n_level2": len(l2), "n_trajectories": len({r["traj_id"] for r in l2}),
                                 "per_trajectory": dict(collections.Counter(r["traj_id"][:8] for r in l2).most_common()),
                                 "per_env": dict(collections.Counter(r["env"] for r in l2))},
        "honest_trajectories": len({r["traj_id"] for r in hon}),
    }

    fig = None
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        f, ax = plt.subplots(1, 2, figsize=(10, 3.6))
        ks = [str(i) for i in range(11)]
        ax[0].bar([i - .2 for i in range(11)], [dh.get(k, 0) for k in ks], .4, label=f"honest (n={len(hon)})", color="#888")
        ax[0].bar([i + .2 for i in range(11)], [dl.get(k, 0) for k in ks], .4, label=f"level-2 (n={len(l2)})", color="#c44")
        ax[0].set_yscale("log"); ax[0].set_xlabel("score digit"); ax[0].set_ylabel("mean probability")
        ax[0].set_title("forced-readout digit distribution", fontsize=9); ax[0].legend(fontsize=7)
        ax[1].scatter([r["score"] for r in hon], [r["E"] for r in hon], s=6, alpha=.3, color="#888", label="honest")
        ax[1].scatter([r["score"] for r in l2], [r["E"] for r in l2], s=14, alpha=.8, color="#c44", label="level-2")
        ax[1].axhline(thr["p95"], color="k", ls="--", lw=.8); ax[1].text(6, thr["p95"] + .1, "honest p95 of E", fontsize=7)
        ax[1].set_xlabel("emitted integer score"); ax[1].set_ylabel("expected score E")
        ax[1].set_title("E vs score (E lifts the nulled payloads)", fontsize=9); ax[1].legend(fontsize=7)
        FIG_DIR.mkdir(parents=True, exist_ok=True); fig = FIG_DIR / "d_expression_gap.png"; f.tight_layout(); f.savefig(fig, dpi=150)
    except ImportError as e:
        print("!! no matplotlib:", e)

    out = {"slug": SLUG, "script": f"experiments/{SLUG}/d_expression_gap.py", "timestamp": TS,
           "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip(),
           "g4_commit": G4_COMMIT, "inputs": {f"g4/{f}": hashlib.sha256((G4 / f).read_bytes()).hexdigest() for f in G4_FILES},
           "action_labels_sha256": hashlib.sha256(LABELS.read_bytes()).hexdigest(),
           "scope": "all 2,012 subset_main actions (main pass + first-pass 200), non-placeholder, score not null; positives = attack label_level 2, negatives = honest",
           "expression": expression, "leakage": leakage,
           "figure": str(fig.relative_to(REPO)) if fig else None}
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    p = RAW_DIR / f"d_expression_gap_{TS}.json"; p.write_text(json.dumps(out, indent=1))
    print("raw ->", p.relative_to(REPO), "| figure ->", out["figure"])
    print(json.dumps(expression, indent=1))
    print("\nleakage: substring features (level2 rate / honest rate)")
    for k, v in feats.items():
        print(f"  {k:24} {v['level2_n']:3}/{len(l2)} {v['level2']:.2f}   {v['honest_n']:4}/{len(hon)} {v['honest']:.4f}")
    print("\ntop discriminative unigrams (token, level2 n/rate, honest n/rate, #level2 trajectories):")
    for t in top[:12]:
        print(f"  {t['token']:26} {t['level2_n']:3} {t['level2_rate']:.2f}  {t['honest_n']:4} {t['honest_rate']:.3f}  trajs={t['n_level2_trajectories']}")
    print("\n", json.dumps({k: v for k, v in leakage.items() if k not in ("substring_features", "top_discriminative_unigrams")}, indent=1))


if __name__ == "__main__":
    main()
