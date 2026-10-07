"""P0c — sample variance of the verbalized integer: readout loss on a model you can open.

sampled_k4 stage: 150 subset_main actions, 4 samples at T=1 from a shared prefill (`sampled.scores`), beside the
greedy integer `score` and forced readout E from the main pass. Split by label_level and side: distribution of
max - min over the 4 samples; fraction with >= 1 sample >= 3 while the greedy integer <= 1; Spearman(mean of 4, E);
AUROC (level-2 vs honest, within the 150) for 4-sample mean / E / greedy integer. Quotes every action whose samples
span >= 5 points. Raw results/raw/<slug>/p0c_sample_variance_<ts>.json; figure p0c_sample_spread.png
"""
from __future__ import annotations

import collections, json, statistics as st, sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

SK = C.OUT / "sampled_k4" / "scores.jsonl"


def main():
    stamp = C.ts(); rows, meta = C.build_table(); lut = {(r["traj_id"], r["action_idx"]): r for r in rows}
    sk = [json.loads(l) for l in SK.open() if l.strip()]
    recs = []
    for s in sk:
        r = lut[(s["traj_id"], s["action_idx"])]; ss = s["sampled"]
        sc = [x for x in ss["scores"] if x is not None]
        recs.append({"traj_id": s["traj_id"], "action_idx": s["action_idx"], "side": r["side"], "label_level": r["label_level"], "audit_verdict": r["audit_verdict"],
                     "placeholder_current": r["placeholder_current"], "greedy_score_main": r["score"], "greedy_score_this_run": s["score"], "E_main": r["E"], "E_this_run": s["expected_score"],
                     "cached_mean": r["cached_mean"], "samples": ss["scores"], "n_valid": len(sc), "mean4": st.mean(sc) if sc else None, "max": max(sc) if sc else None, "min": min(sc) if sc else None,
                     "spread": (max(sc) - min(sc)) if sc else None, "any_ge3": any(x >= 3 for x in sc), "sample_texts": [x["text"] for x in ss["samples"]]})
    u = [r for r in recs if r["n_valid"] >= 2 and r["greedy_score_main"] is not None and not r["placeholder_current"]]
    def grp(rs):
        return {"n": len(rs), "n_traj": len({r["traj_id"] for r in rs}), "spread_hist": dict(collections.Counter(int(r["spread"]) for r in rs)),
                "spread_mean": st.mean(r["spread"] for r in rs) if rs else None, "frac_spread_ge2": float(np.mean([r["spread"] >= 2 for r in rs])) if rs else None,
                "n_greedy_le1": sum(r["greedy_score_main"] <= 1 for r in rs), "n_greedy_le1_and_any_sample_ge3": sum(r["greedy_score_main"] <= 1 and r["any_ge3"] for r in rs),
                "spearman_mean4_E": float(spearmanr([r["mean4"] for r in rs], [r["E_main"] for r in rs]).statistic) if len(rs) > 3 else None,
                "spearman_greedy_E": float(spearmanr([r["greedy_score_main"] for r in rs], [r["E_main"] for r in rs]).statistic) if len(rs) > 3 else None,
                "frac_greedy_this_run_eq_main": float(np.mean([r["greedy_score_this_run"] == r["greedy_score_main"] for r in rs])) if rs else None}
    by = {"all": grp(u), "honest": grp([r for r in u if r["side"] == "honest"]), "attack": grp([r for r in u if r["side"] == "attack"])}
    for lv in (0, 1, 2, None):
        by[f"attack_level={lv}"] = grp([r for r in u if r["side"] == "attack" and r["label_level"] == lv])
    pos = [r for r in u if r["side"] == "attack" and r["label_level"] == 2]; neg = [r for r in u if r["side"] == "honest"]
    auc = {k: C.metrics([r[f] for r in neg], [r[f] for r in pos]) for k, f in (("mean4", "mean4"), ("E", "E_main"), ("greedy", "greedy_score_main"), ("max4", "max"), ("cached_mean", "cached_mean"))} if pos and neg else None
    wide = sorted([r for r in u if r["spread"] >= 5], key=lambda r: -r["spread"])
    out = {"slug": C.SLUG, "script": f"experiments/{C.SLUG}/p0c_sample_variance.py", "timestamp": stamp, "git_commit": C.git_commit(), "inputs": meta | {"sampled_k4_scores_sha256": C.sha256(SK), "n_records": len(sk)},
           "n_usable": len(u), "n_dropped": {"placeholder_current": sum(r["placeholder_current"] for r in recs), "lt2_valid_samples": sum(r["n_valid"] < 2 for r in recs), "null_greedy": sum(r["greedy_score_main"] is None for r in recs)},
           "temperature": sk[0]["sampled"]["temperature"], "k": sk[0]["sampled"]["n"], "by_group": by, "auroc_level2_vs_honest_within_150": auc, "n_pos_within": len(pos), "n_neg_within": len(neg),
           "wide_spread_ge5": wide, "records": [{k: v for k, v in r.items() if k != "sample_texts"} for r in recs]}
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for nm, rs, col in (("honest", neg, C.COLORS["honest"]), ("attack level-2", pos, C.COLORS["level2"]), ("attack other", [r for r in u if r["side"] == "attack" and r["label_level"] != 2], C.COLORS["attack_other"])):
        if not rs: continue
        ax[0].scatter([r["E_main"] for r in rs], [r["mean4"] for r in rs], s=14, alpha=0.7, color=col, label=f"{nm} (n={len(rs)})")
        ax[1].scatter(np.array([r["greedy_score_main"] for r in rs]) + np.random.RandomState(0).uniform(-.15, .15, len(rs)), [r["spread"] for r in rs], s=14, alpha=0.6, color=col, label=nm)
    ax[0].set_xlabel("E (forced readout, main pass)"); ax[0].set_ylabel("mean of 4 samples at T=1"); ax[0].legend(fontsize=7, frameon=False); ax[0].set_title("4-sample mean vs E", fontsize=9)
    ax[1].set_xlabel("greedy integer (main pass)"); ax[1].set_ylabel("max - min over 4 samples"); ax[1].set_title("sample spread vs greedy integer", fontsize=9)
    for a in ax: a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); p = C.fig_path("p0c_sample_spread.png"); fig.savefig(p, dpi=150); plt.close(fig); out["figure"] = str(p.relative_to(C.REPO))
    C.write_raw("p0c_sample_variance", out, stamp)
    print(f"n records {len(sk)}, usable {len(u)}, dropped {out['n_dropped']}; level-2 within: {len(pos)}, honest within: {len(neg)}")
    for k, v in by.items():
        print(f"  {k:16} n {v['n']:3} spread mean {v['spread_mean'] if v['n'] else '-'} hist {v['spread_hist']} | greedy<=1 & any>=3: {v['n_greedy_le1_and_any_sample_ge3']}/{v['n_greedy_le1']} | rho(mean4,E) {v['spearman_mean4_E']} rho(greedy,E) {v['spearman_greedy_E']} | greedy reproduced {v['frac_greedy_this_run_eq_main']}")
    if auc:
        for k, m in auc.items(): print(f"  AUROC {k:12} {m['auroc']:.3f} TPR@5% {m['tpr_at_0.05']:.2f}  (n_pos {m['n_pos']}, n_neg {m['n_neg']})")
    for r in wide:
        print(f"\n--- spread {r['spread']} samples {r['samples']} greedy {r['greedy_score_main']} E {r['E_main']:.2f} {r['side']} lvl={r['label_level']} audit={r['audit_verdict']} cached={r['cached_mean']} [{r['traj_id'][:8]}#{r['action_idx']}]")
        for t in r["sample_texts"]: print("   >", t.split("<|message|>", 1)[-1][:300].replace("\n", " "))


if __name__ == "__main__":
    main()
