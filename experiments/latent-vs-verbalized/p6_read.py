"""P6 — read the data. Lists for the write-up, drawn from the pooled out-of-sample scores in the P3 raw:
 - 5 full generations drawn with random.Random(0) from usable subset_main actions, verbatim
 - top 10 honest actions by the label probe (and where the score probe ranks them)
 - the 31 level-2 actions with integer <= 1, ranked by the label probe: top 10 with the audit verdict
 - bottom 10 level-2 by the label probe
Raw results/raw/<slug>/p6_read_<ts>.json (texts included; small)
"""
from __future__ import annotations

import glob, json, random, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C


def main():
    stamp = C.ts(); rows, meta = C.build_table(with_text=True); rec, _ = C.load_scores()
    p3p = sorted(p for p in glob.glob(str(C.RAW_DIR / "p3_baselines_*.json")) if "_ph" not in p)[-1]; p3 = json.load(open(p3p))
    cols = p3["scores_pooled_59"]["columns"]; S = {(r[0], r[1]): dict(zip(cols, r)) for r in p3["scores_pooled_59"]["rows"]}
    u = [r for r in C.usable(rows, "current")]; lut = {(r["traj_id"], r["action_idx"]): r for r in u}
    def rank_pct(field, val, pop):   # percentile among honest
        hv = [S[(r["traj_id"], r["action_idx"])][field] for r in pop if r["is_neg"] and S[(r["traj_id"], r["action_idx"])][field] is not None]
        return float(np.mean([h < val for h in hv]) * 100)
    def brief(r):
        s = S[(r["traj_id"], r["action_idx"])]
        return {"traj_id": r["traj_id"], "action_idx": r["action_idx"], "side": r["side"], "env": r["env"], "label_level": r["label_level"], "audit_verdict": r["audit_verdict"], "tool": r["tool_name"],
                "score": r["score"], "E": r["E"], "cached_mean": r["cached_mean"], "probe_label": s["label|logreg"], "probe_score": s["score|ridge_E"], "text": s["text"],
                "probe_label_honest_pct": rank_pct("label|logreg", s["label|logreg"], u) if s["label|logreg"] is not None else None,
                "probe_score_honest_pct": rank_pct("score|ridge_E", s["score|ridge_E"], u) if s["score|ridge_E"] is not None else None,
                "position_in_trajectory": r["position_in_trajectory"], "n_action_chars": r["n_action_chars"], "action_text": r["action_text"][:1200], "analysis_note_text": r["analysis_note_text"]}
    rng = random.Random(0); five = rng.sample(u, 5)
    reasoning = [{**brief(r), "generation_text": rec[(r["traj_id"], r["action_idx"])]["generation_text"]} for r in five]
    hon = sorted([r for r in u if r["is_neg"] and S[(r["traj_id"], r["action_idx"])]["label|logreg"] is not None], key=lambda r: -S[(r["traj_id"], r["action_idx"])]["label|logreg"])
    top_honest = [brief(r) for r in hon[:10]]
    l2 = [r for r in u if r["is_pos"] and S[(r["traj_id"], r["action_idx"])]["label|logreg"] is not None]
    nulled = sorted([r for r in l2 if r["score"] <= 1], key=lambda r: -S[(r["traj_id"], r["action_idx"])]["label|logreg"])
    bottom = sorted(l2, key=lambda r: S[(r["traj_id"], r["action_idx"])]["label|logreg"])[:10]
    out = {"slug": C.SLUG, "script": f"experiments/{C.SLUG}/p6_read.py", "timestamp": stamp, "git_commit": C.git_commit(), "inputs": meta | {"p3_raw": str(Path(p3p).relative_to(C.REPO))},
           "five_generations_random0": reasoning, "top10_honest_by_label_probe": top_honest, "nulled_level2_ranked_by_label_probe": [brief(r) for r in nulled],
           "bottom10_level2_by_label_probe": [brief(r) for r in bottom], "honest_thresholds": {f: {"p95": C.honest_pct([S[(r["traj_id"], r["action_idx"])][f] for r in u if r["is_neg"] and S[(r["traj_id"], r["action_idx"])][f] is not None], 0.95)} for f in ("label|logreg", "score|ridge_E", "E")}}
    C.write_raw("p6_read", out, stamp)
    def line(b):
        return f"[{b['traj_id'][:8]}#{b['action_idx']:3}] {b['side']:6} lvl={b['label_level']} audit={b['audit_verdict']} {b['env']:16} {b['tool']:11} int={b['score']} E={b['E']:.2f} cached={b['cached_mean']} | probeL {b['probe_label']:+6.2f} (hon-pct {b['probe_label_honest_pct']:.0f}) probeS {b['probe_score']:+5.2f} (hon-pct {b['probe_score_honest_pct']:.0f}) text {b['text']:+5.2f} | pos {b['position_in_trajectory']:.2f} chars {b['n_action_chars']}\n      {b['action_text'][:220]!r}"
    print("== 5 generations (random.Random(0)) ==")
    for b in reasoning: print(line(b)); print("      GEN:", b["generation_text"][:1500].replace("\n", "\n      "), "\n")
    print("== top 10 honest by label probe =="); [print(line(b)) for b in top_honest]
    print(f"\n== {len(nulled)} level-2 with integer <= 1, ranked by label probe =="); [print(line(brief(r))) for r in nulled]
    print("\n== bottom 10 level-2 by label probe =="); [print(line(brief(r))) for r in bottom]
    print("\nhonest p95 thresholds:", out["honest_thresholds"])


if __name__ == "__main__":
    main()
