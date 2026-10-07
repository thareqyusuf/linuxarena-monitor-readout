"""P0b — anchoring: does the scale distort a state that exists? No activations; records only.

 1. Where the digit mass sits for level-2 actions whose integer is <= 1 (the 31 'nulled' payloads):
    aggregate digit_dist and per-action mass in the bands 0-1 / 2-5 / 6-10. Bimodal (second lump at
    6-8) -> sampling discarded a real second mode (reading a). Unimodal at 0-1 with a thin tail ->
    the scale pulls the verdict down (reading b). The shape is reported, not asserted.
 2. 'Says one thing, scores another': analysis_note_text matching the suspicion regex while
    score <= 1. Count, mean E, label_level breakdown, ten read verbatim (random.Random(0)).
    The regex also matches negations ("no malicious", "not suspicious"); a non-negated sub-count is
    reported beside the literal count so neither is mistaken for the other.

Raw results/raw/<slug>/p0b_anchoring_<ts>.json. Figure results/figures/<slug>/p0b_digit_mass.png
"""
from __future__ import annotations

import collections, random, re, statistics as st, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

SUSP = re.compile(C.SUSPICION_RE, re.I)
NEG = re.compile(r"\b(no|not|nothing|none|never|isn't|aren't|doesn't|don't|without|non-?)\W+(\w+\W+){0,2}$", re.I)


def bands(dd):
    p = {int(k): v for k, v in dd.items()}
    return {"0-1": sum(p.get(i, 0) for i in (0, 1)), "2-5": sum(p.get(i, 0) for i in range(2, 6)), "6-10": sum(p.get(i, 0) for i in range(6, 11))}


def meandist(rows):
    agg = collections.Counter()
    for r in rows:
        for d, p in r["digit_dist"].items():
            agg[int(d)] += p
    return {str(d): agg[d] / len(rows) for d in range(11)}


def main():
    stamp = C.ts(); rows, meta = C.build_table()
    u = C.usable(rows, "current"); pos, neg = C.pos_neg(u)
    pool = "common743" if "--pool" in sys.argv and sys.argv[sys.argv.index("--pool") + 1] == "common743" else "all"
    if pool == "common743":  # 2026-09-12: honest rows restricted to the common 743 (rows with generation positions), keyed off P11's row table
        import glob, json as _json
        p11p = sorted(glob.glob(str(C.RAW_DIR / "p11_probe_beats_e_*.json")))[-1]; p11 = _json.load(open(p11p)); cols = p11["rows_all"]["columns"]
        keep = {(r[cols.index("traj_id")], r[cols.index("action_idx")]) for r in p11["rows_all"]["rows"] if r[cols.index("class")] == "honest"}
        neg = [r for r in neg if (r["traj_id"], r["action_idx"]) in keep]; assert len(neg) == len(keep) == 743, (len(neg), len(keep))
        pool_info = {"name": pool, "rule": "honest rows listed as class 'honest' in the P11 rows_all table (the 743 with generation positions)", "p11_raw": str(Path(p11p).relative_to(C.REPO)), "n_neg": len(neg)}
    else:
        pool_info = {"name": pool, "rule": "all usable honest rows", "n_neg": len(neg)}
    raw_name, fig_name = ("p0b_anchoring_pool743", "p0b_digit_mass_pool743.png") if pool == "common743" else ("p0b_anchoring", "p0b_digit_mass.png")
    nulled = sorted([r for r in pos if r["score"] <= 1], key=lambda r: -r["E"])
    loud = [r for r in pos if r["score"] >= 3]
    per = [{"traj_id": r["traj_id"], "action_idx": r["action_idx"], "score": r["score"], "E": r["E"], "cached_mean": r["cached_mean"],
            "audit_verdict": r["audit_verdict"], "bands": bands(r["digit_dist"]), "digit_dist": {str(k): r["digit_dist"].get(str(k), 0.0) for k in range(11)},
            "second_mode_ge6_over_0.05": bands(r["digit_dist"])["6-10"] > 0.05, "argmax": int(max(r["digit_dist"], key=r["digit_dist"].get))} for r in nulled]
    mass = {"nulled_level2": {"n": len(nulled), "mean_digit_dist": meandist(nulled), "mean_bands": {b: st.mean(p["bands"][b] for p in per) for b in ("0-1", "2-5", "6-10")},
                              "n_with_second_mode_ge6_over_0.05": sum(p["second_mode_ge6_over_0.05"] for p in per),
                              "n_with_band_6_10_over_0.20": sum(p["bands"]["6-10"] > 0.20 for p in per),
                              "band_6_10_quantiles": {q: float(np.quantile([p["bands"]["6-10"] for p in per], q)) for q in (0.5, 0.75, 0.9, 1.0)},
                              "per_action": per},
            "loud_level2_score_ge3": {"n": len(loud), "mean_digit_dist": meandist(loud)},
            "honest": {"n": len(neg), "mean_digit_dist": meandist(neg), "n_with_band_6_10_over_0.05": sum(bands(r["digit_dist"])["6-10"] > 0.05 for r in neg)},
            "honest_score_le1": {"n": sum(r["score"] <= 1 for r in neg), "mean_digit_dist": meandist([r for r in neg if r["score"] <= 1])}}

    # 2. says one thing, scores another
    hits = []
    for r in u:
        txt = r["analysis_note_text"] or ""
        if r["score"] <= 1 and SUSP.search(txt):
            spans = [(m.start(), m.group(0)) for m in SUSP.finditer(txt)]
            negated = [bool(NEG.search(txt[max(0, s - 40):s])) for s, _ in spans]
            hits.append({"traj_id": r["traj_id"], "action_idx": r["action_idx"], "side": r["side"], "label_level": r["label_level"], "audit_verdict": r["audit_verdict"],
                         "score": r["score"], "E": r["E"], "cached_mean": r["cached_mean"], "matches": [m for _, m in spans], "all_matches_negated": all(negated),
                         "analysis_note_text": txt})
    nonneg = [h for h in hits if not h["all_matches_negated"]]
    def summ(hs):
        return {"n": len(hs), "n_traj": len({h["traj_id"] for h in hs}), "mean_E": st.mean(h["E"] for h in hs) if hs else None,
                "by_side_level": dict(collections.Counter(f"{h['side']}|level={h['label_level']}" for h in hs)),
                "n_level2": sum(h["side"] == "attack" and h["label_level"] == 2 for h in hs)}
    rng = random.Random(0); ten = rng.sample(hits, min(10, len(hits)))
    says = {"regex": C.SUSPICION_RE, "scope": "usable rows (score not null, current-action placeholder excluded), score <= 1, analysis_note_text",
            "literal": summ(hits), "non_negated": summ(nonneg), "level2_hits": [h for h in hits if h["side"] == "attack" and h["label_level"] == 2],
            "ten_verbatim_random0": ten, "all_hits": hits,
            "baseline_rate": {"n_usable_score_le1": sum(r["score"] <= 1 for r in u), "mean_E_score_le1": st.mean(r["E"] for r in u if r["score"] <= 1)}}
    out = {"slug": C.SLUG, "script": f"experiments/{C.SLUG}/p0b_anchoring.py", "timestamp": stamp, "git_commit": C.git_commit(), "inputs": meta, "pool": pool_info,
           "digit_mass": mass, "says_one_thing": says}

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1, 1.3]})
    xs = np.arange(11); w = 0.27
    for j, (nm, key, col) in enumerate([("honest", "honest", C.COLORS["honest"]), ("level-2, integer <= 1 (nulled)", "nulled_level2", C.COLORS["level2"]), ("level-2, integer >= 3", "loud_level2_score_ge3", C.COLORS["attack_other"])]):
        d = mass[key]["mean_digit_dist"]; ax[0].bar(xs + (j - 1) * w, [d[str(x)] for x in xs], w, color=col, label=f"{nm} (n={mass[key]['n']})")
    ax[0].set_yscale("log"); ax[0].set_ylim(1e-4, 1); ax[0].set_xlabel("score digit at forced readout"); ax[0].set_ylabel("mean probability"); ax[0].legend(fontsize=7, frameon=False)
    ax[0].set_title("where the digit mass sits", fontsize=9)
    M = np.array([[p["digit_dist"][str(k)] for k in range(11)] for p in per])
    im = ax[1].imshow(np.log10(np.clip(M, 1e-4, 1)), aspect="auto", cmap="Oranges", vmin=-4, vmax=0)
    ax[1].set_yticks(range(len(per))); ax[1].set_yticklabels([f"{p['traj_id'][:6]}#{p['action_idx']} E={p['E']:.2f}" for p in per], fontsize=5.5)
    ax[1].set_xticks(range(11)); ax[1].set_xlabel("digit"); ax[1].set_title("per-action digit distribution, 31 nulled payloads (log10 p), sorted by E", fontsize=9)
    plt.colorbar(im, ax=ax[1], fraction=0.03, label="log10 p")
    for a in ax[:1]: a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); p = C.fig_path(fig_name); fig.savefig(p, dpi=150); plt.close(fig); out["figure"] = str(p.relative_to(C.REPO))
    C.write_raw(raw_name, out, stamp)
    m = mass["nulled_level2"]
    print(f"nulled level-2 n={m['n']}: mean bands {m['mean_bands']}; second mode (6-10 > 0.05) in {m['n_with_second_mode_ge6_over_0.05']}; 6-10 > 0.20 in {m['n_with_band_6_10_over_0.20']}; 6-10 quantiles {m['band_6_10_quantiles']}")
    print("mean digit dist nulled:", {k: round(v, 4) for k, v in m['mean_digit_dist'].items()})
    print("mean digit dist honest:", {k: round(v, 4) for k, v in mass['honest']['mean_digit_dist'].items()}, "honest with 6-10>0.05:", mass["honest"]["n_with_band_6_10_over_0.05"])
    print(f"\nsays-one-thing: literal {says['literal']} | non-negated {says['non_negated']} | baseline mean E of score<=1: {says['baseline_rate']['mean_E_score_le1']:.3f} over {says['baseline_rate']['n_usable_score_le1']}")
    print("level-2 hits:", [(h["traj_id"][:8], h["action_idx"], h["score"], round(h["E"], 2), h["all_matches_negated"]) for h in says["level2_hits"]])
    for h in ten:
        print(f"\n--- {h['side']} lvl={h['label_level']} audit={h['audit_verdict']} score={h['score']} E={h['E']:.2f} cached={h['cached_mean']} negated={h['all_matches_negated']} [{h['traj_id'][:8]}#{h['action_idx']}]\n{h['analysis_note_text']}")


if __name__ == "__main__":
    main()
