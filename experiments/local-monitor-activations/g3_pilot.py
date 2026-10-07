#!/usr/bin/env python3
"""G3 analysis (CPU): reads the three pilot output dirs written by run_monitor.py via g3_pilot.sh and
computes everything TASK-ver2 G3 asks for. Writes results/raw/<slug>/g3_pilot_<ts>.json and
results/figures/<slug>/g3_layer_norms.png.

  1. throughput: seconds/action by length decile (of the pilot), prefill vs total, peak VRAM, payload bytes
  2. noise floor: chunk 256 vs chunk 128 on the same 20 actions — |Δscore|, |Δexpected_score|, cosine distance
     between capture vectors per position (last layer and all layers), digit-mass difference
  3. activation health: shapes, dtype, NaN/Inf, per-layer norm profile per position (median over actions),
     pairwise distinctness across actions at layers >= 1
  4. neutralized render: prompt changed (sha differs, token count), still parses (digit mass), score delta vs original
  5. projection: measured seconds vs the G2 fit -> N for the main run
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve()
SLUG = HERE.parent.name
REPO = HERE.parents[2]
RAW = REPO / "results" / "raw" / SLUG
FIG = REPO / "results" / "figures" / SLUG
OUT = REPO / "checkpoints" / SLUG / "out"
TS = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
sys.path.insert(0, str(HERE.parent))
import monitor_prompt as mp  # noqa: E402

POS = list(mp.POSITIONS)


def load_run(name):
    d = OUT / name
    recs = [json.loads(l) for l in (d / "scores.jsonl").open()]
    idx = json.loads((d / "acts_index.json").read_text())
    acts = {}
    for shard in idx["shards"]:
        acts.update(torch.load(d / shard))
    meta = json.loads((d / "run_meta.json").read_text())
    assert idx["position_order"] == POS
    return recs, acts, meta


R: dict = {"slug": SLUG, "script": str(HERE.relative_to(REPO)), "timestamp": TS, "runs": {}}
runs = {}
for name in ("pilot_c256", "pilot_c128", "pilot_neut_c256"):
    recs, acts, meta = load_run(name)
    runs[name] = (recs, acts, meta)
    R["runs"][name] = {"n_records": len(recs), "n_ok": sum(r["status"] == "ok" for r in recs), "n_acts": len(acts),
                       "status_counts": {s: sum(r["status"] == s for r in recs) for s in {r["status"] for r in recs}},
                       "meta": {k: meta.get(k) for k in ("attn_impl", "attn_chunk_size", "render", "git_commit", "seconds_this_run", "started", "finished")}}
key = lambda r: f"{r['traj_id']}|{r['action_idx']}"  # noqa: E731

# ── 1. throughput ───────────────────────────────────────────────────────────
main_recs, main_acts, main_meta = runs["pilot_c256"]
ok = [r for r in main_recs if r["status"] == "ok"]
ok.sort(key=lambda r: r["n_prompt_tokens"])
n = len(ok)
dec = []
for d in range(10):
    rs = ok[d * n // 10:(d + 1) * n // 10]
    if not rs:
        continue
    dec.append({"decile": d, "n": len(rs), "tokens_min": rs[0]["n_prompt_tokens"], "tokens_max": rs[-1]["n_prompt_tokens"],
                "seconds_mean": float(np.mean([r["seconds"] for r in rs])), "prefill_seconds_mean": float(np.mean([r["prefill_seconds"] for r in rs])),
                "peak_gib_max": max(r["peak_gib"] for r in rs)})
first = ok[0] if ok else None
warm = [r for r in ok if r is not main_recs[0]]
R["throughput"] = {"by_decile": dec, "n_ok": n,
                   "first_action_seconds_incl_warmup": main_recs[0]["seconds"] if main_recs and main_recs[0]["status"] == "ok" else None,
                   "peak_gib_max": max(r["peak_gib"] for r in ok), "peak_gib_at_longest": ok[-1]["peak_gib"] if ok else None,
                   "acts_bytes_per_action": ok[0]["acts_bytes"] if ok else None,
                   "shard_bytes": {p.name: p.stat().st_size for p in sorted((OUT / "pilot_c256").glob("acts_*.pt"))},
                   "seconds_total_run": main_meta.get("seconds_this_run"),
                   "overhead_seconds_mean": float(np.mean([r["seconds"] - r["prefill_seconds"] for r in ok[1:]])) if n > 1 else None,
                   "per_action": [{"key": key(r), "side": r["side"], "n_tokens": r["n_prompt_tokens"], "seconds": r["seconds"], "prefill_seconds": r["prefill_seconds"],
                                   "peak_gib": r["peak_gib"], "score": r["score"], "expected_score": r["expected_score"], "digit_token_mass": r["digit_token_mass"],
                                   "after_message_mass": r["after_message"]["digit_token_mass"], "after_message_top1": r["after_message"]["top10"][0]["token"],
                                   "cached_mean": r["cached_mean"], "has_placeholder": r["has_attachment_placeholder"]} for r in ok]}

# ── 5. projection: measured vs G2 fit ─────────────────────────────────────────
g2 = json.loads(sorted(RAW.glob("g2_prep_*.json"))[-1].read_text())
fit = g2["prefill_fit"]
pred = lambda L: fit["c"] + fit["a"] * L + fit["b"] * L * L + fit["per_action_overhead_s"]  # noqa: E731
meas = [(r["n_prompt_tokens"], r["seconds"], r["prefill_seconds"]) for r in ok[1:]]        # drop warm-up action
ratio = float(sum(s for _, s, _ in meas) / sum(pred(L) for L, _, _ in meas)) if meas else None
# refit quadratic on measured prefill seconds (relative-error weighted, non-negative)
from scipy.optimize import nnls  # noqa: E402
Ls = np.array([L for L, _, _ in meas], float); Ss = np.array([s for _, s, _ in meas]); Ps = np.array([p for _, _, p in meas])
A = np.stack([np.ones_like(Ls), Ls, Ls ** 2], 1); w = 1 / Ss
coef, _ = nnls(A * w[:, None], Ss * w)
new_t = lambda L: coef[0] + coef[1] * L + coef[2] * L * L  # noqa: E731
sub_main = json.loads((HERE.parent / "subset_main.json").read_text())["actions"]
sub_cf = json.loads((HERE.parent / "subset_counterfactual.json").read_text())["actions"]
R["projection"] = {"g2_fit": {k: fit[k] for k in ("c", "a", "b", "per_action_overhead_s")},
                   "measured_over_g2_projection_ratio": ratio,
                   "pilot_fit_total_seconds": {"model": "t = c + a*L + b*L^2 on measured total seconds/action (warm-up action excluded), NNLS, 1/t weights",
                                               "c": float(coef[0]), "a": float(coef[1]), "b": float(coef[2]),
                                               "max_rel_err": float(np.max(np.abs(A @ coef - Ss) / Ss)) if len(Ss) else None},
                   "main_hours_g2": g2["projection"]["main_hours"], "main_hours_pilot_fit": float(sum(new_t(a["n_tokens"]) for a in sub_main) / 3600),
                   "counterfactual_hours_per_render_pilot_fit": float(sum(new_t(a["n_tokens"]) for a in sub_cf) / 3600),
                   "off_by_more_than_20pct": (abs(ratio - 1) > 0.2) if ratio is not None else None}
R["projection"]["total_hours_pilot_fit"] = R["projection"]["main_hours_pilot_fit"] + 2 * R["projection"]["counterfactual_hours_per_render_pilot_fit"]

# ── 2. noise floor ─────────────────────────────────────────────────────────────
noise_recs, noise_acts, _ = runs["pilot_c128"]
by256 = {key(r): r for r in ok}
pairs = []
for r in noise_recs:
    if r["status"] != "ok" or key(r) not in by256:
        continue
    a = by256[key(r)]
    A1 = main_acts[f"{key(r)}|original"].float(); A2 = noise_acts[f"{key(r)}|original"].float()
    cos = torch.nn.functional.cosine_similarity(A1, A2, dim=-1)        # [4, 25]
    pairs.append({"key": key(r), "n_tokens": r["n_prompt_tokens"], "d_score": abs(r["score"] - a["score"]),
                  "d_expected": abs(r["expected_score"] - a["expected_score"]), "d_mass": abs(r["digit_token_mass"] - a["digit_token_mass"]),
                  "cosdist_last_layer": [float(1 - cos[i, -1]) for i in range(4)],
                  "cosdist_max_over_layers_ge1": [float((1 - cos[i, 1:]).max()) for i in range(4)],
                  "cosdist_layer0": [float(1 - cos[i, 0]) for i in range(4)],
                  "acts_identical": bool(torch.equal(A1, A2)), "score_256": a["score"], "score_128": r["score"],
                  "E_256": a["expected_score"], "E_128": r["expected_score"]})


def dist(xs):
    xs = np.array(xs, float)
    return {"n": int(len(xs)), "mean": float(xs.mean()), "median": float(np.median(xs)), "p90": float(np.percentile(xs, 90)), "max": float(xs.max())} if len(xs) else {"n": 0}


R["noise_floor"] = {"n_pairs": len(pairs), "chunk_sizes": [256, 128],
                    "d_score": dist([p["d_score"] for p in pairs]), "n_score_changed": sum(p["d_score"] > 0 for p in pairs),
                    "d_expected_score": dist([p["d_expected"] for p in pairs]), "d_digit_mass": dist([p["d_mass"] for p in pairs]),
                    "cosine_distance_last_layer_by_position": {POS[i]: dist([p["cosdist_last_layer"][i] for p in pairs]) for i in range(4)},
                    "cosine_distance_max_over_layers_by_position": {POS[i]: dist([p["cosdist_max_over_layers_ge1"][i] for p in pairs]) for i in range(4)},
                    "n_acts_identical": sum(p["acts_identical"] for p in pairs), "pairs": pairs}

# ── 3. activation health ───────────────────────────────────────────────────────
allA = torch.stack([main_acts[f"{key(r)}|original"] for r in ok])            # [n, 4, 25, 2880] fp16
Af = allA.float()
norms = Af.norm(dim=-1)                                                         # [n, 4, 25]
health = {"n": n, "shape_per_action": list(allA.shape[1:]), "dtype": str(allA.dtype),
          "n_nan": int(torch.isnan(Af).sum()), "n_inf": int(torch.isinf(Af).sum()),
          "norm_median_by_position_layer": {POS[i]: [float(x) for x in norms[:, i].median(dim=0).values] for i in range(4)},
          "norm_min_by_position_layer": {POS[i]: [float(x) for x in norms[:, i].min(dim=0).values] for i in range(4)},
          "norm_max_by_position_layer": {POS[i]: [float(x) for x in norms[:, i].max(dim=0).values] for i in range(4)},
          "fp16_max_abs": float(Af.abs().max()), "fp16_overflow_risk": float(Af.abs().max()) > 60000}
# distinctness: for each position and layer >= 1, min over action pairs of cosine distance
distinct = {}
for i in range(4):
    X = Af[:, i, 1:, :]                                                          # [n, 24, d]
    Xn = X / X.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    sims = torch.einsum("ald,bld->abl", Xn, Xn)                                  # [n, n, 24]
    eye = torch.eye(n, dtype=torch.bool)[:, :, None]
    sims = sims.masked_fill(eye, -1)
    maxsim = sims.max(dim=1).values                                              # [n, 24] nearest other action
    iu = torch.triu_indices(n, n, 1)
    pair_d = 1 - sims[iu[0], iu[1]]                                              # [n_pairs, 24]
    distinct[POS[i]] = {"min_cosine_distance_over_pairs_by_layer": [float(1 - maxsim[:, l].max()) for l in range(24)],
                        "median_pairwise_cosine_distance_by_layer": [float(x) for x in pair_d.median(dim=0).values],
                        "median_pairwise_cosine_distance_last_layer": float(pair_d[:, -1].median()),
                        "p10_pairwise_cosine_distance_last_layer": float(pair_d[:, -1].quantile(0.1)),
                        "any_identical_pair_layers_ge1": bool((sims >= 1 - 1e-6).any())}
    # layer 0 (embedding) identical pairs are expected at template positions
    E0 = Af[:, i, 0, :]
    distinct[POS[i]]["layer0_n_identical_pairs"] = int(((E0[:, None] == E0[None]).all(-1).sum() - n) // 2)
health["distinct"] = distinct
R["activation_health"] = health

# ── 4. neutralized render ──────────────────────────────────────────────────────
neut_recs, neut_acts, _ = runs["pilot_neut_c256"]
neut = []
for r in neut_recs:
    a = by256.get(key(r))
    if r["status"] != "ok" or a is None:
        continue
    A1 = main_acts[f"{key(r)}|original"].float(); A2 = neut_acts[f"{key(r)}|neutralized"].float()
    cos = torch.nn.functional.cosine_similarity(A1, A2, dim=-1)
    neut.append({"key": key(r), "side": r["side"], "prompt_changed": r["prompt_sha256"] != a["prompt_sha256"],
                 "sha_matches_manifest": r["prompt_sha256_matches_manifest"], "n_tokens_original": a["n_prompt_tokens"], "n_tokens_neutralized": r["n_prompt_tokens"],
                 "events": r["neutralize_events"], "parses": r["digit_token_mass"] > 0.9, "digit_token_mass": r["digit_token_mass"],
                 "score_original": a["score"], "score_neutralized": r["score"], "E_original": a["expected_score"], "E_neutralized": r["expected_score"],
                 "cosdist_last_layer": [float(1 - cos[i, -1]) for i in range(4)], "cached_mean": r["cached_mean"]})
R["neutralized"] = {"n": len(neut), "n_prompt_changed": sum(x["prompt_changed"] for x in neut), "n_parses": sum(x["parses"] for x in neut),
                    "n_with_action_events": sum(x["events"]["action_events"] > 0 for x in neut),
                    "d_expected_score": dist([abs(x["E_neutralized"] - x["E_original"]) for x in neut]),
                    "signed_d_expected_score_mean": float(np.mean([x["E_neutralized"] - x["E_original"] for x in neut])) if neut else None,
                    "cosine_distance_last_layer_by_position": {POS[i]: dist([x["cosdist_last_layer"][i] for x in neut]) for i in range(4)},
                    "per_action": neut}

# ── scores overview ────────────────────────────────────────────────────────────
R["scores"] = {"after_message_digit_mass_max": max(r["after_message"]["digit_token_mass"] for r in ok),
               "after_message_top1_tokens": {t: sum(r["after_message"]["top10"][0]["token"] == t for r in ok) for t in {r["after_message"]["top10"][0]["token"] for r in ok}},
               "after_score_tag_digit_mass_min": min(r["digit_token_mass"] for r in ok),
               "score_hist": {str(v): sum(r["score"] == v for r in ok) for v in range(11)},
               "expected_score_by_side": {s: dist([r["expected_score"] for r in ok if r["side"] == s]) for s in ("honest", "attack")},
               "cached_mean_by_side": {s: dist([r["cached_mean"] for r in ok if r["side"] == s]) for s in ("honest", "attack")},
               "spearman_expected_vs_cached": None}
try:
    from scipy.stats import spearmanr
    xs = [r["expected_score"] for r in ok]; ys = [r["cached_mean"] for r in ok]
    R["scores"]["spearman_expected_vs_cached"] = float(spearmanr(xs, ys).statistic)
except Exception as e:  # noqa: BLE001
    R["scores"]["spearman_error"] = str(e)

# ── figure ─────────────────────────────────────────────────────────────────────
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
cols = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
for i, pname in enumerate(POS):
    med = np.array(health["norm_median_by_position_layer"][pname]); lo = np.array(health["norm_min_by_position_layer"][pname]); hi = np.array(health["norm_max_by_position_layer"][pname])
    axes[0].plot(range(25), med, color=cols[i], lw=2, label=pname); axes[0].fill_between(range(25), lo, hi, color=cols[i], alpha=0.12, lw=0)
axes[0].set_yscale("log"); axes[0].set_xlabel("layer (0 = embedding output)"); axes[0].set_ylabel("‖h‖₂ (median, min–max band)"); axes[0].legend(frameon=False, fontsize=9)
axes[0].set_title(f"residual-stream norm per position, {n} pilot actions", fontsize=10)
for i, pname in enumerate(POS):
    axes[1].plot(range(1, 25), np.log10(np.clip(distinct[pname]["min_cosine_distance_over_pairs_by_layer"], 1e-8, None)), color=cols[i], lw=2, label=pname)
axes[1].set_xlabel("layer"); axes[1].set_ylabel("log10 min over action pairs of cosine distance"); axes[1].legend(frameon=False, fontsize=9)
axes[1].set_title("distinctness: nearest other action, per layer", fontsize=10)
for ax in axes:
    ax.grid(color="#e5e5e5", lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
fig.tight_layout(); FIG.mkdir(parents=True, exist_ok=True); fig.savefig(FIG / "g3_layer_norms.png", dpi=150); plt.close(fig)
R["figure"] = str((FIG / "g3_layer_norms.png").relative_to(REPO))
R["finished"] = dt.datetime.now(dt.timezone.utc).isoformat()
out = RAW / f"g3_pilot_{TS}.json"
out.write_text(json.dumps(R, indent=1, default=str))
print(json.dumps({k: R[k] for k in ("runs", "projection", "scores")}, indent=1, default=str))
print("throughput by decile:"); [print("  ", d) for d in dec]
print("throughput:", {k: v for k, v in R["throughput"].items() if k not in ("by_decile", "per_action", "shard_bytes")})
print("noise floor:", json.dumps({k: v for k, v in R["noise_floor"].items() if k != "pairs"}, indent=1))
print("health:", json.dumps({k: v for k, v in health.items() if "by_position_layer" not in k and k != "distinct"}, indent=1))
print("distinct:", json.dumps({p: {k: v for k, v in d.items() if "by_layer" not in k} for p, d in distinct.items()}, indent=1))
print("neutralized:", json.dumps({k: v for k, v in R["neutralized"].items() if k != "per_action"}, indent=1))
for x in neut:
    print("   ", x["key"][:12], x["side"], "changed", x["prompt_changed"], "tok", x["n_tokens_original"], "->", x["n_tokens_neutralized"], "events", x["events"], "E", round(x["E_original"], 2), "->", round(x["E_neutralized"], 2), "score", x["score_original"], "->", x["score_neutralized"])
print("raw ->", out.relative_to(REPO))
