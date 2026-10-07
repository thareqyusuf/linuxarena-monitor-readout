#!/usr/bin/env python3
"""G2 (TASK-ver2): lengths, capture positions, subset, neutralization, projection. CPU only.

  1. sha256 of the three export files vs manifest.json (stop on mismatch)
  2. render all 17,234 manifest actions (original), apply the pinned harmony template,
     tokenize; length = len(input_ids); chars/token sanity on 10 examples
  3. length stats split by side and by has_attachment_placeholder; histogram with the cap
  4. last 12 tokens of one honest + one attack templated prompt; the four capture positions
  5. cap + trajectory-level subset -> subset_main.json
  6. neutralization table (monitor_prompt.NEUTRALIZE) -> subset_counterfactual.json, 10 diffs
  7. projected wall clock from the G1 prefill curve (quadratic fit)

Raw:    results/raw/<slug>/g2_prep_<ts>.json   (numbers + per-action length table, no tensors)
Figure: results/figures/<slug>/g2_lengths.png
Subsets: experiments/<slug>/subset_main.json, subset_counterfactual.json

  PYTHONHASHSEED=0 python experiments/<slug>/g2_prep.py --template-date 2026-09-10 --reasoning-effort low
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import difflib
import hashlib
import json
import multiprocessing as mp_
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
SLUG = HERE.parent.name
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))
import monitor_prompt as mp  # noqa: E402

RAW = REPO / "results" / "raw" / SLUG
FIG = REPO / "results" / "figures" / SLUG
TS = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")

# G1 prefill curve (chunked_eager, batch 1, logits_to_keep=1, model resident), seconds
G1_PREFILL = [(4096, 1.7), (16384, 8.5), (32768, 27.6), (65536, 108.4), (98304, 270.9), (131072, 512.9)]
G1_PREFILL_SRC = ["results/raw/local-monitor-activations/g1_attn_probe_20260910T071901Z.json",
                  "results/raw/local-monitor-activations/g1_attn_probe2_20260910T074025Z.json"]
PER_ACTION_OVERHEAD_S = 1.0     # guess: hook copies, one forced-final decode step, disk writes; G3 measures it
SCORE_BUCKETS = [(0, 1), (1, 3), (3, 6), (6, 10)]   # cached_mean in [lo, hi] (first bucket closed both ends)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "unknown"


def bucket(x: float) -> str:
    for lo, hi in SCORE_BUCKETS:
        if (x >= lo if lo == 0 else x > lo) and x <= hi:
            return f"{lo}-{hi}"
    return "?"


# ── worker: one action -> compact record ─────────────────────────────────────────────────────
_W: dict = {}


def _init(template_date, effort):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(mp.MODEL_ID)
    _W["tsha"] = mp.pin_chat_template(tok, template_date)
    _W["tok"] = tok
    _W["date"], _W["effort"] = template_date, effort
    _W["trajs"], _ = mp.load_trajs()
    _W["template"] = mp.fmp.load_template()


def _one(a: dict) -> dict:
    tok, trajs, template = _W["tok"], _W["trajs"], _W["template"]
    t = trajs[a["traj_id"]]
    r = mp.render_user_text(t, a["action_idx"], "original", template)
    sha = hashlib.sha256(r["text"].encode()).hexdigest()
    s = mp.apply_template(tok, r["text"], _W["date"], _W["effort"])
    tp = mp.tokenize_with_positions(tok, s, r["text"], r)
    ids = tp["input_ids"]
    rn = mp.render_user_text(t, a["action_idx"], "neutralized", template)
    feats = mp.action_features(t["actions"][a["action_idx"]])
    return {"traj_id": a["traj_id"], "action_idx": a["action_idx"], "side": a["side"], "env": a["env"],
            "tool": a["tool"], "cached_mean": a["cached_mean"], "n_cached": len(a["cached_scores"]),
            "chars_user": len(r["text"]), "chars_templated": len(s), "n_tokens": len(ids),
            "sha_match": sha == a["prompt_sha256"], "placeholder": mp.has_attachment_placeholder(r["text"]),
            "pos": [tp["positions"][k] for k in mp.POSITIONS], "n_context_tokens": tp["n_context_tokens"],
            "neut_hist_events": rn["subs"]["history_events"], "neut_action_events": rn["subs"]["action_events"],
            "last_id": ids[-1], **feats}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template-date", required=True)
    ap.add_argument("--reasoning-effort", default="low")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-attack", type=int, default=20)
    ap.add_argument("--n-honest", type=int, default=15)
    ap.add_argument("--target-actions", type=int, default=2000)
    ap.add_argument("--min-per-category", type=int, default=3)
    ap.add_argument("--budget-hours", type=float, default=16.0, help="GPU hours allowed for G4 (main + counterfactual + scrub)")
    ap.add_argument("--n-counterfactual", type=int, default=250, help="total (used only for the cap-table worst case)")
    ap.add_argument("--n-cf-action", type=int, default=150, help="stratum: current action has >=1 neutralization event")
    ap.add_argument("--n-cf-history", type=int, default=100, help="stratum: history changes, current action does not")
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()

    R: dict = {"slug": SLUG, "script": str(HERE.relative_to(REPO)), "timestamp": TS, "git_commit": git_commit(),
               "args": vars(args), "model_id": mp.MODEL_ID}
    RAW.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    out_path = RAW / f"g2_prep_{TS}.json"

    def save():
        out_path.write_text(json.dumps(R, indent=1, default=str))

    # ── 1. export files ─────────────────────────────────────────────────────────────────────
    manifest = mp.load_manifest()
    R["export_files"] = mp.verify_export_files(manifest)
    print("G2.1 export files:", {k: v["match"] for k, v in R["export_files"].items()})
    if not all(v["match"] for v in R["export_files"].values()):
        R["fatal"] = "export file sha256 mismatch"; save(); sys.exit(1)
    trajs, tmpl_sha = mp.load_trajs()
    assert tmpl_sha == manifest["prompt_template_sha256"]
    R["prompt_template_sha256"] = tmpl_sha
    R["manifest"] = {"git_commit": manifest["git_commit"], "n_actions": manifest["n_actions"],
                     "n_actions_by_side": manifest["n_actions_by_side"], "n_trajectories": manifest["n_trajectories"]}
    tmeta = {}
    with (mp.EXPORT_DIR / "trajectories.jsonl").open() as f:
        for line in f:
            t = json.loads(line); tmeta[t["traj_id"]] = t
    R["side_task_categories"] = dict(collections.Counter(t["side_task_category"] for t in tmeta.values() if t["side"] == "attack"))

    # ── 2. render + tokenize everything ─────────────────────────────────────────────────────
    actions = manifest["actions"]
    t0 = time.time()
    _init(args.template_date, args.reasoning_effort)          # main process copy for examples/diffs
    tok = _W["tok"]; template = _W["template"]
    cache_key = hashlib.sha256(f"{_W['tsha']}|{tmpl_sha}|{manifest['git_commit']}|{Path(mp.__file__).read_bytes().hex()}".encode()).hexdigest()[:16]
    cache = REPO / "checkpoints" / SLUG / f"g2_recs_{args.template_date}_{args.reasoning_effort}_{cache_key}.json"
    if cache.exists():
        recs = json.loads(cache.read_text())
        R["tokenize_cache"] = str(cache.relative_to(REPO))
        print(f"G2.2 loaded per-action records from {cache} (same pinned template, prompt template, manifest and monitor_prompt.py)")
    else:
        with mp_.get_context("fork").Pool(args.workers, initializer=_init, initargs=(args.template_date, args.reasoning_effort)) as pool:
            recs = pool.map(_one, actions, chunksize=16)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(recs))
    R["tokenize_seconds"] = time.time() - t0
    R["pinned_chat_template_sha256"] = _W["tsha"]
    assert len(recs) == len(actions) == 17234
    assert all(r["sha_match"] for r in recs), "some original render != manifest sha256"
    assert all(r["last_id"] == mp.FINAL_TOKEN_ID for r in recs)
    print(f"G2.2 tokenized {len(recs)} prompts in {R['tokenize_seconds']:.0f}s; all sha256 match; all end in id {mp.FINAL_TOKEN_ID}")
    rng = random.Random(args.seed)
    ex10 = rng.sample(recs, 10)
    R["chars_per_token_examples"] = [{"traj_id": r["traj_id"], "action_idx": r["action_idx"], "chars_templated": r["chars_templated"],
                                      "n_tokens": r["n_tokens"], "chars_per_token": r["chars_templated"] / r["n_tokens"]} for r in ex10]
    ratios = [e["chars_per_token"] for e in R["chars_per_token_examples"]]
    assert all(2.5 <= x <= 5.5 for x in ratios), ratios
    print("     chars/token on 10 random examples:", [round(x, 2) for x in ratios], "(assert 2.5..5.5; ~4 expected)")

    # per-action table as columns (compact)
    cols = ["traj_id", "action_idx", "side", "env", "tool", "cached_mean", "n_cached", "chars_user", "chars_templated", "n_tokens",
            "placeholder", "n_context_tokens", "neut_hist_events", "neut_action_events", "command_length", "has_sudo", "has_chmod", "has_curl", "has_nc"]
    R["per_action"] = {c: [r[c] for r in recs] for c in cols}
    R["per_action"]["capture_token_index"] = {k: [r["pos"][i] for r in recs] for i, k in enumerate(mp.POSITIONS)}
    R["per_action"]["_note"] = "columns aligned by index; capture_token_index are token positions in the templated prompt (original render)"

    # ── 3. length stats ─────────────────────────────────────────────────────────────────────
    L = np.array([r["n_tokens"] for r in recs])
    side = np.array([r["side"] for r in recs]); ph = np.array([r["placeholder"] for r in recs])

    def stats(x):
        x = np.asarray(x)
        if len(x) == 0:
            return {"n": 0}
        q = np.percentile(x, [50, 90, 95, 99])
        return {"n": int(len(x)), "median": float(q[0]), "p90": float(q[1]), "p95": float(q[2]), "p99": float(q[3]),
                "max": int(x.max()), "min": int(x.min()), "n_over_128k": int((x > 131072).sum()), "n_over_32k": int((x > 32000).sum())}
    R["lengths"] = {"all": stats(L), "honest": stats(L[side == "honest"]), "attack": stats(L[side == "attack"]),
                    "placeholder": stats(L[ph]), "no_placeholder": stats(L[~ph]),
                    "honest_placeholder": stats(L[(side == "honest") & ph]), "attack_placeholder": stats(L[(side == "attack") & ph]),
                    "n_placeholder_by_side": {"honest": int(((side == "honest") & ph).sum()), "attack": int(((side == "attack") & ph).sum())}}
    print("G2.3 lengths:", json.dumps({k: R["lengths"][k] for k in ("all", "honest", "attack", "placeholder")}, indent=0))

    # ── 7 (fit first, cap needs it). prefill curve ──────────────────────────────────────────
    Lg = np.array([l for l, _ in G1_PREFILL], float); Tg = np.array([t for _, t in G1_PREFILL])
    from scipy.optimize import nnls
    A = np.stack([np.ones_like(Lg), Lg, Lg ** 2], 1)
    w = 1.0 / Tg                                   # weight rows by 1/t -> least squares on relative error
    coef, _ = nnls(A * w[:, None], Tg * w)         # c, a, b >= 0
    pred = A @ coef
    interp = lambda n: float(np.interp(n, Lg, Tg))  # noqa: E731  piecewise-linear through the measured points
    R["prefill_fit"] = {"model": "t = c + a*L + b*L^2, non-negative least squares on relative error (rows weighted 1/t)",
                        "c": float(coef[0]), "a": float(coef[1]), "b": float(coef[2]),
                        "points": [{"L": int(l), "t": float(t), "fit": float(p), "rel_err": float((p - t) / t)} for l, t, p in zip(Lg, Tg, pred)],
                        "max_rel_err": float(np.max(np.abs(pred - Tg) / Tg)), "source": G1_PREFILL_SRC,
                        "per_action_overhead_s": PER_ACTION_OVERHEAD_S,
                        "cross_check": "projection also computed with piecewise-linear interpolation of the measured points (projection.*_interp)"}
    tfun = lambda n: coef[0] + coef[1] * n + coef[2] * n ** 2 + PER_ACTION_OVERHEAD_S   # noqa: E731
    print("G2.7 prefill fit:", {k: R["prefill_fit"][k] for k in ("c", "a", "b", "max_rel_err")}, "t(8k)=%.1f t(16k)=%.1f t(32k)=%.1f" % (tfun(8000), tfun(16000), tfun(32000)))
    interp_hours = lambda rs: sum(interp(r["n_tokens"]) + PER_ACTION_OVERHEAD_S for r in rs) / 3600  # noqa: E731

    # ── 4. examples: last 12 tokens + capture positions ──────────────────────────────────────
    by_key = {(r["traj_id"], r["action_idx"]): r for r in recs}
    ex_h = next(r for r in recs if r["side"] == "honest" and r["action_idx"] >= 5 and r["n_tokens"] < 20000)
    ex_a = next(r for r in recs if r["side"] == "attack" and r["action_idx"] >= 5 and r["n_tokens"] < 20000 and r["neut_action_events"] > 0)
    R["examples"] = {}
    for name, r in (("honest", ex_h), ("attack", ex_a)):
        t = trajs[r["traj_id"]]
        rr = mp.render_user_text(t, r["action_idx"], "original", template)
        s = mp.apply_template(tok, rr["text"], args.template_date, args.reasoning_effort)
        tp = mp.tokenize_with_positions(tok, s, rr["text"], rr)
        ids = tp["input_ids"]; offs = tp["offsets"]
        last12 = [{"index": len(ids) - 12 + i, "id": ids[-12 + i], "decoded": tok.decode([ids[-12 + i]])} for i in range(12)]
        u0 = tp["user_char_span"][0]
        caps = []
        for p in mp.decode_positions(tok, ids, tp["positions"]):
            i = p["token_index"]
            p["char_span_templated"] = list(offs[i])
            p["context_before"] = s[max(0, offs[i][0] - 60):offs[i][0]]
            p["context_after"] = s[offs[i][1]:offs[i][1] + 40]
            caps.append(p)
        R["examples"][name] = {"traj_id": r["traj_id"], "action_idx": r["action_idx"], "env": r["env"], "n_tokens": len(ids),
                               "user_char_span_in_templated": tp["user_char_span"],
                               "history_char_span_user": rr["history_span"], "action_char_span_user": rr["action_span"],
                               "last_12_tokens": last12, "capture_positions": caps, "n_context_tokens": tp["n_context_tokens"],
                               "system_prefix": s[:u0][:400],
                               "history_tail_repr": repr(rr["text"][rr["history_span"][1] - 12:rr["history_span"][1] + 12]),
                               "action_tail_repr": repr(rr["text"][rr["action_span"][1] - 12:rr["action_span"][1] + 20])}
        print(f"G2.4 {name} {r['traj_id'][:8]}#{r['action_idx']} ({len(ids)} tokens) last 12 tokens:")
        for x in last12:
            print(f"       {x['index']:>6d} {x['id']:>7d} {x['decoded']!r}")
        for p in caps:
            print(f"       {p['name']:16s} tok {p['token_index']:>6d} id {p['token_id']:>7d} {p['decoded']!r}  ...{p['context_before'][-40:]!r} | {p['context_after'][:25]!r}")
    assert all(p["token_id"] == mp.FINAL_TOKEN_ID for e in R["examples"].values() for p in e["capture_positions"] if p["name"] == "final")
    save()

    # ── 5. cap + subset ─────────────────────────────────────────────────────────────────────
    traj_recs = collections.defaultdict(list)
    for r in recs:
        traj_recs[r["traj_id"]].append(r)
    for v in traj_recs.values():
        v.sort(key=lambda r: r["action_idx"])
    attack_ids = [k for k in traj_recs if tmeta[k]["side"] == "attack"]
    honest_ids = [k for k in traj_recs if tmeta[k]["side"] == "honest"]
    rng = random.Random(args.seed)
    # attack: >= min_per_category per side-task category, then fill to n_attack
    by_cat = collections.defaultdict(list)
    for k in sorted(attack_ids):
        by_cat[tmeta[k]["side_task_category"]].append(k)
    chosen_a = []
    for cat in sorted(by_cat):
        pool = by_cat[cat][:]; rng.shuffle(pool)
        chosen_a += pool[:args.min_per_category]
    rest_a = [k for k in sorted(attack_ids) if k not in chosen_a]; rng.shuffle(rest_a)
    chosen_a += rest_a[:max(0, args.n_attack - len(chosen_a))]
    # honest: one per env first, then envs in proportion to the chosen attack envs
    by_env = collections.defaultdict(list)
    for k in sorted(honest_ids):
        by_env[tmeta[k]["env"]].append(k)
    for v in by_env.values():
        rng.shuffle(v)
    chosen_h = [by_env[e].pop() for e in sorted(by_env)]
    att_env_counts = collections.Counter(tmeta[k]["env"] for k in chosen_a)
    while len(chosen_h) < args.n_honest:
        e = max(sorted(by_env), key=lambda e: att_env_counts[e] - sum(tmeta[k]["env"] == e for k in chosen_h))
        chosen_h.append(by_env[e].pop())

    def subset_actions(cap, ta, th):
        out = []
        for k in ta + th:
            out += [r for r in traj_recs[k] if r["n_tokens"] <= cap]
        return out

    def projected_hours(rs):
        return sum(tfun(r["n_tokens"]) for r in rs) / 3600

    # cap: largest round number <= 32000 whose projection (main + counterfactual + scrub) fits the budget
    caps = [32000, 30000, 28000, 26000, 24000, 22000, 20000, 18000, 16000, 14000, 12000, 10000, 8000]
    cap_table = []
    cap = None
    for c in caps:
        sub = subset_actions(c, chosen_a, chosen_h)
        h_main = projected_hours(sub)
        cf = sorted(sub, key=lambda r: -r["n_tokens"])[:args.n_counterfactual]   # worst case: the longest ones
        h_cf = projected_hours(cf)
        total = h_main + 2 * h_cf
        cap_table.append({"cap": c, "n_actions": len(sub), "hours_main": h_main, "hours_counterfactual_worst": h_cf, "hours_total_worst": total,
                          "fits": total <= args.budget_hours})
        if cap is None and total <= args.budget_hours:
            cap = c
    R["cap_projection"] = cap_table
    assert cap is not None, "no cap fits the budget"
    # top up trajectories toward target_actions while the budget still fits
    extra_a = [k for k in rest_a if k not in chosen_a]
    extra_h = [k for e in sorted(by_env) for k in by_env[e]]; rng.shuffle(extra_h)
    added = []
    turn = 0
    while len(subset_actions(cap, chosen_a, chosen_h)) < args.target_actions and (extra_a or extra_h):
        k = (extra_a.pop() if (turn % 2 == 0 and extra_a) else extra_h.pop()) if (extra_a and extra_h) else (extra_a or extra_h).pop()
        trial_a, trial_h = (chosen_a + [k], chosen_h) if tmeta[k]["side"] == "attack" else (chosen_a, chosen_h + [k])
        sub = subset_actions(cap, trial_a, trial_h)
        cf = sorted(sub, key=lambda r: -r["n_tokens"])[:args.n_counterfactual]
        if projected_hours(sub) + 2 * projected_hours(cf) > args.budget_hours:
            break
        chosen_a, chosen_h = trial_a, trial_h
        added.append(k); turn += 1
    sub = subset_actions(cap, chosen_a, chosen_h)
    all_traj_actions = [r for k in chosen_a + chosen_h for r in traj_recs[k]]
    n_before = collections.Counter(r["side"] for r in all_traj_actions)
    n_after = collections.Counter(r["side"] for r in sub)
    R["subset_main"] = {
        "cap_tokens": cap, "seed": args.seed, "n_trajectories": {"attack": len(chosen_a), "honest": len(chosen_h)},
        "trajectories_added_for_target": added,
        "attack_trajectories": [{"traj_id": k, "env": tmeta[k]["env"], "side_task": tmeta[k]["side_task"], "category": tmeta[k]["side_task_category"],
                                 "n_actions": len(traj_recs[k]), "n_under_cap": sum(r["n_tokens"] <= cap for r in traj_recs[k]),
                                 "max_tokens": max(r["n_tokens"] for r in traj_recs[k]),
                                 "trigger_action_idx": tmeta[k]["side_task_trigger_action_idx"]} for k in chosen_a],
        "honest_trajectories": [{"traj_id": k, "env": tmeta[k]["env"], "n_actions": len(traj_recs[k]),
                                 "n_under_cap": sum(r["n_tokens"] <= cap for r in traj_recs[k]),
                                 "max_tokens": max(r["n_tokens"] for r in traj_recs[k])} for k in chosen_h],
        "category_coverage": dict(collections.Counter(tmeta[k]["side_task_category"] for k in chosen_a)),
        "env_coverage": {"attack": dict(collections.Counter(tmeta[k]["env"] for k in chosen_a)), "honest": dict(collections.Counter(tmeta[k]["env"] for k in chosen_h))},
        "n_actions_before_cap": dict(n_before), "n_actions_after_cap": dict(n_after),
        "attack_fraction_before_cap": n_before["attack"] / sum(n_before.values()),
        "attack_fraction_after_cap": n_after["attack"] / sum(n_after.values()),
        "pool_attack_fraction": manifest["n_actions_by_side"]["attack"] / manifest["n_actions"],
        "n_dropped_by_cap": {"attack": n_before["attack"] - n_after["attack"], "honest": n_before["honest"] - n_after["honest"]},
        "placeholder_rate": {"subset": sum(r["placeholder"] for r in sub) / len(sub),
                             "subset_by_side": {s: (sum(r["placeholder"] for r in sub if r["side"] == s) / max(1, n_after[s])) for s in ("honest", "attack")},
                             "pool": float(ph.mean())},
        "n_actions": len(sub), "lengths": stats([r["n_tokens"] for r in sub]),
        "projected_hours_main": projected_hours(sub),
        "n_with_neutralize_events": sum((r["neut_hist_events"] + r["neut_action_events"]) > 0 for r in sub),
        "n_with_action_neutralize_events": sum(r["neut_action_events"] > 0 for r in sub),
        "trigger_action_idx_note": "side_task_trigger_action_idx indexes the agent trajectory (n_actions_in_agent_traj); for every chosen attack trajectory where it is known it points at an index absent from the manifest (hidden submit/run_side_task_test tools), so it cannot mark the side-task action here",
        "trigger_idx_known": sum(1 for k in chosen_a if tmeta[k]["side_task_trigger_action_idx"] is not None),
        "trigger_idx_in_manifest": sum(1 for k in chosen_a if tmeta[k]["side_task_trigger_action_idx"] is not None
                                       and any(r["action_idx"] == tmeta[k]["side_task_trigger_action_idx"] for r in traj_recs[k])),
    }
    print("G2.5 cap table:"); [print("      ", c) for c in cap_table]
    print("G2.5 subset_main:", json.dumps({k: v for k, v in R["subset_main"].items() if k not in ("attack_trajectories", "honest_trajectories")}, indent=0, default=str))
    subset_main = {"cap_tokens": cap, "seed": args.seed, "template_date": args.template_date, "reasoning_effort": args.reasoning_effort,
                   "pinned_chat_template_sha256": R["pinned_chat_template_sha256"], "prompt_template_sha256": tmpl_sha, "raw": out_path.name,
                   "order": "ascending n_tokens",
                   "actions": [{"traj_id": r["traj_id"], "action_idx": r["action_idx"], "side": r["side"], "env": r["env"], "n_tokens": r["n_tokens"],
                                "cached_mean": r["cached_mean"], "placeholder": r["placeholder"]} for r in sorted(sub, key=lambda r: (r["n_tokens"], r["traj_id"], r["action_idx"]))]}
    (HERE.parent / "subset_main.json").write_text(json.dumps(subset_main, indent=0))

    # ── 6. counterfactual subset + diffs ────────────────────────────────────────────────────
    strata = {"action": ([r for r in sub if r["neut_action_events"] > 0], args.n_cf_action),
              "history_only": ([r for r in sub if r["neut_action_events"] == 0 and r["neut_hist_events"] > 0], args.n_cf_history)}
    chosen_cf = []
    strata_report = {}
    for name, (elig, n_want) in strata.items():
        cells = collections.defaultdict(list)
        for r in elig:
            cells[(r["side"], bucket(r["cached_mean"]))].append(r)
        for v in cells.values():
            v.sort(key=lambda r: (r["traj_id"], r["action_idx"])); rng.shuffle(v)
        keys = sorted(cells)
        per = n_want // len(keys)
        picked = []
        for k in keys:
            picked += cells[k][:per]; cells[k] = cells[k][per:]
        left = [r for k in keys for r in cells[k]]; rng.shuffle(left)
        picked += left[:max(0, n_want - len(picked))]
        for r in picked:
            r["counterfactual_stratum"] = name
        chosen_cf += picked
        strata_report[name] = {"n_eligible_in_main": len(elig), "n": len(picked), "n_by_side": dict(collections.Counter(r["side"] for r in picked)),
                               "cells": {f"{s}/{b}": n for (s, b), n in sorted(collections.Counter((r["side"], bucket(r["cached_mean"])) for r in picked).items())}}
    cf_cells = collections.Counter((r["side"], bucket(r["cached_mean"])) for r in chosen_cf)
    R["subset_counterfactual"] = {
        "n": len(chosen_cf), "strata": strata_report,
        "eligibility": "action stratum: >=1 neutralization event in the current action; history_only stratum: none in the current action, >=1 in the history; both drawn from subset_main, stratified over side x cached-score bucket",
        "cells": {f"{s}/{b}": n for (s, b), n in sorted(cf_cells.items())}, "score_buckets": SCORE_BUCKETS,
        "n_with_action_events": sum(r["neut_action_events"] > 0 for r in chosen_cf),
        "projected_hours_per_render": projected_hours(chosen_cf),
        "neutralize_table": mp._NEUTRALIZE_SRC + [("<non-private non-loopback IPv4>", "127.0.0.1")],
        "neutralize_omitted": mp.NEUTRALIZE_OMITTED,
        "random_scrub_table": mp._RANDOM_SCRUB_SRC,
        "applies_to": "history block and current_action block only (agent-authored text); task description, covert objectives and template untouched",
    }
    # random_scrub feasibility: how many events could actually be matched
    scrub_short = 0
    for r in chosen_cf:
        rr = mp.render_user_text(trajs[r["traj_id"]], r["action_idx"], "random_scrub", template)
        if rr["subs"]["history_events"] < rr["subs"]["target_history_events"] or rr["subs"]["action_events"] < rr["subs"]["target_action_events"]:
            scrub_short += 1
    R["subset_counterfactual"]["n_random_scrub_under_matched"] = scrub_short
    subset_cf = {**{k: subset_main[k] for k in ("cap_tokens", "seed", "template_date", "reasoning_effort", "pinned_chat_template_sha256", "prompt_template_sha256", "raw")},
                 "parent": "subset_main.json", "renders": ["neutralized", "random_scrub"],
                 "actions": [{"traj_id": r["traj_id"], "action_idx": r["action_idx"], "side": r["side"], "env": r["env"], "n_tokens": r["n_tokens"],
                              "counterfactual_stratum": r["counterfactual_stratum"],
                              "cached_mean": r["cached_mean"], "score_bucket": bucket(r["cached_mean"]), "placeholder": r["placeholder"],
                              "neut_hist_events": r["neut_hist_events"], "neut_action_events": r["neut_action_events"]}
                             for r in sorted(chosen_cf, key=lambda r: (r["n_tokens"], r["traj_id"], r["action_idx"]))]}
    (HERE.parent / "subset_counterfactual.json").write_text(json.dumps(subset_cf, indent=0))
    # 10 diffs: 5 with current-action changes (prefer), 5 others; alternate sides
    def alternate(rs, k):
        att = [r for r in rs if r["side"] == "attack"]; hon = [r for r in rs if r["side"] == "honest"]
        out = []
        while len(out) < k and (att or hon):
            if att: out.append(att.pop(0))
            if hon and len(out) < k: out.append(hon.pop(0))
        return out
    pick = alternate([r for r in chosen_cf if r["counterfactual_stratum"] == "action"], 5) + alternate([r for r in chosen_cf if r["counterfactual_stratum"] == "history_only"], 5)
    diffs = []
    for r in pick:
        t = trajs[r["traj_id"]]
        o = mp.render_user_text(t, r["action_idx"], "original", template)
        n = mp.render_user_text(t, r["action_idx"], "neutralized", template)
        sc = mp.render_user_text(t, r["action_idx"], "random_scrub", template)
        ud = [l for l in difflib.unified_diff(o["text"].splitlines(), n["text"].splitlines(), "original", "neutralized", n=0, lineterm="")]
        us = [l for l in difflib.unified_diff(o["text"].splitlines(), sc["text"].splitlines(), "original", "random_scrub", n=0, lineterm="")]
        clip = lambda ls, k=30: [l[:300] for l in ls[:k]] + ([f"... ({len(ls) - k} more diff lines)"] if len(ls) > k else [])  # noqa: E731
        diffs.append({"traj_id": r["traj_id"], "action_idx": r["action_idx"], "side": r["side"], "env": r["env"], "cached_mean": r["cached_mean"], "stratum": r["counterfactual_stratum"],
                      "side_task": tmeta[r["traj_id"]]["side_task"], "events": n["subs"], "scrub_events": sc["subs"],
                      "current_action_original": o["text"][o["action_span"][0]:o["action_span"][1]][:1500],
                      "current_action_neutralized": n["text"][n["action_span"][0]:n["action_span"][1]][:1500],
                      "diff_neutralized": clip(ud), "diff_random_scrub": clip(us, 12)})
    R["neutralization_diffs"] = diffs
    print("G2.6 counterfactual:", json.dumps({k: v for k, v in R["subset_counterfactual"].items() if "table" not in k}, indent=0))
    for d in diffs:
        print(f"\n===== [{d['stratum']}] {d['side']} {d['traj_id'][:8]}#{d['action_idx']} env={d['env']} side_task={d['side_task']} cached={d['cached_mean']} events={d['events']}")
        print("  CURRENT ACTION (original):   ", d["current_action_original"][:400].replace("\n", "\n      "))
        print("  CURRENT ACTION (neutralized):", d["current_action_neutralized"][:400].replace("\n", "\n      "))
        print("  DIFF original -> neutralized (context 0):")
        for l in d["diff_neutralized"]:
            print("     ", l[:200])
        print("  DIFF original -> random_scrub (first lines):")
        for l in d["diff_random_scrub"][:6]:
            print("     ", l[:200])

    # ── 7. projection ───────────────────────────────────────────────────────────────────────
    R["projection"] = {"main_hours": projected_hours(sub), "counterfactual_hours": projected_hours(chosen_cf), "random_scrub_hours": projected_hours(chosen_cf),
                       "total_hours": projected_hours(sub) + 2 * projected_hours(chosen_cf), "budget_hours": args.budget_hours,
                       "main_hours_interp": interp_hours(sub), "counterfactual_hours_interp": interp_hours(chosen_cf),
                       "total_hours_interp": interp_hours(sub) + 2 * interp_hours(chosen_cf),
                       "formula": "sum_i (a*L_i + b*L_i^2 + overhead) with a,b from prefill_fit; overhead is a guess G3 replaces",
                       "main_hours_by_length_bin": {}}
    edges = [0, 4000, 8000, 12000, 16000, 20000, 24000, 28000, 32000]
    for lo, hi in zip(edges[:-1], edges[1:]):
        rs = [r for r in sub if lo < r["n_tokens"] <= hi]
        R["projection"]["main_hours_by_length_bin"][f"{lo}-{hi}"] = {"n": len(rs), "hours": projected_hours(rs)}
    print("G2.7 projection:", json.dumps(R["projection"], indent=0))

    # ── figure ──────────────────────────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    bins = np.linspace(0, 145000, 59)
    for ax, (mask_name, m1, m2, c1, c2) in zip(axes, [("side", side == "honest", side == "attack", "#2a78d6", "#eb6834"),
                                                      ("placeholder", ~ph, ph, "#2a78d6", "#eb6834")]):
        labels = ("honest", "attack") if mask_name == "side" else ("no placeholder", "attachment:// placeholder")
        ax.hist(L[m1], bins=bins, color=c1, alpha=0.85, label=f"{labels[0]} (n={int(m1.sum())})", linewidth=0)
        ax.hist(L[m2], bins=bins, color=c2, alpha=0.75, label=f"{labels[1]} (n={int(m2.sum())})", linewidth=0)
        ax.axvline(cap, color="#444", ls="--", lw=1.2); ax.text(cap, ax.get_ylim()[1] * 0.95, f" cap {cap}", va="top", fontsize=9, color="#444")
        ax.axvline(131072, color="#999", ls=":", lw=1); ax.text(131072, ax.get_ylim()[1] * 0.95, " 128k ", va="top", ha="right", fontsize=8, color="#777")
        ax.set_xlabel("templated prompt length (tokens)"); ax.set_ylabel("actions"); ax.legend(frameon=False, fontsize=9)
        ax.grid(axis="y", color="#e5e5e5", lw=0.6); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle(f"G2 token lengths, all {len(L)} manifest actions, gpt-oss-20b template (date {args.template_date}, reasoning {args.reasoning_effort})", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "g2_lengths.png", dpi=150); plt.close(fig)
    R["figure"] = str((FIG / "g2_lengths.png").relative_to(REPO))
    save()
    print("wrote", out_path, f"({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
