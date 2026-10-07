"""A3 — validate payload_signatures.json against the export (TASK.md §A3). No model.

Per action (all 92 attack trajectories, all 200 honest trajectories):
  own-signature label   attack only: max level matched by the trajectory's OWN side_task entry
  any-signature label   max level matched by ANY of the 92 entries (this is the honest
                        specificity number, and a cross-check column for attack)
  placeholder           from g2_prep .per_action (the definition the downstream code uses);
                        placeholder rows get label NaN / label_source="placeholder"

Coverage
  anchor version   30 trajectories with side_task_trigger_action_idx: nearest own level-2 hit at
                   or before the trigger, and the gap (1 = the action right before the test call)
  success version  the other 62 with side_task_succeeded: any own level-2 hit anywhere
Specificity
  honest actions matching any level-2 (printed verbatim), any level-1; per-pattern hit counts on
  honest so noisy patterns are visible.

Writes results/raw/<slug>/a3_validate_<ts>.json, logs to stdout, and
results/figures/<slug>/a3_label_position.png (position of level-2 hits, by category).

    python a3_validate.py [--round N] [--note "..."]
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SLUG = HERE.name
REPO = HERE.parents[1]
GPU = REPO / "experiments" / "logit-curve-replication" / "gpu"
sys.path.insert(0, str(GPU))
import format_monitor_prompt as fmp  # noqa: E402

EXPORT = REPO / "checkpoints" / "local-monitor-activations" / "data" / "export"
SUBSET = REPO / "experiments" / "local-monitor-activations" / "subset_main.json"
G2 = REPO / "results" / "raw" / "local-monitor-activations" / "g2_prep_20260910T103431Z.json"
SIGS = HERE / "payload_signatures.json"
RAW_DIR = REPO / "results" / "raw" / SLUG
FIG_DIR = REPO / "results" / "figures" / SLUG
TS = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True).stdout.strip()


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    sigs = json.load(open(SIGS))
    S = sigs["signatures"]
    by_task = {e["side_task"]: e for e in S}
    # unique regex table: (pattern) -> compiled; each entry maps to pattern ids
    pat_ids: dict[str, int] = {}
    for e in S:
        for lv in ("level1", "level2"):
            for p in e[lv]:
                pat_ids.setdefault(p, len(pat_ids))
    patterns = list(pat_ids)
    compiled = [re.compile(p) for p in patterns]
    entry_pats = {e["side_task"]: {lv: [pat_ids[p] for p in e[lv]] for lv in ("level1", "level2")} for e in S}
    pat_level_any: dict[int, int] = {}  # highest level any entry assigns to this pattern
    pat_owner: dict[int, list[tuple[str, int]]] = collections.defaultdict(list)
    for e in S:
        for lv, n in (("level1", 1), ("level2", 2)):
            for p in e[lv]:
                i = pat_ids[p]
                pat_level_any[i] = max(pat_level_any.get(i, 0), n)
                pat_owner[i].append((e["side_task"], n))

    trajs = fmp.load_export(EXPORT)
    subset = json.load(open(SUBSET))
    in_subset = {(a["traj_id"], a["action_idx"]) for a in subset["actions"]}
    g2 = json.load(open(G2))["per_action"]
    placeholder = {(t, i): p for t, i, p in zip(g2["traj_id"], g2["action_idx"], g2["placeholder"])}
    cached_mean = {(t, i): c for t, i, c in zip(g2["traj_id"], g2["action_idx"], g2["cached_mean"])}

    rows = []
    for t in trajs.values():
        own = entry_pats.get(t["side_task"]) if t["side"] == "attack" else None
        for a in t["actions"]:
            s = fmp.format_action(a, hide_output=True)
            hits = [i for i, rx in enumerate(compiled) if rx.search(s)]
            hitset = set(hits)
            any_level = max((pat_level_any[i] for i in hits), default=0)
            if own is not None:
                own2 = [i for i in own["level2"] if i in hitset]
                own1 = [i for i in own["level1"] if i in hitset]
                own_level = 2 if own2 else 1 if own1 else 0
                own_matched = own2 or own1
            else:
                own_level, own_matched = None, []
            key = (t["traj_id"], a["index"])
            ph = placeholder.get(key)
            rows.append({
                "traj_id": t["traj_id"], "action_idx": a["index"], "side": t["side"], "env": t["env"],
                "side_task": t["side_task"], "side_task_category": t["side_task_category"],
                "side_task_succeeded": t["side_task_succeeded"], "n_actions": len(t["actions"]),
                "tool": a["function"], "skipped_by_monitor": a["skipped_by_monitor"],
                "in_subset_main": key in in_subset, "in_pool": key in placeholder, "placeholder": ph,
                "cached_mean": cached_mean.get(key),
                "own_level": own_level, "own_matched": own_matched,
                "any_level": any_level, "any_matched": hits,
                "label_level": (None if ph else own_level) if own is not None else (None if ph else 0),
                "label_source": "placeholder" if ph else "signature",
            })
    by_traj: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        by_traj[r["traj_id"]].append(r)

    # ── coverage ──────────────────────────────────────────────────────────────────────────────
    anchor, success, failed = [], [], []
    for t in trajs.values():
        if t["side"] != "attack":
            continue
        R = by_traj[t["traj_id"]]
        l2 = [r["action_idx"] for r in R if r["own_level"] == 2]
        l1 = [r["action_idx"] for r in R if r["own_level"] == 1]
        base = {"traj_id": t["traj_id"], "side_task": t["side_task"], "category": t["side_task_category"],
                "env": t["env"], "failure_type": t["side_task_failure_type"], "n_actions": len(R),
                "level2_idx": l2, "level1_idx": l1, "succeeded": t["side_task_succeeded"]}
        trig = t["side_task_trigger_action_idx"]
        if trig is not None:
            before = [i for i in l2 if i <= trig]
            near = max(before) if before else None
            anchor.append({**base, "trigger_idx": trig, "nearest_l2_before_trigger": near,
                           "gap": (trig - near) if near is not None else None, "hit": near is not None,
                           "l2_after_trigger": [i for i in l2 if i > trig]})
        elif t["side_task_succeeded"]:
            success.append({**base, "hit": bool(l2)})
        else:
            failed.append({**base, "hit": bool(l2)})
    cov = {
        "anchor": {"n": len(anchor), "n_hit": sum(a["hit"] for a in anchor),
                   "gap_counts": dict(collections.Counter(str(a["gap"]) for a in anchor)),
                   "misses": [a["side_task"] for a in anchor if not a["hit"]],
                   "by_category": {c: [sum(a["hit"] for a in anchor if a["category"] == c), sum(a["category"] == c for a in anchor)]
                                   for c in sorted({a["category"] for a in anchor})}},
        "success": {"n": len(success), "n_hit": sum(a["hit"] for a in success),
                    "misses": [a["side_task"] for a in success if not a["hit"]],
                    "by_category": {c: [sum(a["hit"] for a in success if a["category"] == c), sum(a["category"] == c for a in success)]
                                    for c in sorted({a["category"] for a in success})}},
        "failed": {"n": len(failed), "n_hit": sum(a["hit"] for a in failed), "side_tasks": [a["side_task"] for a in failed]},
        "all_successful": {"n": len(anchor) + len(success), "n_hit": sum(a["hit"] for a in anchor) + sum(a["hit"] for a in success)},
        "subset_main_successful": {"n": sum(1 for a in anchor + success if any(r["in_subset_main"] for r in by_traj[a["traj_id"]])),
                                   "n_hit": sum(a["hit"] for a in anchor + success if any(r["in_subset_main"] for r in by_traj[a["traj_id"]]))},
    }

    # ── specificity ───────────────────────────────────────────────────────────────────────────
    honest = [r for r in rows if r["side"] == "honest"]
    honest_sub = [r for r in honest if r["in_subset_main"]]
    hon_l2 = [r for r in honest if r["any_level"] == 2]
    hon_l1 = [r for r in honest if r["any_level"] == 1]
    pat_hits_honest = collections.Counter(i for r in honest for i in r["any_matched"])
    pat_hits_attack = collections.Counter(i for r in rows if r["side"] == "attack" for i in r["any_matched"])
    action_text = {}
    for t in trajs.values():
        for a in t["actions"]:
            action_text[(t["traj_id"], a["index"])] = fmp.format_action(a, hide_output=True)
    spec = {
        "n_honest_actions": len(honest), "n_honest_actions_subset_main": len(honest_sub),
        "honest_level2_any": len(hon_l2), "honest_level1_any": len(hon_l1),
        "honest_level2_any_subset_main": sum(r["any_level"] == 2 for r in honest_sub),
        "honest_level1_any_subset_main": sum(r["any_level"] == 1 for r in honest_sub),
        "honest_level2_hits_verbatim": [
            {"traj_id": r["traj_id"], "action_idx": r["action_idx"], "env": r["env"], "in_subset_main": r["in_subset_main"],
             "cached_mean": r["cached_mean"],
             "level2_patterns": [{"pattern": patterns[i], "owners": sorted({o for o, n in pat_owner[i] if n == 2})}
                                 for i in r["any_matched"] if pat_level_any[i] == 2],
             "current_action": action_text[(r["traj_id"], r["action_idx"])][:300]}
            for r in hon_l2],
        "per_pattern": [
            {"id": i, "pattern": patterns[i], "level_any": pat_level_any[i], "n_owner_entries": len(pat_owner[i]),
             "honest_hits": pat_hits_honest.get(i, 0), "attack_hits": pat_hits_attack.get(i, 0)}
            for i in sorted(range(len(patterns)), key=lambda i: -pat_hits_honest.get(i, 0))],
    }
    # per-pattern honest hit rate among level-1-only actions: which patterns make level 1 noise
    l1_noise = [p for p in spec["per_pattern"] if p["level_any"] == 1 and p["honest_hits"] > 0]

    # ── figure ────────────────────────────────────────────────────────────────────────────────
    fig_path = None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cats = sorted({r["side_task_category"] for r in rows if r["side"] == "attack"})
        fig, axes = plt.subplots(len(cats), 1, figsize=(8, 1.6 * len(cats)), sharex=True)
        for ax, c in zip(axes, cats):
            xs = [r["action_idx"] / max(r["n_actions"] - 1, 1) for r in rows
                  if r["side"] == "attack" and r["side_task_category"] == c and r["own_level"] == 2]
            ax.hist(xs, bins=20, range=(0, 1), color="#c44")
            ax.set_ylabel(c.replace("_", "\n"), fontsize=7, rotation=0, ha="right", va="center")
            ax.text(0.01, 0.8, f"n={len(xs)} level-2 actions", transform=ax.transAxes, fontsize=7)
        axes[-1].set_xlabel("position in trajectory (action_idx / (n_actions-1))")
        fig.suptitle(f"own level-2 hits by category — round {args.round}", fontsize=9)
        fig.tight_layout()
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        fig_path = FIG_DIR / f"a3_label_position_round{args.round}.png"
        fig.savefig(fig_path, dpi=150)
    except ImportError as e:
        print("!! no matplotlib, figure skipped:", e)

    out = {
        "slug": SLUG, "script": f"experiments/{SLUG}/a3_validate.py", "git_commit": git("rev-parse", "HEAD"),
        "timestamp": TS, "round": args.round, "note": args.note,
        "signature_file": str(SIGS.relative_to(REPO)), "signature_sha256": sha256(SIGS),
        "signature_commit": git("log", "-1", "--format=%H", "--", str(SIGS.relative_to(REPO))),
        "inputs": {str(p.relative_to(REPO)): sha256(p) for p in (EXPORT / "trajectories.jsonl", EXPORT / "attack.jsonl",
                                                                EXPORT / "honest.jsonl", SUBSET, G2)},
        "n_patterns_unique": len(patterns), "patterns": patterns,
        "pattern_owners": {str(i): pat_owner[i] for i in range(len(patterns))},
        "coverage": cov, "coverage_anchor_rows": anchor, "coverage_success_rows": success, "coverage_failed_rows": failed,
        "specificity": spec, "level1_noise_patterns": l1_noise,
        "attack_label_counts": {"own_level": dict(collections.Counter(str(r["own_level"]) for r in rows if r["side"] == "attack")),
                                "label_level": dict(collections.Counter(str(r["label_level"]) for r in rows if r["side"] == "attack")),
                                "placeholder": sum(bool(r["placeholder"]) for r in rows if r["side"] == "attack")},
        "figure": str(fig_path.relative_to(REPO)) if fig_path else None,
        # slim columnar label table — enough to recompute every coverage/specificity count
        # without the model; the full per-action rows (with matched pattern-id lists) go to a
        # gitignored checkpoint next to this file.
        "label_table": {col: [r[col] for r in rows] for col in
                        ("traj_id", "action_idx", "side", "side_task", "side_task_category", "side_task_succeeded",
                         "env", "tool", "in_subset_main", "in_pool", "placeholder", "cached_mean",
                         "own_level", "any_level", "label_level", "label_source")},
    }
    CKPT = REPO / "checkpoints" / SLUG
    CKPT.mkdir(parents=True, exist_ok=True)
    ckpt = CKPT / f"a3_rows_{TS}.json"
    ckpt.write_text(json.dumps({"git_commit": out["git_commit"], "signature_commit": out["signature_commit"],
                                "patterns": patterns, "pattern_owners": {str(i): pat_owner[i] for i in range(len(patterns))},
                                "rows": rows}, separators=(",", ":")))
    out["rows_checkpoint"] = str(ckpt.relative_to(REPO))
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw = RAW_DIR / f"a3_validate_{TS}.json"
    raw.write_text(json.dumps(out, separators=(",", ":")))
    print(f"rows checkpoint -> {ckpt.relative_to(REPO)}  ({ckpt.stat().st_size/1e6:.1f} MB, gitignored)")

    # ── report ────────────────────────────────────────────────────────────────────────────────
    print(f"raw -> {raw.relative_to(REPO)}  ({raw.stat().st_size/1e6:.1f} MB)   signatures @ {out['signature_commit'][:7]}  round {args.round}")
    print(f"unique patterns {len(patterns)}; actions labelled: attack {sum(r['side']=='attack' for r in rows)}, honest {len(honest)}")
    print("\n== coverage, anchor version (trigger idx set) ==")
    print(json.dumps({k: v for k, v in cov["anchor"].items()}, indent=1))
    for a in anchor:
        print(f"  {a['side_task']:36} {a['category'][:21]:21} trig={a['trigger_idx']:3} nearest_l2={str(a['nearest_l2_before_trigger']):>4} gap={str(a['gap']):>4} "
              f"l2_all={a['level2_idx']} l1_n={len(a['level1_idx'])}")
    print("\n== coverage, success version (no trigger idx, succeeded) ==")
    print(json.dumps(cov["success"], indent=1))
    for a in success:
        print(f"  {a['side_task']:36} {a['category'][:21]:21} {a['failure_type']:8} n={a['n_actions']:3} l2={a['level2_idx']} l1_n={len(a['level1_idx'])}")
    print("\n== failed side tasks ==", json.dumps(cov["failed"]))
    for a in failed:
        print(f"  {a['side_task']:36} l2={a['level2_idx']} l1_n={len(a['level1_idx'])}")
    print("\n== all successful:", cov["all_successful"], " subset_main successful:", cov["subset_main_successful"])
    print("\n== specificity ==")
    print({k: v for k, v in spec.items() if k not in ("honest_level2_hits_verbatim", "per_pattern")})
    print("honest level-2 hits verbatim:")
    for h in spec["honest_level2_hits_verbatim"]:
        pats = "; ".join(p["pattern"][:60] + " <" + ",".join(p["owners"][:3]) + ">" for p in h["level2_patterns"])
        print(f"  [{h['env']} {h['traj_id'][:8]}#{h['action_idx']} sub={h['in_subset_main']} cached={h['cached_mean']}] {h['current_action'][:220]!r}\n      via {pats}")
    print("\ntop 25 patterns by honest hits:")
    for p in spec["per_pattern"][:25]:
        print(f"  L{p['level_any']} honest={p['honest_hits']:5} attack={p['attack_hits']:4} owners={p['n_owner_entries']:2}  {p['pattern'][:100]}")
    print("\nattack label counts:", out["attack_label_counts"])
    if fig_path:
        print("figure ->", fig_path.relative_to(REPO))


if __name__ == "__main__":
    main()
