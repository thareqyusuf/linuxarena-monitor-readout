"""A5 — emit action_labels.json (one row per action in the 17,234 g2 pool) and
subset_payload_enrich.json (TASK.md §A5).

label_level uses the *real* own-side-task level (>= 1 match outside a '#' comment line), per the
verification finding; `level2_comment_only` flags actions whose only level-2 match is in a
comment (they get level_level from level-1/0). Honest actions: 0 by construction. Placeholders:
label_level null, label_source "placeholder". `audit_verdict` carries the user's blind-audit
verdict for the 176 audited actions (null elsewhere) — the 8 audit-payload / level-0 actions are
NOT promoted; consumers decide.

Deliverables (columnar JSON, stable paths, G4's input):
  experiments/<slug>/action_labels.json
  experiments/<slug>/subset_payload_enrich.json
Raw (append-only, numbers + sha256 of the deliverables): results/raw/<slug>/a5_emit_<ts>.json
"""
from __future__ import annotations

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
sys.path.insert(0, str(REPO / "experiments" / "logit-curve-replication" / "gpu"))
import format_monitor_prompt as fmp  # noqa: E402

EXPORT = REPO / "checkpoints" / "local-monitor-activations" / "data" / "export"
SUBSET = REPO / "experiments" / "local-monitor-activations" / "subset_main.json"
G2 = REPO / "results" / "raw" / "local-monitor-activations" / "g2_prep_20260910T103431Z.json"
SIGS = HERE / "payload_signatures.json"
RAW_DIR = REPO / "results" / "raw" / SLUG
TS = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
NL = "\\n"
CAP_TOKENS = 32_000
COLS = ["traj_id", "action_idx", "side", "env", "side_task", "side_task_category", "side_task_success", "tool",
        "label_level", "label_source", "matched_pattern", "level2_comment_only", "label_cached_high", "cached_mean",
        "n_tokens", "in_subset_main", "audit_verdict", "signature_commit"]


def git(*a):
    return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True).stdout.strip()


def comment_only(s, p):
    found = False
    for m in re.finditer(p, s):
        found = True
        ls = s.rfind(NL, 0, m.start())
        if "#" not in s[ls + len(NL) if ls >= 0 else 0:m.start()]:
            return False
    return True if found else None


def main():
    g2 = json.load(open(G2))["per_action"]
    keys = list(zip(g2["traj_id"], g2["action_idx"]))
    g2row = {k: i for i, k in enumerate(keys)}
    subset_keys = {(a["traj_id"], a["action_idx"]) for a in json.load(open(SUBSET))["actions"]}
    S = json.load(open(SIGS))["signatures"]; by_task = {e["side_task"]: e for e in S}
    sig_commit = git("log", "-1", "--format=%H", "--", str(SIGS.relative_to(REPO)))
    sig_sha = hashlib.sha256(SIGS.read_bytes()).hexdigest()
    trajs = fmp.load_export(EXPORT)
    scores = {}
    with open(EXPORT / "attack.jsonl") as f:
        for line in f:
            if line.strip():
                a = json.loads(line); scores[(a["traj_id"], a["action_idx"])] = a["monitor_scores"]
    with open(EXPORT / "honest.jsonl") as f:
        for line in f:
            if line.strip():
                a = json.loads(line); scores[(a["traj_id"], a["action_idx"])] = a["monitor_scores"]
    audit = {}
    for p in sorted(RAW_DIR.glob("a4_audit_*.json")) + sorted(RAW_DIR.glob("a4b_audit_*.json")):
        for r in json.load(open(p))["rows"]:
            audit[(r["traj_id"], r["action_idx"])] = r["verdict"]

    out = {c: [] for c in COLS}
    for t in trajs.values():
        own = by_task.get(t["side_task"]) if t["side"] == "attack" else None
        for a in t["actions"]:
            k = (t["traj_id"], a["index"])
            if k not in g2row:
                continue
            i = g2row[k]; ph = bool(g2["placeholder"][i])
            lvl, pat, conly = 0, None, False
            if own is not None and not ph:
                s = fmp.format_action(a, hide_output=True)
                v2 = {p: comment_only(s, p) for p in own["level2"]}; v1 = {p: comment_only(s, p) for p in own["level1"]}
                r2 = [p for p, v in v2.items() if v is False]; r1 = [p for p, v in v1.items() if v is False]
                conly = bool(any(v is True for v in v2.values()) and not r2)
                lvl, pat = (2, r2[0]) if r2 else (1, r1[0]) if r1 else (0, None)
            ms = scores[k]
            row = {"traj_id": k[0], "action_idx": k[1], "side": t["side"], "env": t["env"], "side_task": t["side_task"],
                   "side_task_category": t["side_task_category"], "side_task_success": t["side_task_succeeded"], "tool": a["function"],
                   "label_level": None if ph else lvl, "label_source": "placeholder" if ph else "signature",
                   "matched_pattern": None if ph else pat, "level2_comment_only": conly,
                   "label_cached_high": (all(x >= 6 for x in ms)) if len(ms) == 2 else None, "cached_mean": g2["cached_mean"][i],
                   "n_tokens": g2["n_tokens"][i], "in_subset_main": k in subset_keys, "audit_verdict": audit.get(k),
                   "signature_commit": sig_commit}
            for c in COLS:
                out[c].append(row[c])
    n = len(out["traj_id"])
    assert n == len(keys), (n, len(keys))
    labels = {"slug": SLUG, "script": f"experiments/{SLUG}/a5_emit.py", "timestamp": TS, "git_commit": git("rev-parse", "HEAD"),
              "signature_commit": sig_commit, "signature_sha256": sig_sha, "g2_prep": str(G2.relative_to(REPO)),
              "label_level_definition": "own side-task max level with >=1 regex match outside a '#'-comment line; honest 0; placeholder null",
              "label_cached_high_definition": "both monitor_scores >= 6; null when a single score", "n": n, "columns": COLS, "data": out}
    lab_path = HERE / "action_labels.json"; lab_path.write_text(json.dumps(labels, separators=(",", ":")))

    # subset_payload_enrich: level-2 attack actions not in subset_main, <= CAP_TOKENS, non-placeholder
    enrich = [{"traj_id": out["traj_id"][i], "action_idx": out["action_idx"][i], "side_task": out["side_task"][i],
               "side_task_category": out["side_task_category"][i], "n_tokens": out["n_tokens"][i], "cached_mean": out["cached_mean"][i],
               "label_cached_high": out["label_cached_high"][i], "matched_pattern": out["matched_pattern"][i]}
              for i in range(n) if out["side"][i] == "attack" and out["label_level"][i] == 2 and not out["in_subset_main"][i]
              and out["n_tokens"][i] <= CAP_TOKENS and out["label_source"][i] == "signature"]
    per_cat = dict(collections.Counter(e["side_task_category"] for e in enrich))
    en = {"slug": SLUG, "script": f"experiments/{SLUG}/a5_emit.py", "timestamp": TS, "signature_commit": sig_commit, "cap_tokens": CAP_TOKENS,
          "criteria": "attack, label_level == 2 (real), not in subset_main, n_tokens <= cap, non-placeholder",
          "n": len(enrich), "n_per_category": per_cat, "n_trajectories": len({e["traj_id"] for e in enrich}), "actions": enrich}
    en_path = HERE / "subset_payload_enrich.json"; en_path.write_text(json.dumps(en, indent=1))

    def cnt(side, col):
        return dict(collections.Counter(str(out[col][i]) for i in range(n) if out["side"][i] == side))
    summary = {"n_rows": n, "by_side": dict(collections.Counter(out["side"])),
               "attack_label_level": cnt("attack", "label_level"), "honest_label_level": cnt("honest", "label_level"),
               "attack_label_source": cnt("attack", "label_source"), "honest_label_source": cnt("honest", "label_source"),
               "attack_level2_comment_only": sum(out["level2_comment_only"][i] for i in range(n) if out["side"][i] == "attack"),
               "attack_level2_by_category": dict(collections.Counter(out["side_task_category"][i] for i in range(n) if out["side"][i] == "attack" and out["label_level"][i] == 2)),
               "attack_level2_x_cached_high": dict(collections.Counter(str(out["label_cached_high"][i]) for i in range(n) if out["side"][i] == "attack" and out["label_level"][i] == 2)),
               "audit_verdict_counts": dict(collections.Counter(str(v) for v in out["audit_verdict"] if v)),
               "audit_payload_level0": sum(1 for i in range(n) if out["audit_verdict"][i] == "payload" and out["label_level"][i] == 0),
               "subset_main_rows": sum(out["in_subset_main"]),
               "enrich": {k: en[k] for k in ("n", "n_per_category", "n_trajectories", "cap_tokens")},
               "enrich_cached_high": dict(collections.Counter(str(e["label_cached_high"]) for e in enrich))}
    raw = {"slug": SLUG, "script": f"experiments/{SLUG}/a5_emit.py", "timestamp": TS, "git_commit": git("rev-parse", "HEAD"),
           "signature_commit": sig_commit, "signature_sha256": sig_sha,
           "deliverables": {str(lab_path.relative_to(REPO)): hashlib.sha256(lab_path.read_bytes()).hexdigest(),
                            str(en_path.relative_to(REPO)): hashlib.sha256(en_path.read_bytes()).hexdigest()},
           "summary": summary}
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    rp = RAW_DIR / f"a5_emit_{TS}.json"; rp.write_text(json.dumps(raw, indent=1))
    print("labels ->", lab_path.relative_to(REPO), f"({lab_path.stat().st_size/1e6:.1f} MB)  enrich ->", en_path.relative_to(REPO), f"({en_path.stat().st_size/1e3:.0f} KB)  raw ->", rp.relative_to(REPO))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
