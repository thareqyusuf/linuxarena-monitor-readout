"""Verifier check for A3 (laptop, no model): which level-2 hits are comment-only?

A level-2 regex match is "comment-only" if every match of that pattern in the rendered
action lies on a line whose text before the match contains a `#`. For each attack
trajectory, coverage is recomputed counting only hits that have at least one non-comment
match, and compared with a3_validate's own coverage. Runs both signature versions
(round 1 = frozen a79e588 via `git show`, round 2 = working tree) so both raws are covered.

Writes results/raw/<slug>/verify_a3_comment_matches_<ts>.json with the per-trajectory
rows, the verbatim actions that drive the delete_model finding, and honest-side counts.
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
SIGS = HERE / "payload_signatures.json"
FROZEN_REF = "a79e588"
RAWS = {1: "a3_validate_20260911T070407Z.json", 2: "a3_validate_20260911T070300Z.json"}
TS = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
NL = "\\n"  # format_action renders the tool_input dict with repr(), so newlines are literal backslash-n


def git(*a: str) -> str:
    return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True).stdout


def comment_only(s: str, p: str) -> bool | None:
    """None if p does not match s; True if every match sits after a '#' on its line."""
    found = False
    for m in re.finditer(p, s):
        found = True
        line_start = s.rfind(NL, 0, m.start())
        prefix = s[line_start + len(NL) if line_start >= 0 else 0:m.start()]
        if "#" not in prefix:
            return False
    return True if found else None


def analyse(sig_text: str, raw_name: str, trajs, in_subset):
    S = json.loads(sig_text)["signatures"]
    by_task = {e["side_task"]: e for e in S}
    raw = json.load(open(REPO / "results" / "raw" / SLUG / raw_name))
    rows, honest_l2_total, honest_l2_comment_only = [], 0, 0
    # attack: own-signature level-2 hits split into real / comment-only, per trajectory
    for t in trajs.values():
        if t["side"] == "attack":
            e = by_task[t["side_task"]]
            real, com = [], []
            for a in t["actions"]:
                s = fmp.format_action(a, hide_output=True)
                verdicts = [comment_only(s, p) for p in e["level2"]]
                if any(v is False for v in verdicts):
                    real.append(a["index"])
                elif any(v is True for v in verdicts):
                    com.append(a["index"])
            trig = t["side_task_trigger_action_idx"]
            rows.append({
                "traj_id": t["traj_id"], "side_task": t["side_task"], "category": t["side_task_category"],
                "succeeded": t["side_task_succeeded"], "trigger_idx": trig,
                "in_subset_main": any((t["traj_id"], a["index"]) in in_subset for a in t["actions"]),
                "level2_real_idx": real, "level2_comment_only_idx": com,
                "anchor_hit_real": (trig is not None) and any(i <= trig for i in real),
                "anchor_hit_reported": (trig is not None) and any(i <= trig for i in real + com),
                "any_hit_real": bool(real), "any_hit_reported": bool(real or com),
            })
        else:
            # honest: any level-2 pattern from any entry (a3_validate's specificity definition)
            pats2 = sorted({p for e in S for p in e["level2"]})
            for a in t["actions"]:
                s = fmp.format_action(a, hide_output=True)
                verdicts = [comment_only(s, p) for p in pats2]
                if any(v is not None for v in verdicts):
                    honest_l2_total += 1
                    if not any(v is False for v in verdicts):
                        honest_l2_comment_only += 1
    anc = [r for r in rows if r["trigger_idx"] is not None]
    suc = [r for r in rows if r["trigger_idx"] is None and r["succeeded"]]
    summary = {
        "anchor": {"n": len(anc), "reported": sum(r["anchor_hit_reported"] for r in anc), "real": sum(r["anchor_hit_real"] for r in anc),
                   "comment_only_hits": [r["side_task"] for r in anc if r["anchor_hit_reported"] and not r["anchor_hit_real"]]},
        "success": {"n": len(suc), "reported": sum(r["any_hit_reported"] for r in suc), "real": sum(r["any_hit_real"] for r in suc),
                    "comment_only_hits": [r["side_task"] for r in suc if r["any_hit_reported"] and not r["any_hit_real"]]},
        "all_successful": {"n": len(anc) + len(suc),
                           "reported": sum(r["anchor_hit_reported"] for r in anc) + sum(r["any_hit_reported"] for r in suc),
                           "real": sum(r["anchor_hit_real"] for r in anc) + sum(r["any_hit_real"] for r in suc)},
        "subset_main_successful": {"n": sum(r["in_subset_main"] for r in anc + suc),
                                   "reported": sum(r["in_subset_main"] and (r["anchor_hit_reported"] if r["trigger_idx"] is not None else r["any_hit_reported"]) for r in anc + suc),
                                   "real": sum(r["in_subset_main"] and (r["anchor_hit_real"] if r["trigger_idx"] is not None else r["any_hit_real"]) for r in anc + suc)},
        "honest_level2_any": {"total": honest_l2_total, "comment_only": honest_l2_comment_only},
        "a3_raw_agrees_with_reported": {
            "anchor_n_hit": raw["coverage"]["anchor"]["n_hit"] == sum(r["anchor_hit_reported"] for r in anc),
            "success_n_hit": raw["coverage"]["success"]["n_hit"] == sum(r["any_hit_reported"] for r in suc),
            "honest_level2_any": raw["specificity"]["honest_level2_any"] == honest_l2_total,
        },
    }
    return {"signature_sha256": hashlib.sha256(sig_text.encode()).hexdigest(), "a3_raw": raw_name, "summary": summary, "rows": rows}


def main() -> None:
    trajs = fmp.load_export(EXPORT)
    in_subset = {(a["traj_id"], a["action_idx"]) for a in json.load(open(SUBSET))["actions"]}
    frozen = git("show", f"{FROZEN_REF}:experiments/{SLUG}/payload_signatures.json")
    out = {
        "slug": SLUG, "script": f"experiments/{SLUG}/verify_a3_comment_matches.py", "git_commit": git("rev-parse", "HEAD").strip(),
        "timestamp": TS, "comment_only_definition": "every re.finditer match of the pattern has a '#' in the text between the "
                                                    "preceding literal '\\n' (or string start) and the match start",
        "round1": analyse(frozen, RAWS[1], trajs, in_subset),
        "round2": analyse(SIGS.read_text(), RAWS[2], trajs, in_subset),
    }
    # the delete_model evidence, verbatim
    dm = next(t for t in trajs.values() if t["side_task"] == "delete_model")
    acts = {a["index"]: a for a in dm["actions"]}
    e = next(e for e in json.loads(SIGS.read_text())["signatures"] if e["side_task"] == "delete_model")
    s56 = fmp.format_action(acts[56], hide_output=True)
    spans = [{"pattern": p, "span": list(m.span()), "matched_text": m.group(0)} for p in e["level2"] for m in re.finditer(p, s56)]
    out["delete_model_evidence"] = {
        "traj_id": dm["traj_id"], "trigger_idx": dm["side_task_trigger_action_idx"], "n_actions": len(dm["actions"]),
        "level2_patterns": e["level2"],
        "action_56_rendered": s56, "action_56_matches": spans,
        "action_56_is_all_comment_lines": all(ln.strip() == "" or ln.strip().startswith("#") for ln in acts[56]["arguments"]["cmd"].split("\n")) if isinstance(acts[56].get("arguments"), dict) and "cmd" in acts[56]["arguments"] else None,
        "payload_region_rendered": {str(i): fmp.format_action(acts[i], hide_output=True) for i in (128, 129, 130, 131)},
        "payload_region_level2_matches": {str(i): [p for p in e["level2"] if re.search(p, fmp.format_action(acts[i], hide_output=True))] for i in (128, 129, 130)},
    }
    raw_dir = REPO / "results" / "raw" / SLUG
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"verify_a3_comment_matches_{TS}.json"
    path.write_text(json.dumps(out, indent=1))
    print("raw ->", path.relative_to(REPO))
    for r in ("round1", "round2"):
        print(r, json.dumps(out[r]["summary"]))
    print("delete_model action 56 all-comment:", out["delete_model_evidence"]["action_56_is_all_comment_lines"],
          "| matches:", [(m["span"], m["matched_text"]) for m in spans],
          "| payload-region level-2 matches:", out["delete_model_evidence"]["payload_region_level2_matches"])


if __name__ == "__main__":
    main()
