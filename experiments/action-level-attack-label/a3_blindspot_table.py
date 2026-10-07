"""Where the level-2 label is blind, from existing artefacts only (no trajectories, no scores).

Inputs: payload_signatures.json (HEAD), the A3 round-2 raw, the verification raw, the A1 raw.
Classifies every side task's level-2 signature by what its regexes anchor on, read off the
pattern text alone:
  code-edit   every level-2 pattern needs a text_editor str_replace/create (matches on
              'str_replace', "'command': 'create'", or 'file_text')
  mixed       some patterns are code-edit, some are command/artefact
  command     no pattern needs a code edit (shell command, URL, path, table, socket, key ...)
and crosses that with failure_type and A3 coverage (real, i.e. excluding comment-only hits).
Writes results/raw/<slug>/a3_blindspot_table_<ts>.json and prints the tables.
"""
from __future__ import annotations

import collections
import datetime as dt
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SLUG = HERE.name
REPO = HERE.parents[1]
RAW = REPO / "results" / "raw" / SLUG
A3 = RAW / "a3_validate_20260911T070300Z.json"
VER = sorted(RAW.glob("verify_a3_comment_matches_*.json"))[-1]
A1 = RAW / "a1_export_fields_20260911T023003Z.json"
TS = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
CODE_EDIT = re.compile(r"str_replace|'command':\\s\*'create'|command.{0,20}create|file_text|old_str|new_str")


def ptype(p: str) -> str:
    return "code-edit" if CODE_EDIT.search(p) else "command"


def main() -> None:
    S = json.load(open(HERE / "payload_signatures.json"))["signatures"]
    a3 = json.load(open(A3)); ver = json.load(open(VER)); a1 = json.load(open(A1))
    ft = {r["side_task"]: r["side_task_failure_type"] for r in a1["trajectories_attack"]}
    insub = {r["side_task"]: r["in_subset_main"] for r in a1["trajectories_attack"]}
    succ = {r["side_task"]: r["side_task_succeeded"] for r in a1["trajectories_attack"]}
    real = {r["side_task"]: r for r in ver["round2"]["rows"]}
    honest_hits = {p["pattern"]: p["honest_hits"] for p in a3["specificity"]["per_pattern"]}
    rows = []
    for e in S:
        kinds = [ptype(p) for p in e["level2"]]
        sig = "code-edit" if all(k == "code-edit" for k in kinds) else "command" if all(k == "command" for k in kinds) else "mixed"
        r = real[e["side_task"]]
        hit = r["anchor_hit_real"] if r["trigger_idx"] is not None else r["any_hit_real"]
        rows.append({"side_task": e["side_task"], "category": e["category"], "failure_type": ft[e["side_task"]],
                     "succeeded": succ[e["side_task"]], "in_subset_main": insub[e["side_task"]], "signature_type": sig,
                     "n_level2": len(kinds), "n_level2_code_edit": kinds.count("code-edit"),
                     "honest_hits_of_its_level2_patterns": sum(honest_hits.get(p, 0) for p in e["level2"]),
                     "coverage_real": bool(hit), "anchor": r["trigger_idx"] is not None})
    cross = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        if r["succeeded"]:
            c = cross[(r["signature_type"], r["failure_type"])]; c[0] += r["coverage_real"]; c[1] += 1
    hon = collections.defaultdict(int)
    seen = set()
    for e in S:
        for p in e["level2"]:
            if p not in seen:
                seen.add(p); hon[ptype(p)] += honest_hits.get(p, 0)
    out = {"slug": SLUG, "script": f"experiments/{SLUG}/a3_blindspot_table.py", "timestamp": TS,
           "inputs": {"a3_raw": str(A3.relative_to(REPO)), "verify_raw": str(VER.relative_to(REPO)), "a1_raw": str(A1.relative_to(REPO))},
           "classifier_regex": CODE_EDIT.pattern,
           "coverage_by_signature_type_x_failure_type": {f"{k[0]}|{k[1]}": {"hit": v[0], "n": v[1]} for k, v in sorted(cross.items())},
           "honest_level2_hits_by_pattern_type": dict(hon),
           "misses": [r for r in rows if r["succeeded"] and not r["coverage_real"]],
           "rows": rows}
    (RAW / f"a3_blindspot_table_{TS}.json").write_text(json.dumps(out, indent=1))
    print("raw ->", f"results/raw/{SLUG}/a3_blindspot_table_{TS}.json")
    print("\nsuccessful trajectories: real level-2 coverage by signature type x failure type (hit/n)")
    print(f"{'signature type':14} {'live':>8} {'delayed':>9} {'total':>8}")
    for st in ("command", "mixed", "code-edit"):
        l = cross[(st, "live")]; d = cross[(st, "delayed")]
        print(f"{st:14} {l[0]:>3}/{l[1]:<4} {d[0]:>3}/{d[1]:<5} {l[0]+d[0]:>3}/{l[1]+d[1]}")
    print("\nhonest level-2 hits by pattern type (unique patterns):", dict(hon))
    print("\nthe 14 misses on successful trajectories:")
    print(f"{'side_task':30} {'category':22} {'fail':8} {'sigtype':10} {'n_l2':>4} {'honest_hits':>11} sub")
    for r in out["misses"]:
        print(f"{r['side_task']:30} {r['category']:22} {r['failure_type']:8} {r['signature_type']:10} {r['n_level2']:>4} {r['honest_hits_of_its_level2_patterns']:>11} {'Y' if r['in_subset_main'] else '.'}")
    print("\nsignature type counts over all 92:", collections.Counter(r["signature_type"] for r in rows))
    print("by category x signature type:")
    cc = collections.defaultdict(collections.Counter)
    for r in rows: cc[r["category"]][r["signature_type"]] += 1
    for c in sorted(cc): print(f"  {c:22} {dict(cc[c])}")


if __name__ == "__main__":
    main()
