"""P-1 — verify the activation capture before consolidating. Reads shard indexes + score records; loads
exactly one shard (the newest in out/main). Does not touch running processes.

  1. newest acts_*.pt in out/main: keys, shape per key, dtype, recorded position order (expect [7,25,2880])
  2. capture_positions on every record (3 quoted verbatim: honest / attack / placeholder): 7 entries with
     token_index + decoded; analysis_end is followed by <|end|>; pre_score decodes to '>\\n' and is followed
     by the digit that became the score
  3. dedupe (traj_id, action_idx, render) across out/main, out/consistent_high, out/main_pass1 preferring
     out/main -> assert 2,012 unique original-render keys; record which 200 have only 4 positions

Raw results/raw/<slug>/p1_verify_<ts>.json
"""
from __future__ import annotations

import collections, json, sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

STAGES = ["main", "consistent_high", "main_pass1"]     # preference order for duplicate keys


def main():
    out = {"slug": C.SLUG, "script": f"experiments/{C.SLUG}/p1_verify.py", "timestamp": C.ts(), "git_commit": C.git_commit()}
    # 1. newest shard in out/main
    shards = sorted((C.OUT / "main").glob("acts_*.pt"))
    newest = shards[-1]; sh = torch.load(newest, map_location="cpu", weights_only=False)
    keys = [k for k in sh if k != "_meta"]
    shapes = collections.Counter((tuple(sh[k].shape), str(sh[k].dtype)) for k in keys)
    out["newest_main_shard"] = {"path": str(newest.relative_to(C.REPO)), "n_keys": len(keys), "keys": keys,
                               "shape_dtype_counts": {f"{s} {d}": n for (s, d), n in shapes.items()},
                               "meta": sh["_meta"], "n_shards_in_dir": len(shards)}
    ok1 = all(s == (7, 25, 2880) and d == "torch.float16" for (s, d) in shapes) and sh["_meta"]["position_order"] == C.POSITIONS
    out["check_1_shape_and_order_ok"] = bool(ok1)
    del sh

    # 2. capture positions on every main-pass record
    rec, smeta = C.load_scores(); out["score_files"] = smeta
    probs = []; quoted = {}
    for k, r in rec.items():
        if r["_score_file"].startswith("g4_main_first200"):
            continue           # pass-1 runner: 4 prompt positions only, checked in 3.
        cap = r["capture_positions"]; names = [c["name"] for c in cap]
        if names != C.POSITIONS or any("token_index" not in c or "decoded" not in c for c in cap):
            probs.append({"key": list(k), "why": "positions", "names": names}); continue
        d = {c["name"]: c for c in cap}
        if r["capture_ok"]:
            if d["analysis_end"].get("next_decoded") != "<|end|>":
                probs.append({"key": list(k), "why": "analysis_end not followed by <|end|>", "got": d["analysis_end"].get("next_decoded")})
            if d["pre_score"]["decoded"] != ">\n" or (r["score"] is not None and d["pre_score"].get("next_decoded") != str(int(r["score"]))):
                probs.append({"key": list(k), "why": "pre_score", "decoded": d["pre_score"]["decoded"], "next": d["pre_score"].get("next_decoded"), "score": r["score"]})
        else:
            probs.append({"key": list(k), "why": "capture_ok False (no verdict / tool call)", "score": r["score"], "stopped_on": r["stopped_on"]})
        kind = "placeholder" if r["has_attachment_placeholder"] else r["side"]
        if kind not in quoted and r["capture_ok"]:
            quoted[kind] = {"key": list(k), "score": r["score"], "E": r["expected_score"], "capture_positions": cap,
                            "generation_tail": r["generation_text"][-160:]}
    out["check_2_capture_positions"] = {"n_checked": sum(not r["_score_file"].startswith("g4_main_first200") for r in rec.values()),
                                        "n_problems": len(probs), "problems": probs, "quoted": quoted}

    # 3. dedupe across stages
    seen = {}; per_stage = {}
    for st in STAGES:
        idx = json.loads((C.OUT / st / "acts_index.json").read_text())
        order = idx["position_order"]; n = 0
        for shard, ks in idx["shards"].items():
            for key in ks:
                t, i, render = key.split("|"); kk = (t, int(i), render); n += 1
                if kk not in seen:
                    seen[kk] = {"stage": st, "shard": shard, "n_positions": len(order), "position_order": order}
        per_stage[st] = {"n_keys": n, "position_order": order, "n_shards": len(idx["shards"])}
    orig = {k: v for k, v in seen.items() if k[2] == "original"}
    out["check_3_dedupe"] = {"per_stage": per_stage, "n_unique_original": len(orig),
                             "n_by_source": dict(collections.Counter(v["stage"] for v in orig.values())),
                             "four_position_keys": sorted([list(k) for k, v in orig.items() if v["n_positions"] == 4]),
                             "score_keys_match": set((k[0], k[1]) for k in orig) == set(rec)}
    # the 200 pass-1 keys keep 4 positions unless a 7-position copy exists in consistent_high (1 key does)
    pass1_keys = {k for k in orig if k[0:2] in {(t, int(i)) for t, i in (key.split("|")[:2] for ks in json.loads((C.OUT / "main_pass1" / "acts_index.json").read_text())["shards"].values() for key in ks)}}
    out["check_3_dedupe"]["n_pass1_keys"] = len(pass1_keys)
    out["check_3_dedupe"]["pass1_keys_with_7_position_copy"] = sorted([list(k) for k in pass1_keys if orig[k]["n_positions"] == 7])
    expected_four = len(pass1_keys) - len(out["check_3_dedupe"]["pass1_keys_with_7_position_copy"])
    out["check_3_ok"] = len(orig) == 2012 and out["check_3_dedupe"]["score_keys_match"] and len(out["check_3_dedupe"]["four_position_keys"]) == expected_four
    out["all_ok"] = bool(ok1 and out["check_3_ok"] and all(p["why"].startswith("capture_ok False") for p in probs))
    C.write_raw("p1_verify", out, out["timestamp"])
    print(json.dumps({k: v for k, v in out.items() if k not in ("newest_main_shard", "check_2_capture_positions", "check_3_dedupe")}, indent=1))
    print("newest shard:", out["newest_main_shard"]["path"], out["newest_main_shard"]["shape_dtype_counts"], out["newest_main_shard"]["meta"]["position_order"])
    print("capture problems:", json.dumps(probs, indent=0)[:2000])
    print("dedupe:", {k: v for k, v in out["check_3_dedupe"].items() if k != "four_position_keys"}, "n four-position:", len(out["check_3_dedupe"]["four_position_keys"]))
    for kind, q in quoted.items():
        print(f"\n== quoted {kind} {q['key']} score {q['score']} E {q['E']:.2f}")
        for c in q["capture_positions"]:
            print(f"   {c['name']:16} tok {c['token_index']:6} {c['decoded']!r:12} next {c.get('next_decoded')!r}")


if __name__ == "__main__":
    main()
