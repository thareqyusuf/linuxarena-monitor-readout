#!/usr/bin/env python3
"""G6' integrity of one or more stage dirs under checkpoints/<slug>/out (CPU): file count; every shard loads; shard keys ==
scores.jsonl ok-keys; line count == stage size - (not yet done) - documented skips/failures; sha256 manifest; one tensor
opened with its position order printed. With --copy-raw, copies each stage's scores.jsonl into results/raw/<slug>/ in
parts of <= 9 MB (g4_<stage>_scores_<ts>_partN.jsonl) so the per-action records are committed.
Writes results/raw/<slug>/g6_integrity_<ts>.json (includes the manifest).
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve()
SLUG = HERE.parent.name
REPO = HERE.parents[2]
RAW = REPO / "results" / "raw" / SLUG
OUT = REPO / "checkpoints" / SLUG / "out"
sys.path.insert(0, str(HERE.parent))
import run_monitor as rm  # noqa: E402


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_stage(stage: str, run_order: dict, ts: str, copy_raw: bool) -> dict:
    d = OUT / stage
    st = next((s for s in run_order["stages"] if s["stage"] == stage), None)
    r: dict = {"stage": stage, "dir": str(d.relative_to(REPO)), "exists": d.exists()}
    if not d.exists():
        return r
    files = sorted(p for p in d.iterdir() if p.is_file())
    r["n_files"] = len(files)
    r["manifest"] = {p.name: {"bytes": p.stat().st_size, "sha256": sha256_file(p)} for p in files}
    lines = [l for l in (d / "scores.jsonl").read_text().splitlines() if l.strip()] if (d / "scores.jsonl").exists() else []
    recs = [json.loads(l) for l in lines]
    keys = [rm.key_of_rec(x) for x in recs]
    by_status = {}
    for x in recs:
        by_status[x.get("status")] = by_status.get(x.get("status"), 0) + 1
    ok_keys = {rm.key_of_rec(x) for x in recs if x.get("status") == "ok"}
    idx = json.loads((d / "acts_index.json").read_text()) if (d / "acts_index.json").exists() else {"shards": {}, "position_order": None}
    shard_keys, shard_load = [], {}
    for name, ks in idx["shards"].items():
        try:
            sh = torch.load(d / name)
            tk = [k for k in sh if k != "_meta"]
            shard_load[name] = {"loads": True, "n": len(tk), "index_n": len(ks), "keys_match_index": set(tk) == set(ks),
                                "meta_position_order": sh.get("_meta", {}).get("position_order"),
                                "shape": list(sh[tk[0]].shape) if tk else None, "dtype": str(sh[tk[0]].dtype) if tk else None}
            shard_keys += tk
        except Exception as e:  # noqa: BLE001
            shard_load[name] = {"loads": False, "error": str(e)[:200]}
    stage_n = st["n"] if st else None
    stage_keys = {f"{e['key']}|{st['render']}" for e in st["actions"]} if st else None
    r.update({"scores_lines": len(lines), "unique_keys": len(set(keys)), "duplicate_keys": len(keys) - len(set(keys)),
              "by_status": by_status, "n_shards": len(idx["shards"]), "shards": shard_load,
              "all_shards_load": all(v.get("loads") for v in shard_load.values()),
              "shard_keys_eq_ok_keys": set(shard_keys) == ok_keys, "n_shard_keys": len(shard_keys), "n_ok_keys": len(ok_keys),
              "index_position_order": idx.get("position_order"), "stage_size": stage_n,
              "n_not_yet_done": (len(stage_keys - set(keys)) if stage_keys is not None else None),
              "keys_outside_stage": (len(set(keys) - stage_keys) if stage_keys is not None else None),
              "line_count_identity": (len(lines) == stage_n - len(stage_keys - set(keys))) if stage_keys is not None else None,
              "bak_files": sorted(p.name for p in d.glob("scores.jsonl.bak_*"))})
    if shard_keys:
        name0 = next(iter(idx["shards"]))
        sh = torch.load(d / name0); k0 = next(k for k in sh if k != "_meta")
        t = sh[k0]
        r["one_tensor"] = {"shard": name0, "key": k0, "shape": list(t.shape), "dtype": str(t.dtype),
                           "position_order": sh.get("_meta", {}).get("position_order"),
                           "row_finite": [bool(torch.isfinite(t[i]).all()) for i in range(t.shape[0])],
                           "row_norm_last_layer": [float(t[i, -1].float().norm()) for i in range(t.shape[0])]}
        print(f"[{stage}] one tensor {k0} from {name0}: shape {list(t.shape)} {t.dtype} position order {r['one_tensor']['position_order']} "
              f"finite rows {r['one_tensor']['row_finite']}")
    if copy_raw and lines:
        parts, cur, size = [], [], 0
        for l in lines:
            if size + len(l) + 1 > 9 * 2**20 and cur:
                parts.append(cur); cur, size = [], 0
            cur.append(l); size += len(l) + 1
        if cur:
            parts.append(cur)
        RAW.mkdir(parents=True, exist_ok=True)
        r["raw_copies"] = []
        for i, part in enumerate(parts, 1):
            p = RAW / f"g4_{stage}_scores_{ts}_part{i}.jsonl"
            p.write_text("\n".join(part) + "\n")
            r["raw_copies"].append({"path": str(p.relative_to(REPO)), "lines": len(part), "bytes": p.stat().st_size, "sha256": sha256_file(p)})
    r["pass"] = bool(r["all_shards_load"] and r["shard_keys_eq_ok_keys"] and r["duplicate_keys"] == 0
                     and (r["line_count_identity"] in (True, None)) and all(v.get("keys_match_index") for v in shard_load.values()))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="+", default=["main"])
    ap.add_argument("--run-order", type=Path, default=HERE.parent / "run_order.json")
    ap.add_argument("--copy-raw", action="store_true")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    ro = json.loads(args.run_order.read_text())
    R = {"slug": SLUG, "script": str(HERE.relative_to(REPO)), "timestamp": ts, "git_commit": rm.git_commit(),
         "run_order_sha256": sha256_file(args.run_order), "stages": {s: check_stage(s, ro, ts, args.copy_raw) for s in args.stages}}
    R["all_pass"] = all(v.get("pass") for v in R["stages"].values() if v.get("exists"))
    out = RAW / f"g6_integrity_{args.tag + '_' if args.tag else ''}{ts}.json"
    RAW.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(R, indent=1))
    for s, v in R["stages"].items():
        print(json.dumps({k: x for k, x in v.items() if k not in ("manifest", "shards", "one_tensor")}, indent=1))
    print("ALL PASS:", R["all_pass"], "raw ->", out.relative_to(REPO))


if __name__ == "__main__":
    main()
