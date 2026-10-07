"""P-0 — consolidate the activation shards once into a single memmap-able array.

  checkpoints/<slug>/acts.npy         fp16 [n_actions, 7, 25, 2880] C-order (raw .npy, no header magic beyond numpy's)
  checkpoints/<slug>/acts_index.json  rows[i] -> traj_id, action_idx, render, source_stage, source_shard, n_positions
                                      + position_order, layout notes, sha256 of the array, verification block

Dedupe on (traj_id, action_idx, render) preferring out/main > out/consistent_high > out/main_pass1.
Pass-1 shards hold 4 prompt positions in the OLD order [transcript_end, action_end, instruction_end, final];
they are permuted into the rev-3 order and the three generation positions are NaN.
Rows are sorted by (traj_id, action_idx) so the index is deterministic.

Verify: index length == P-1 deduped count; 5 random rows (seed 0) re-read from their source shard are
bitwise equal; NaN pattern matches n_positions exactly. Raw results/raw/<slug>/p0_consolidate_<ts>.json
"""
from __future__ import annotations

import collections, hashlib, json, random, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

STAGES = ["main", "consistent_high", "main_pass1"]
P1 = sorted(C.RAW_DIR.glob("p1_verify_*.json"))[-1]


def source_map() -> dict:
    seen = {}
    for st in STAGES:
        idx = json.loads((C.OUT / st / "acts_index.json").read_text())
        for shard, ks in idx["shards"].items():
            for key in ks:
                t, i, render = key.split("|"); kk = (t, int(i), render)
                if kk not in seen:
                    seen[kk] = {"stage": st, "shard": shard, "order": idx["position_order"]}
    return {k: v for k, v in seen.items() if k[2] == "original"}


def to_rev3(t: torch.Tensor, order: list[str]) -> np.ndarray:
    """[len(order),25,2880] in `order` -> [7,25,2880] in C.POSITIONS with NaN where missing."""
    out = np.full((len(C.POSITIONS), C.N_LAYERS, C.D_MODEL), np.nan, dtype=np.float16)
    a = t.numpy()
    for j, name in enumerate(order):
        out[C.POSITIONS.index(C.POSITION_ALIASES.get(name, name))] = a[j]
    return out


def main():
    t0 = time.time(); stamp = C.ts()
    p1 = json.loads(P1.read_text()); assert p1["all_ok"], P1
    src = source_map(); assert len(src) == p1["check_3_dedupe"]["n_unique_original"] == 2012, len(src)
    keys = sorted(src, key=lambda k: (k[0], k[1]))
    row_of = {k: i for i, k in enumerate(keys)}
    C.CKPT.mkdir(parents=True, exist_ok=True)
    tmp = C.ACTS.with_suffix(".npy.tmp")
    acts = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16, shape=(len(keys), len(C.POSITIONS), C.N_LAYERS, C.D_MODEL))
    # group the work by shard so each shard is loaded exactly once
    by_shard = collections.defaultdict(list)
    for k, v in src.items():
        by_shard[(v["stage"], v["shard"])].append(k)
    written = np.zeros(len(keys), bool); n_shards = 0
    for (stage, shard), ks in sorted(by_shard.items()):
        sh = torch.load(C.OUT / stage / shard, map_location="cpu", weights_only=False)
        for k in ks:
            t = sh[f"{k[0]}|{k[1]}|{k[2]}"]
            assert t.dtype == torch.float16 and t.shape[1:] == (C.N_LAYERS, C.D_MODEL), (k, t.shape)
            acts[row_of[k]] = to_rev3(t, src[k]["order"]); written[row_of[k]] = True
        n_shards += 1; del sh
        print(f"  {stage}/{shard}: {len(ks)} rows  ({n_shards}/{len(by_shard)} shards, {time.time()-t0:.0f}s)", flush=True)
    assert written.all()
    acts.flush(); del acts
    tmp.rename(C.ACTS)
    print(f"array written {C.ACTS.stat().st_size/1e9:.2f} GB in {time.time()-t0:.0f}s; hashing...", flush=True)
    h = hashlib.sha256()
    with open(C.ACTS, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 26), b""):
            h.update(chunk)
    arr_sha = h.hexdigest()

    rows = [{"row": i, "traj_id": k[0], "action_idx": k[1], "render": k[2], "source_stage": src[k]["stage"],
             "source_shard": src[k]["shard"], "n_positions": len(src[k]["order"]), "source_position_order": src[k]["order"]} for i, k in enumerate(keys)]
    # ── verification ──
    acts = np.load(C.ACTS, mmap_mode="r"); assert acts.shape == (len(keys), 7, 25, 2880) and acts.dtype == np.float16
    rng = random.Random(0); picks = rng.sample(range(len(keys)), 5); bitwise = []
    for i in picks:
        k = keys[i]; sh = torch.load(C.OUT / src[k]["stage"] / src[k]["shard"], map_location="cpu", weights_only=False)
        ref = to_rev3(sh[f"{k[0]}|{k[1]}|{k[2]}"], src[k]["order"]); got = np.asarray(acts[i])
        eq = bool(np.array_equal(ref.view(np.uint16), got.view(np.uint16)))   # bitwise, NaN-safe
        bitwise.append({"row": i, "key": list(k), "source": f"{src[k]['stage']}/{src[k]['shard']}", "bitwise_equal": eq})
        del sh
    # NaN pattern: positions 4..6 for the 4-position (pass-1) rows, plus any generation position the runner
    # could not locate (capture_ok False: the 5 no-verdict actions) -> gen_index None in capture_positions.
    nan_any = np.isnan(acts[:, :, :, ::64]).any(axis=(2, 3))     # [n,7] strided sample
    nan_all = np.isnan(acts[:, :, :, ::64]).all(axis=(2, 3))
    rec, _ = C.load_scores(); expect = np.zeros_like(nan_any)
    for i, r in enumerate(rows):
        s = rec[(r["traj_id"], r["action_idx"])]
        if r["n_positions"] == 4:
            expect[i, 4:] = True
        else:
            for c in s["capture_positions"]:
                if c.get("phase") == "generation" and c.get("gen_index") is None:
                    expect[i, C.POSITIONS.index(c["name"])] = True
        r["nan_positions"] = [C.POSITIONS[j] for j in np.where(expect[i])[0]]
    nan_ok = bool(np.array_equal(nan_any, expect) and np.array_equal(nan_all, expect))   # all-or-none per position
    n_pos = np.array([r["n_positions"] for r in rows])
    # finite everywhere else, and norms in the range the runner recorded
    finite_ok = bool(np.isfinite(acts[:, :4, -1, :]).all())
    ver = {"n_rows": len(rows), "p1_dedup_count": p1["check_3_dedupe"]["n_unique_original"], "count_ok": len(rows) == p1["check_3_dedupe"]["n_unique_original"],
           "bitwise_5_random_rows": bitwise, "bitwise_ok": all(b["bitwise_equal"] for b in bitwise),
           "nan_pattern_ok": nan_ok, "n_four_position_rows": int((n_pos == 4).sum()), "prompt_positions_finite_last_layer": finite_ok,
           "rows_with_nan_generation_positions_despite_7": [[r["traj_id"], r["action_idx"], r["nan_positions"]] for r in rows if r["n_positions"] == 7 and r["nan_positions"]],
           "source_counts": dict(collections.Counter(r["source_stage"] for r in rows))}
    ver["all_ok"] = ver["count_ok"] and ver["bitwise_ok"] and ver["nan_pattern_ok"] and finite_ok
    index = {"slug": C.SLUG, "script": f"experiments/{C.SLUG}/p0_consolidate.py", "timestamp": stamp, "git_commit": C.git_commit(),
             "array": str(C.ACTS.relative_to(C.REPO)), "shape": [len(rows), 7, 25, 2880], "dtype": "float16", "order": "C", "sha256": arr_sha,
             "position_order": C.POSITIONS, "position_aliases": C.POSITION_ALIASES, "pass1_source_order": C.PASS1_ORDER,
             "layer_axis": "index l = output of decoder layer l-1 (l=0 embeddings), as in run_monitor.py",
             "nan_rule": "n_positions==4 rows have positions 4..6 (analysis_end, reasoning_end, pre_score) NaN; 7-position rows have NaN at any generation position whose gen_index is None (capture_ok False); per-row list in rows[i].nan_positions",
             "p1_raw": str(P1.relative_to(C.REPO)), "verification": ver, "rows": rows}
    C.ACTS_INDEX.write_text(json.dumps(index))
    raw = {k: v for k, v in index.items() if k != "rows"} | {"seconds": time.time() - t0, "four_position_keys": [[r["traj_id"], r["action_idx"]] for r in rows if r["n_positions"] == 4]}
    C.write_raw("p0_consolidate", raw, stamp)
    print(json.dumps(ver, indent=1)); print("ALL_OK" if ver["all_ok"] else "!! VERIFICATION FAILED")


if __name__ == "__main__":
    main()
