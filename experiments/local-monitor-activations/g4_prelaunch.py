#!/usr/bin/env python3
"""G4.0 pre-launch checks (TASK-ver3). Six checks; the raw file records every number; any failure => do not launch.

  1. prompt-side rows + expected_score bitwise equal to the old shard (out/consistent_high_pass1) for the same 3 keys
  2. generation determinism: generated ids with capture hooks == without hooks, token for token
  3. position decoding: decoded token +-5 around each generation-side position; analysis_end precedes <|end|>,
     reasoning_end precedes </reasoning>, pre_score is followed by the parsed digit; digit_token_mass_post >= 0.99, argmax == score
  4. resume: run 4 actions of stage low_cached in a subprocess, SIGTERM after 2, relaunch: no duplicate keys, no recompute
     (plus the SIGKILL variant, which exercises the orphan reconciliation)
  5. peak VRAM on the longest action (<= 32k) of subset_main with hooks on: <= 24 GiB
  6. shape [7, 25, 2880] fp16, no NaN/Inf on the 3, position-order list present in a shard written through shard_payload()

Writes results/raw/<slug>/g4_prelaunch_<ts>.json. Model is loaded once, in-process, AFTER the subprocess check (VRAM).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve()
SLUG = HERE.parent.name
REPO = HERE.parents[2]
RAW = REPO / "results" / "raw" / SLUG
OUT = REPO / "checkpoints" / SLUG / "out"
sys.path.insert(0, str(HERE.parent))
import run_monitor as rm  # noqa: E402

TS = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
OLD_ORDER = ["transcript_end", "action_end", "instruction_end", "final"]


def load_old(old_dir: Path) -> tuple[dict, dict]:
    recs = {rm.key_of_rec(r): r for r in (json.loads(l) for l in (old_dir / "scores.jsonl").open())}
    idx = json.loads((old_dir / "acts_index.json").read_text())
    assert idx["position_order"] == OLD_ORDER, idx["position_order"]
    where = {k: s for s, keys in idx["shards"].items() for k in keys}
    return recs, where


def old_acts(old_dir: Path, where: dict, key: str) -> torch.Tensor:
    sh = torch.load(old_dir / where[key])
    return sh[key]


# ── check 4: resume (subprocess) ─────────────────────────────────────────────────────────────
def resume_check(args, variant: str) -> dict:
    scratch = OUT / f"_prelaunch_resume_{variant}_{TS}"
    cmd = [sys.executable, str(HERE.parent / "run_monitor.py"), "--attn", args.attn, "--attn-chunk-size", str(args.attn_chunk_size),
           "--max-prompt-tokens", str(args.max_prompt_tokens), "--reasoning-effort", args.reasoning_effort, "--template-date", args.template_date,
           "--max-new-tokens", str(args.max_new_tokens), "--checkpoint-every", "25", "--gate-at", "0",
           "--run-order", str(args.run_order), "--stage", "low_cached", "--limit", "4", "--out", str(scratch)]
    env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "HF_HOME": os.environ.get("HF_HOME", "/workspace/hf")}
    log1 = scratch.parent / f"{scratch.name}_run1.log"
    scratch.mkdir(parents=True, exist_ok=True)
    r: dict = {"variant": variant, "cmd": " ".join(cmd), "scratch": str(scratch.relative_to(REPO))}
    with log1.open("w") as f:
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=REPO)
        t0 = time.time()
        while True:
            time.sleep(1)
            txt = log1.read_text()
            if re.search(r"^\[2/4\] ok", txt, re.M) or p.poll() is not None or time.time() - t0 > 900:
                break
        assert p.poll() is None, "runner exited before the 2nd action: " + txt[-2000:]
        os.kill(p.pid, signal.SIGTERM if variant == "sigterm" else signal.SIGKILL)
        rc1 = p.wait(timeout=600)
    txt1 = log1.read_text()
    r["run1_rc"] = rc1
    r["run1_ok_lines"] = len(re.findall(r"^\[\d/4\] ok", txt1, re.M))
    sc1 = [json.loads(l) for l in (scratch / "scores.jsonl").read_text().splitlines() if l.strip()] if (scratch / "scores.jsonl").exists() else []
    idx1 = json.loads((scratch / "acts_index.json").read_text()) if (scratch / "acts_index.json").exists() else {"shards": {}}
    r["after_kill"] = {"scores_lines": len(sc1), "index_keys": sum(len(v) for v in idx1["shards"].values()),
                       "keys": [rm.key_of_rec(x) for x in sc1]}
    n_before = r["run1_ok_lines"]                     # the signal is honoured after the in-flight action, so this may be 3
    first_n = [json.dumps(x, sort_keys=True) for x in sc1[:n_before]]
    # relaunch
    log2 = scratch.parent / f"{scratch.name}_run2.log"
    with log2.open("w") as f:
        rc2 = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=REPO).returncode
    txt2 = log2.read_text()
    m = re.search(r"== stage low_cached: (\d+) actions to do \((\d+) already", txt2)
    sc2 = [json.loads(l) for l in (scratch / "scores.jsonl").read_text().splitlines() if l.strip()]
    idx2 = json.loads((scratch / "acts_index.json").read_text())
    keys2 = [rm.key_of_rec(x) for x in sc2]
    ikeys2 = [k for v in idx2["shards"].values() for k in v]
    baks = sorted(p.name for p in scratch.glob("scores.jsonl.bak_*"))
    r["relaunch"] = {"rc": rc2, "todo_at_start": int(m.group(1)) if m else None, "already_done_at_start": int(m.group(2)) if m else None,
                     "scores_lines": len(sc2), "unique_keys": len(set(keys2)), "index_keys": len(ikeys2), "index_unique": len(set(ikeys2)),
                     "scores_keys_eq_index_keys": set(keys2) == set(ikeys2), "bak_files": baks,
                     "reconcile_line": next((l for l in txt2.splitlines() if l.startswith("== stage")), None)}
    final_ok = len(sc2) == 4 and len(set(keys2)) == 4 and len(set(ikeys2)) == 4 and r["relaunch"]["scores_keys_eq_index_keys"] and rc2 == 0
    if variant == "sigterm":
        # every action completed before the exit was flushed; the relaunch skipped exactly those and left their records byte-identical
        r["relaunch"]["records_before_kill_unchanged"] = [json.dumps(x, sort_keys=True) for x in sc2[:n_before]] == first_n
        r["pass"] = (rc1 == 130 and n_before >= 2 and r["after_kill"]["index_keys"] == n_before and r["after_kill"]["scores_lines"] == n_before
                     and r["relaunch"]["already_done_at_start"] == n_before and r["relaunch"]["todo_at_start"] == 4 - n_before
                     and r["relaunch"]["records_before_kill_unchanged"] and not baks and final_ok)
    else:
        # nothing was flushed to the index; the ok lines on disk are orphans -> moved to one .bak, all 4 recomputed
        r["pass"] = (r["after_kill"]["index_keys"] == 0 and r["after_kill"]["scores_lines"] >= 2 and len(baks) == 1
                     and r["relaunch"]["already_done_at_start"] == 0 and r["relaunch"]["todo_at_start"] == 4 and final_ok)
    r["log1"], r["log2"] = str(log1.relative_to(REPO)), str(log2.relative_to(REPO))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attn", default="chunked_eager")
    ap.add_argument("--attn-chunk-size", type=int, default=256)
    ap.add_argument("--max-prompt-tokens", type=int, default=32000)
    ap.add_argument("--reasoning-effort", default="low")
    ap.add_argument("--template-date", default="2026-09-10")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--run-order", type=Path, default=HERE.parent / "run_order.json")
    ap.add_argument("--old-dir", type=Path, default=OUT / "consistent_high_pass1")
    ap.add_argument("--skip-resume", action="store_true")
    ap.add_argument("--render", default="original")
    ap.add_argument("--n-samples", type=int, default=0); ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--sample-seed", type=int, default=0); ap.add_argument("--sample-budget", type=int, default=1024)
    ap.add_argument("--subset", default=None); ap.add_argument("--order", default="ascending"); ap.add_argument("--stage", default="prelaunch")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    assert os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "").startswith("expandable_segments")
    R: dict = {"slug": SLUG, "script": str(HERE.relative_to(REPO)), "timestamp": TS, "git_commit": rm.git_commit(),
               "settings": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}, "checks": {}}

    # ── 4. resume (before the in-process model load) ─────────────────────
    if not args.skip_resume:
        print("== check 4: resume (SIGTERM variant)", flush=True)
        R["checks"]["4_resume_sigterm"] = resume_check(args, "sigterm")
        print(json.dumps(R["checks"]["4_resume_sigterm"], indent=1), flush=True)
        print("== check 4b: resume (SIGKILL variant, orphan reconciliation)", flush=True)
        R["checks"]["4b_resume_sigkill"] = resume_check(args, "sigkill")
        print(json.dumps(R["checks"]["4b_resume_sigkill"], indent=1), flush=True)

    # ── pick the 3 keys ───────────────────────────────────────────────────
    old_recs, where = load_old(args.old_dir)
    def pick(f):
        return next(r for r in old_recs.values() if f(r))
    honest = pick(lambda r: r["side"] == "honest" and not r["has_attachment_placeholder"])
    payload = pick(lambda r: r["side"] == "attack" and not r["has_attachment_placeholder"] and (r.get("score") or 0) >= 7)
    placeholder = pick(lambda r: r["has_attachment_placeholder"] and (r.get("score") or 0) >= 7)
    trio = {"honest": honest, "attack_payload": payload, "placeholder": placeholder}
    R["keys"] = {k: {"key": rm.key_of_rec(r), "old_score": r["score"], "old_expected_score": r["expected_score"], "n_prompt_tokens": r["n_prompt_tokens"],
                     "side_task": r["side_task"]} for k, r in trio.items()}
    print("== keys:", json.dumps(R["keys"], indent=1), flush=True)

    runner = rm.Runner(args)
    runner.load_model()
    R["model"] = {"load_seconds": runner.load_s, "resident_gib": torch.cuda.memory_allocated() / 2**30,
                  "mxfp4_native": not runner.model.hf_quantizer.quantization_config.dequantize}
    c1, c2, c3, c6 = {}, {}, {}, {}
    new_acts = {}
    for label, old in trio.items():
        a = {"traj_id": old["traj_id"], "action_idx": old["action_idx"], "n_tokens": old["n_prompt_tokens"]}
        key = rm.key_of_rec(old)
        rec, acts = runner.run_action(a, run_stage="prelaunch", run_order_index=None, hooks=True)
        assert rec["status"] == "ok", rec.get("reason")
        new_acts[key] = acts
        # 1. bitwise vs the old shard
        oa = old_acts(args.old_dir, where, key)                       # [4, 25, 2880] in OLD_ORDER
        reord = torch.stack([oa[OLD_ORDER.index(rm.PROMPT_POS_SOURCE[n])] for n in rm.PROMPT_POSITIONS])
        d = (acts[:4].float() - reord.float()).abs()
        c1[label] = {"key": key, "max_abs_diff_prompt_rows": float(d.max()), "n_nonzero": int((d > 0).sum()),
                     "rows_bitwise_equal": bool(torch.equal(acts[:4], reord)),
                     "expected_score_old": old["expected_score"], "expected_score_new": rec["expected_score"],
                     "expected_score_equal": old["expected_score"] == rec["expected_score"],
                     "digit_dist_equal": old["digit_dist"] == rec["digit_dist"],
                     "score_old": old["score"], "score_new": rec["score"], "score_equal": old["score"] == rec["score"],
                     "generation_text_equal": old["generation"] == rec["generation"],
                     "prompt_positions_old": {p["name"]: p["token_index"] for p in old["capture_positions"]},
                     "prompt_positions_new": {p["name"]: p["token_index"] for p in rec["capture_positions"][:4]},
                     "pass": bool(torch.equal(acts[:4], reord)) and old["expected_score"] == rec["expected_score"] and old["score"] == rec["score"]
                             and old["generation"] == rec["generation"]}
        print(f"== check 1 [{label}]: {json.dumps(c1[label])}", flush=True)
        # 2. determinism with hooks removed
        rec_nh, _ = runner.run_action(a, run_stage="prelaunch", run_order_index=None, hooks=False)
        assert rec_nh["status"] == "ok" and not rec_nh["hooks_attached"]
        runner.capture.attach()
        c2[label] = {"key": key, "ids_with_hooks": len(rec["generated_ids"]), "ids_without_hooks": len(rec_nh["generated_ids"]),
                     "ids_equal": rec["generated_ids"] == rec_nh["generated_ids"], "text_equal": rec["generation"] == rec_nh["generation"],
                     "expected_score_equal": rec["expected_score"] == rec_nh["expected_score"],
                     "pass": rec["generated_ids"] == rec_nh["generated_ids"] and rec["generation"] == rec_nh["generation"]}
        print(f"== check 2 [{label}]: {json.dumps(c2[label])}", flush=True)
        # 3. position decoding
        gp = {p["name"]: p for p in rec["capture_positions"][4:]}
        wa = gp["reasoning_end"]["window_after"] or []
        c3[label] = {"key": key, "score": rec["score"], "capture_ok": rec["capture_ok"], "capture_clean": rec["capture_clean"],
                     "positions": gp,
                     "analysis_end_next_is_end": gp["analysis_end"]["next_decoded"] == "<|end|>",
                     "reasoning_end_next_starts_close": "".join(wa).startswith("</reasoning>"),
                     "pre_score_decoded": gp["pre_score"]["decoded"], "pre_score_next_decoded": gp["pre_score"]["next_decoded"],
                     "pre_score_next_is_parsed_digit": gp["pre_score"]["next_decoded"] == str(int(rec["score"])),
                     "digit_token_mass_post": rec["digit_token_mass_post"], "argmax_digit_post": rec["argmax_digit_post"],
                     "argmax_digit_post_matches_score": rec["argmax_digit_post_matches_score"],
                     "expected_score_post": rec["expected_score_post"], "expected_score_forced": rec["expected_score"],
                     "n_generated_tokens": rec["n_generated_tokens"], "analysis_note_text": rec["analysis_note_text"]}
        c3[label]["pass"] = (rec["capture_ok"] and c3[label]["analysis_end_next_is_end"] and c3[label]["reasoning_end_next_starts_close"]
                             and c3[label]["pre_score_next_is_parsed_digit"] and rec["digit_token_mass_post"] >= 0.99
                             and rec["argmax_digit_post_matches_score"])
        print(f"== check 3 [{label}]: pass={c3[label]['pass']}", flush=True)
        for n in rm.GEN_POSITIONS:
            e = gp[n]
            print(f"   {n:14s} gen_index {e['gen_index']} abs {e['token_index']} clean {e['clean']} | before {e['window_before']} | "
                  f"[{e['decoded']!r}] | after {e['window_after']}", flush=True)
        print(f"   pre_score digit mass {rec['digit_token_mass_post']:.6f} argmax {rec['argmax_digit_post']} score {rec['score']} "
              f"E_post {rec['expected_score_post']:.4f} vs E_forced {rec['expected_score']:.4f}", flush=True)
        # 6. shape / finiteness
        c6[label] = {"key": key, "shape": list(acts.shape), "dtype": str(acts.dtype), "all_finite": bool(torch.isfinite(acts).all()),
                     "n_nan_rows": int(torch.isnan(acts).flatten(1).any(1).sum()),
                     "row_norms_last_layer": {n: float(acts[i, -1].float().norm()) for i, n in enumerate(rm.POSITIONS)},
                     "pass": list(acts.shape) == [7, 25, 2880] and acts.dtype == torch.float16 and bool(torch.isfinite(acts).all())}
        print(f"== check 6 [{label}]: {json.dumps(c6[label])}", flush=True)
    # 6b. a shard written through shard_payload() carries the position order
    scratch = OUT / f"_prelaunch_shard_{TS}"; scratch.mkdir(parents=True, exist_ok=True)
    sp = scratch / "acts_0000.pt"
    torch.save(rm.shard_payload(new_acts, R["git_commit"], "original", "prelaunch"), sp)
    back = torch.load(sp)
    c6["shard"] = {"path": str(sp.relative_to(REPO)), "meta": back["_meta"], "tensor_keys": [k for k in back if k != "_meta"],
                   "position_order_present": back.get("_meta", {}).get("position_order") == list(rm.POSITIONS),
                   "tensors_roundtrip_equal": all(torch.equal(back[k], new_acts[k]) for k in new_acts),
                   "pass": back.get("_meta", {}).get("position_order") == list(rm.POSITIONS) and all(torch.equal(back[k], new_acts[k]) for k in new_acts)}
    print(f"== check 6 [shard]: {json.dumps({k: v for k, v in c6['shard'].items() if k != 'meta'})}", flush=True)
    R["checks"].update({"1_bitwise_vs_pass1": c1, "2_generation_determinism": c2, "3_position_decoding": c3, "6_shape_finite_order": c6})

    # ── 5. peak VRAM on the longest action <= 32k ─────────────────────────
    sub = json.loads((HERE.parent / "subset_main.json").read_text())["actions"]
    longest = max((a for a in sub if a["n_tokens"] <= args.max_prompt_tokens), key=lambda a: a["n_tokens"])
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    rec, acts = runner.run_action(longest, run_stage="prelaunch", run_order_index=None, hooks=True)
    c5 = {"key": f"{longest['traj_id']}|{longest['action_idx']}|original", "n_prompt_tokens": rec["n_prompt_tokens"], "status": rec["status"],
          "peak_gib": rec.get("peak_gib"), "peak_gib_process": torch.cuda.max_memory_allocated() / 2**30,
          "reserved_gib": torch.cuda.memory_reserved() / 2**30, "seconds": rec["seconds"], "prefill_seconds": rec.get("prefill_seconds"),
          "generation_seconds": rec.get("generation_seconds"), "n_generated_tokens": rec.get("n_generated_tokens"), "score": rec.get("score"),
          "capture_ok": rec.get("capture_ok"), "pass": rec["status"] == "ok" and rec["peak_gib"] <= 24.0}
    print(f"== check 5: {json.dumps(c5)}", flush=True)
    R["checks"]["5_peak_vram_longest"] = c5

    # ── summary ──────────────────────────────────────────────────────────
    summary = {"1_bitwise_vs_pass1": all(v["pass"] for v in c1.values()),
               "2_generation_determinism": all(v["pass"] for v in c2.values()),
               "3_position_decoding": all(v["pass"] for v in c3.values()),
               "4_resume": (R["checks"].get("4_resume_sigterm", {}).get("pass") and R["checks"].get("4b_resume_sigkill", {}).get("pass")) if not args.skip_resume else None,
               "5_peak_vram_le_24gib": c5["pass"],
               "6_shape_finite_order": all(v["pass"] for v in c6.values())}
    R["summary"] = summary
    R["all_pass"] = all(v for v in summary.values() if v is not None)
    RAW.mkdir(parents=True, exist_ok=True)
    out = RAW / f"g4_prelaunch_{TS}.json"
    out.write_text(json.dumps(R, indent=1, ensure_ascii=False))
    print("\n== SUMMARY", json.dumps(summary, indent=1), "\nALL PASS:", R["all_pass"], "\nraw ->", out.relative_to(REPO), flush=True)


if __name__ == "__main__":
    main()
