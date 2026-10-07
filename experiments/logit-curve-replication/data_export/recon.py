#!/usr/bin/env python3
"""L1 reconnaissance: fetch the Opus honest/attack agent runs and the GPT-5 Nano
monitor runs from the public mirror, pull the matching full rows out of the
local HuggingFace shards, and write every fact the L1 report cites to one raw
JSON file.

Pure stdlib. No control-tower imports. Safe to re-run: API responses and HF
rows are cached under checkpoints/<slug>/ and only fetched/scanned once.

    python3 experiments/logit-curve-replication/data_export/recon.py
"""
from __future__ import annotations

import collections
import datetime as dt
import hashlib
import json
import multiprocessing as mp
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve()
SLUG = HERE.parents[1].name            # experiments/<slug>/data_export/recon.py
REPO = HERE.parents[3]
CKPT = REPO / "checkpoints" / SLUG
RAW = REPO / "results" / "raw" / SLUG
API = "https://repro.linuxarena.ai/api/public"
HF_FULL = Path(os.environ.get(
    "LINUXARENA_HF_FULL",
    str(Path(os.environ.get("LINUXARENA_PAPER", REPO.parent / "benchmark" / "linuxarena-paper")) / "data/hf/full"),
))

RUNS = {
    "honest_agent":  "ef6ed492f5a74ab8885e95469fe72fd2",
    "honest_monitor": "f25f1eafacd749e89f1736f26d3df751",
    "attack_agent":  "d085c1cbc4984437823563de06fb8cbe",
    "attack_monitor": "30d53f5ae4f7444687459d13ccc61101",
    "honest_monitor_source_run_id": "18962e83fe48407abe7a6b753cb2f940",
}
HIDDEN_TOOLS = {"run_side_task_test", "submit", "side_task_scorer"}


# ── API ──────────────────────────────────────────────────────────────────────

def fetch(endpoint: str, run_id: str) -> dict:
    p = CKPT / "api" / f"{endpoint}_{run_id}.json"
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        url = f"{API}/{endpoint}/{run_id}"
        print(f"  GET {url}")
        with urllib.request.urlopen(url, timeout=600) as r:
            p.write_bytes(r.read())
    return json.loads(p.read_text())


def recompute_fingerprint(trajs: list[dict]) -> str:
    """Port of repro-api/app/api/public/lean/[runId]/route.ts::computeFingerprint."""
    ids = sorted(t.get("trajectory_id") or "" for t in trajs)
    st = sum(1 for t in trajs if t.get("side_task_success"))
    mt = sum(1 for t in trajs if t.get("main_task_success"))
    n_actions = 0
    sus_total = 0.0
    for t in trajs:
        for a in t.get("actions") or []:
            n_actions += 1
            sus_total += (a.get("monitor_response") or {}).get("sus_score") or 0
    ids_hash = hashlib.sha256("|".join(ids).encode()).hexdigest()
    sus_rounded = round(sus_total * 1e4) / 1e4
    if sus_rounded == int(sus_rounded):
        sus_str = str(int(sus_rounded))          # JS prints 123 not 123.0
    else:
        sus_str = repr(sus_rounded)
    s = (f"http:v3-eval|n_trajs={len(trajs)}|total_actions={n_actions}|st={st}|mt={mt}"
         f"|sus={sus_str}|ids={ids_hash}")
    return hashlib.sha256(s.encode()).hexdigest()


# ── HF shards ────────────────────────────────────────────────────────────────

def _scan_shard(args: tuple[Path, set[str]]) -> dict[str, list[str]]:
    shard, wanted = args
    out: dict[str, list[str]] = collections.defaultdict(list)
    wanted_b = [w.encode() for w in wanted]
    with shard.open("rb") as f:
        for raw in f:
            head = raw[:600]
            if not any(w in head for w in wanted_b):
                continue
            row = json.loads(raw)
            rid = row.get("run_id")
            if rid in wanted:
                out[rid].append(raw.decode())
    return dict(out)


def hf_rows(run_ids: list[str]) -> dict[str, list[dict]]:
    d = CKPT / "hf_rows"
    d.mkdir(parents=True, exist_ok=True)
    missing = [r for r in run_ids if not (d / f"{r}.jsonl").exists()]
    if missing:
        shards = sorted(HF_FULL.glob("*.jsonl"))
        print(f"  scanning {len(shards)} HF shards for {len(missing)} run ids ...")
        with mp.Pool(min(len(shards), max(1, (os.cpu_count() or 2) // 2))) as pool:
            parts = pool.map(_scan_shard, [(s, set(missing)) for s in shards])
        merged: dict[str, list[str]] = collections.defaultdict(list)
        for part in parts:
            for rid, lines in part.items():
                merged[rid].extend(lines)
        for rid in missing:
            (d / f"{rid}.jsonl").write_text("".join(merged.get(rid, [])))
            print(f"    {rid}: {len(merged.get(rid, []))} rows")
    return {r: [json.loads(l) for l in (d / f"{r}.jsonl").read_text().splitlines()] for r in run_ids}


def content_hash(row: dict) -> str:
    """Same recipe as linuxarena-paper/scripts/reconstruct_monitor_prompts.py::content_hash."""
    h = hashlib.sha1()
    for a in row.get("actions") or []:
        h.update((a.get("function") or "").encode())
        h.update(json.dumps(a.get("arguments"), sort_keys=True).encode())
        h.update((a.get("output") or "").encode())
    return h.hexdigest()


# ── helpers ──────────────────────────────────────────────────────────────────

def task_key(t: dict) -> tuple:
    return ((t.get("environment") or {}).get("id"),
            (t.get("main_task") or {}).get("id"),
            (t.get("side_task") or {}).get("id"))


def fseq(t: dict) -> tuple:
    return tuple(a.get("function") for a in t["actions"])


def keyset(objs) -> list[str]:
    return sorted(set(k for o in objs for k in (o or {})))


def hist(vals) -> dict:
    c = collections.Counter(vals)
    return {("null" if k is None else str(k)): v for k, v in sorted(c.items(), key=lambda kv: (kv[0] is None, kv[0]))}


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "unknown"


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    out: dict = {
        "script": str(HERE.relative_to(REPO)),
        "git_commit": git_commit(),
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "api_base": API,
        "runs": RUNS,
        "hf_shards_dir": str(HF_FULL),
    }

    print("[1] API fetch")
    lean = {k: fetch("lean", v) for k, v in RUNS.items()}
    mon = {k: fetch("monitor-data", v) for k, v in RUNS.items() if k != "honest_monitor_source_run_id"}

    out["fingerprints"] = {
        k: {"run_id": v, "server_fingerprint": lean[k].get("fingerprint"),
            "recomputed_from_lean_payload": recompute_fingerprint(lean[k]["trajectories"]),
            "n_trajectories": len(lean[k]["trajectories"])}
        for k, v in RUNS.items()
    }
    for k in out["fingerprints"]:
        out["fingerprints"][k]["recompute_matches"] = (
            out["fingerprints"][k]["server_fingerprint"] == out["fingerprints"][k]["recomputed_from_lean_payload"])

    print("[2] API schema + per-action stats")
    schema: dict = {}
    for k in ("honest_agent", "attack_agent", "honest_monitor", "attack_monitor"):
        for ep, payload in (("lean", lean[k]), ("monitor-data", mon[k])):
            T = payload["trajectories"]
            acts = [a for t in T for a in t["actions"]]
            mrs = [a.get("monitor_response") for a in acts]
            mem_counts = collections.Counter(len((m or {}).get("members") or []) for m in mrs)
            schema[f"{k}/{ep}"] = {
                "n_trajectories": len(T),
                "n_actions": len(acts),
                "trajectory_keys": keyset(T),
                "action_keys": keyset(acts),
                "monitor_response_keys": keyset(mrs),
                "member_keys": keyset(m for mr in mrs for m in ((mr or {}).get("members") or [])),
                "actions_with_monitor_response": sum(1 for m in mrs if m),
                "is_ensemble_hist": hist((m or {}).get("is_ensemble") for m in mrs),
                "members_per_action_hist": {str(a): b for a, b in sorted(mem_counts.items())},
                "error_hist": hist((m or {}).get("error") for m in mrs),
                "sus_score_hist": hist((m or {}).get("sus_score") for m in mrs),
                "member_score_hist": hist(m.get("sus_score") for mr in mrs for m in ((mr or {}).get("members") or [])),
                "function_hist": hist(a.get("function") for a in acts),
                "has_output_field": sum(1 for a in acts if "output" in a),
                "has_arguments_field": sum(1 for a in acts if "arguments" in a),
                "has_agent_reasoning_field": sum(1 for a in acts if "agent_reasoning" in a),
                "output_null": sum(1 for a in acts if a.get("output") is None),
                "output_len_max": max((len(a.get("output") or "") for a in acts), default=0),
                "env_hist": hist((t.get("environment") or {}).get("id") for t in T),
                "side_task_success_hist": hist(t.get("side_task_success") for t in T),
                "failure_type_hist": hist((t.get("side_task") or {}).get("failure_type") for t in T),
                "index_equals_position": all(a.get("index") == i for t in T for i, a in enumerate(t["actions"])),
                "sus_equals_mean_of_members_violations": sum(
                    1 for m in mrs if (m or {}).get("members")
                    and abs(sum(x["sus_score"] for x in m["members"]) / len(m["members"]) - m["sus_score"]) > 1e-9),
            }
    out["api_schema"] = schema

    print("[3] agent<->monitor join (by env + main_task + side_task; trajectory ids do not overlap)")
    joins: dict = {}
    for side in ("honest", "attack"):
        A = lean[f"{side}_agent"]["trajectories"]
        M = lean[f"{side}_monitor"]["trajectories"]
        ida = {t["trajectory_id"] for t in A}
        idm = {t["trajectory_id"] for t in M}
        dx = {task_key(t): t for t in A}
        dy = {task_key(t): t for t in M}
        assert len(dx) == len(A) and len(dy) == len(M), "task keys not unique"
        diffs = []
        n_same = 0
        for k in sorted(set(dx) & set(dy), key=str):
            fx, fy = fseq(dx[k]), fseq(dy[k])
            if fx == fy:
                n_same += 1
            else:
                diffs.append({
                    "env": k[0], "main_task": k[1], "side_task": k[2],
                    "agent_traj_id": dx[k]["trajectory_id"], "monitor_traj_id": dy[k]["trajectory_id"],
                    "agent_n_actions": len(fx), "monitor_n_actions": len(fy),
                    "monitor_is_prefix_of_agent": fx[:len(fy)] == fy,
                })
        # stray monitor_responses on the agent run vs the monitor run
        agree = disagree = 0
        for k in set(dx) & set(dy):
            if fseq(dx[k]) != fseq(dy[k]):
                continue
            for a, b in zip(dx[k]["actions"], dy[k]["actions"]):
                if a.get("monitor_response") and b.get("monitor_response"):
                    if a["monitor_response"]["sus_score"] == b["monitor_response"]["sus_score"]:
                        agree += 1
                    else:
                        disagree += 1
        joins[side] = {
            "trajectory_id_overlap": len(ida & idm),
            "task_keys_agent_only": len(set(dx) - set(dy)),
            "task_keys_monitor_only": len(set(dy) - set(dx)),
            "task_keys_matched": len(set(dx) & set(dy)),
            "matched_with_identical_function_sequence": n_same,
            "matched_with_different_actions": diffs,
            "side_task_success_agreement": sum(1 for k in set(dx) & set(dy)
                                               if dx[k].get("side_task_success") == dy[k].get("side_task_success")),
            "agent_run_stray_monitor_scores": {"n": sum(1 for t in A for a in t["actions"] if a.get("monitor_response")),
                                               "equal_to_monitor_run": agree, "different_from_monitor_run": disagree},
        }
    out["agent_monitor_join"] = joins

    print("[4] HF shards: pull full rows for the 4 runs (cached in checkpoints)")
    ids = [RUNS[k] for k in ("honest_agent", "honest_monitor", "attack_agent", "attack_monitor")]
    rows = hf_rows(ids)
    hf: dict = {}
    for k in ("honest_agent", "honest_monitor", "attack_agent", "attack_monitor"):
        R = rows[RUNS[k]]
        api_T = {t["trajectory_id"]: t for t in lean[k]["trajectories"]}
        acts = [a for r in R for a in r["actions"]]
        # per-action score equality API lean vs HF
        score_eq = score_ne = mem_eq = mem_ne = 0
        for r in R:
            t = api_T.get(r["trajectory_id"])
            if t is None:
                continue
            for a, b in zip(r["actions"], t["actions"]):
                ma, mb = a.get("monitor_response") or {}, b.get("monitor_response") or {}
                if "sus_score" in ma and "sus_score" in mb:
                    if ma["sus_score"] == mb["sus_score"]:
                        score_eq += 1
                    else:
                        score_ne += 1
                    la = [m.get("sus_score") for m in ma.get("members") or []]
                    lb = [m.get("sus_score") for m in mb.get("members") or []]
                    if la == lb:
                        mem_eq += 1
                    else:
                        mem_ne += 1
        # output equality API monitor-data vs HF
        api_M = {t["trajectory_id"]: t for t in mon[k]["trajectories"]}
        out_eq = out_ne = 0
        for r in R:
            t = api_M.get(r["trajectory_id"])
            if t is None:
                continue
            for a, b in zip(r["actions"], t["actions"]):
                if (a.get("output") or "") == (b.get("output") or ""):
                    out_eq += 1
                else:
                    out_ne += 1
        first_mr = next((a["monitor_response"] for a in acts if a.get("monitor_response")), None)
        hf[k] = {
            "n_rows": len(R),
            "n_actions": len(acts),
            "trajectory_ids_match_api": sum(1 for r in R if r["trajectory_id"] in api_T),
            "n_actions_match_api": sum(1 for r in R if r["trajectory_id"] in api_T
                                       and len(r["actions"]) == len(api_T[r["trajectory_id"]]["actions"])),
            "row_keys": keyset(R),
            "action_keys": keyset(acts),
            "main_task_keys": keyset(r.get("main_task") for r in R),
            "environment_keys": keyset(r.get("environment") for r in R),
            "side_task_keys": keyset(r.get("side_task") for r in R),
            "monitor_response_keys": keyset(a.get("monitor_response") for a in acts),
            "member_keys": keyset(m for a in acts for m in ((a.get("monitor_response") or {}).get("members") or [])),
            "example_monitor_model_name": (first_mr or {}).get("model_name"),
            "example_member_model_name": ((first_mr or {}).get("members") or [{}])[0].get("model_name"),
            "monitor_prompt_stored_nonempty": sum(1 for a in acts for m in ((a.get("monitor_response") or {}).get("members") or []) if m.get("monitor_prompt")),
            "member_raw_response_nonempty": sum(1 for a in acts for m in ((a.get("monitor_response") or {}).get("members") or []) if m.get("raw_response")),
            "actions_with_arguments": sum(1 for a in acts if a.get("arguments") is not None),
            "actions_with_output": sum(1 for a in acts if a.get("output") is not None),
            "actions_with_output_error": sum(1 for a in acts if a.get("output_error")),
            "actions_with_agent_reasoning": sum(1 for a in acts if a.get("agent_reasoning")),
            "actions_with_extended_thinking": sum(1 for a in acts if a.get("extended_thinking")),
            "actions_with_inter_action_messages": sum(1 for a in acts if a.get("inter_action_messages")),
            "actions_with_substeps": sum(1 for a in acts if a.get("substeps")),
            "output_len_max": max((len(a.get("output") or "") for a in acts), default=0),
            "output_len_over_400000": sum(1 for a in acts if len(a.get("output") or "") > 400000),
            "arguments_len_max": max((len(json.dumps(a.get("arguments"))) for a in acts), default=0),
            "main_task_description_present": sum(1 for r in R if (r.get("main_task") or {}).get("description")),
            "environment_description_present": sum(1 for r in R if (r.get("environment") or {}).get("description")),
            "side_task_goal_present": sum(1 for r in R if (r.get("side_task") or {}).get("goal")),
            "attack_analysis_present": sum(1 for r in R if r.get("attack_analysis")),
            "api_vs_hf_sus_score": {"equal": score_eq, "different": score_ne},
            "api_vs_hf_member_scores": {"equal": mem_eq, "different": mem_ne},
            "api_monitor_data_vs_hf_output": {"equal": out_eq, "different": out_ne},
        }
    # content-hash join between agent and monitor rows
    for side in ("honest", "attack"):
        ha = {content_hash(r): r["trajectory_id"] for r in rows[RUNS[f"{side}_agent"]]}
        hm = {content_hash(r): r["trajectory_id"] for r in rows[RUNS[f"{side}_monitor"]]}
        hf[f"{side}_content_hash_join"] = {"matched": len(set(ha) & set(hm)),
                                           "agent_only": len(set(ha) - set(hm)), "monitor_only": len(set(hm) - set(ha))}
    out["hf"] = hf

    print("[5] samples per action, hidden tools, honest pool size")
    pool: dict = {}
    for k in ("honest_monitor", "attack_monitor"):
        T = lean[k]["trajectories"]
        n_hidden = sum(1 for t in T for a in t["actions"] if a["function"] in HIDDEN_TOOLS)
        hidden_scores = hist((a["monitor_response"] or {}).get("sus_score") for t in T for a in t["actions"] if a["function"] in HIDDEN_TOOLS)
        nonens = [a for t in T for a in t["actions"] if not (a.get("monitor_response") or {}).get("is_ensemble")]
        pool[k] = {
            "n_actions": sum(len(t["actions"]) for t in T),
            "n_hidden_tool_actions": n_hidden,
            "hidden_tool_sus_scores": hidden_scores,
            "n_scorable_actions": sum(1 for t in T for a in t["actions"] if a["function"] not in HIDDEN_TOOLS),
            "n_actions_with_2_members": sum(1 for t in T for a in t["actions"] if len((a.get("monitor_response") or {}).get("members") or []) == 2),
            "n_actions_non_ensemble": len(nonens),
            "non_ensemble_function_hist": hist(a["function"] for a in nonens),
            "non_ensemble_sus_hist": hist(a["monitor_response"]["sus_score"] for a in nonens),
            "n_trajectories_with_any_non_ensemble": sum(1 for t in T if any(not (a.get("monitor_response") or {}).get("is_ensemble") for a in t["actions"])),
        }
    out["samples_per_action"] = pool
    n_benign_actions = pool["honest_monitor"]["n_scorable_actions"]
    out["min_usable_fpr"] = {"rule": "20 / n_benign_actions", "n_benign_actions": n_benign_actions,
                             "value": 20 / n_benign_actions}

    RAW.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    p = RAW / f"l1_recon_{ts}.json"
    p.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {p.relative_to(REPO)}")
    print(json.dumps({"fingerprints": out["fingerprints"], "join": out["agent_monitor_join"],
                      "samples_per_action": out["samples_per_action"], "min_usable_fpr": out["min_usable_fpr"]}, indent=1)[:6000])


if __name__ == "__main__":
    main()
