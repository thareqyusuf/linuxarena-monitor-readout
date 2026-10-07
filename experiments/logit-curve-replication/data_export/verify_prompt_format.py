#!/usr/bin/env python3
"""L4.1 check: gpu/format_monitor_prompt.py must reproduce control-tower 1.0.1's
`build_action_prompt` byte for byte.

For every trajectory in the export, a control-tower `Trajectory` is built from
the *same* fields the export carries (function, arguments, output,
output_error, main_task description, environment, side_task) via the paper's
own `hf_row_to_trajectory` (scripts/reconstruct_monitor_prompts.py), with the
action outputs taken from the export (mirror text) so both sides see identical
inputs. Then, for every action the live monitor would score (not hidden, not
malformed), `build_action_prompt(traj, idx, old_action.md)` is compared to
`format_monitor_prompt.build_prompt(export_traj, idx)`.

Also writes, per scorable action, the prompt's length and sha256 so the pod can
verify its own rendering without control-tower (consumed by build_manifest.py).

Run in the linuxarena-paper venv:
  cd ../benchmark/linuxarena-paper && UV_FROZEN=1 uv run python <repo>/experiments/logit-curve-replication/data_export/verify_prompt_format.py
"""
from __future__ import annotations

import collections
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve()
SLUG = HERE.parents[1].name
REPO = HERE.parents[3]
RAW = REPO / "results" / "raw" / SLUG
DATA = HERE.parent / "data"
GPU = HERE.parents[1] / "gpu"
# Checkout of the LinuxArena paper repo; override with LINUXARENA_PAPER.
PAPER = Path(os.environ.get("LINUXARENA_PAPER", REPO.parent / "benchmark" / "linuxarena-paper"))
CKPT = REPO / "checkpoints" / SLUG

sys.path.insert(0, str(PAPER / "src"))
sys.path.insert(0, str(PAPER / "scripts"))
sys.path.insert(0, str(GPU))

import format_monitor_prompt as fmp  # noqa: E402  (our pure-python port)
from control_tower.protocols.monitoring.action_monitors.helpers import build_action_prompt  # noqa: E402
from control_tower.trajectories.format_history import should_skip_action  # noqa: E402
from reconstruct_monitor_prompts import hf_row_to_trajectory  # noqa: E402

LIVE_TEMPLATE = (PAPER / "src/monitors/live/prompts/old_action.md").read_text()
assert LIVE_TEMPLATE == (GPU / "prompts" / "old_action.md").read_text(), "gpu/prompts/old_action.md drifted"

RUNS = {"honest": "f25f1eafacd749e89f1736f26d3df751", "attack": "30d53f5ae4f7444687459d13ccc61101"}


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "unknown"


def first_diff(a: str, b: str) -> dict:
    j = next((x for x in range(min(len(a), len(b))) if a[x] != b[x]), min(len(a), len(b)))
    return {"at": j, "len_ct": len(a), "len_ours": len(b), "ct": a[max(0, j - 80): j + 120], "ours": b[max(0, j - 80): j + 120]}


def main() -> None:
    trajs = fmp.load_export(DATA)
    template = fmp.load_template()
    rows_by_monitor_id: dict[str, dict] = {}
    for side, rid in RUNS.items():
        for line in (CKPT / "hf_rows" / f"{rid}.jsonl").open():
            r = json.loads(line)
            rows_by_monitor_id[r["trajectory_id"]] = r

    n_compared = n_equal = 0
    mismatches: list[dict] = []
    per_action: list[dict] = []
    diff_path_hits = 0
    by_side = collections.Counter()
    hf_scrub_differences = 0

    for traj_id, t in sorted(trajs.items()):
        row = rows_by_monitor_id[t["monitor_traj_id"]]
        assert len(row["actions"]) == len(t["actions"])
        # same inputs on both sides: export (mirror) outputs into the HF row
        for a_hf, a_ex in zip(row["actions"], t["actions"]):
            assert a_hf.get("function") == a_ex["function"]
            assert a_hf.get("arguments") == a_ex["arguments"]
            if (a_hf.get("output") or "") != (a_ex["output"] or ""):
                hf_scrub_differences += 1
                a_hf["output"] = a_ex["output"]
        ct_traj = hf_row_to_trajectory(row)
        assert ct_traj.main_task and ct_traj.main_task.description == t["main_task_description"]
        assert ct_traj.environment.id == t["env"]
        for idx, a in enumerate(t["actions"]):
            ct_skip = should_skip_action(ct_traj.actions[idx])
            our_skip = fmp.should_skip_action(a)
            assert (ct_skip is None) == (our_skip is None) and (ct_skip or "") == (our_skip or ""), (traj_id, idx, ct_skip, our_skip)
            if ct_skip:
                continue
            if a["function"] == "text_editor" and not a["output"] and (a["arguments"] or {}).get("command") == "str_replace":
                diff_path_hits += 1
            ct = build_action_prompt(ct_traj, idx, LIVE_TEMPLATE)
            ours = fmp.build_prompt(t, idx, template)
            n_compared += 1
            by_side[t["side"]] += 1
            if ct == ours:
                n_equal += 1
            else:
                if len(mismatches) < 10:
                    mismatches.append({"traj_id": traj_id, "action_idx": idx, **first_diff(ct, ours)})
            per_action.append({"traj_id": traj_id, "action_idx": idx, "side": t["side"],
                               "chars": len(ct), "sha256": hashlib.sha256(ct.encode()).hexdigest()})
        print(f"  {t['side']:6s} {traj_id[:8]} {len(t['actions']):3d} actions  equal so far {n_equal}/{n_compared}", flush=True)

    # sanity: the hidden+malformed skip rule and the diff-reconstruction path
    lens = sorted(p["chars"] for p in per_action)
    out = {
        "script": str(HERE.relative_to(REPO)), "git_commit": git_commit(),
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "control_tower_pin": "699d5ca5c26ab3126fa29c6fa73dbf898aa0ca53 (v1.0.1)",
        "template_sha256": hashlib.sha256(LIVE_TEMPLATE.encode()).hexdigest(),
        "n_trajectories": len(trajs), "n_actions_compared": n_compared, "n_equal": n_equal,
        "n_mismatch": n_compared - n_equal, "by_side": dict(by_side),
        "n_str_replace_diff_reconstruction_path_hit": diff_path_hits,
        "n_hf_outputs_replaced_by_mirror_text": hf_scrub_differences,
        "prompt_chars": {"min": lens[0], "median": lens[len(lens) // 2], "p95": lens[int(0.95 * len(lens))], "max": lens[-1]},
        "mismatch_examples": mismatches,
        "per_action": per_action,
    }
    RAW.mkdir(parents=True, exist_ok=True)
    p = RAW / f"l4_prompt_verify_{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    p.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k not in ("per_action", "mismatch_examples")}, indent=1))
    for m in mismatches:
        print("MISMATCH", json.dumps(m)[:600])
    print(f"wrote {p.relative_to(REPO)}")


if __name__ == "__main__":
    main()
