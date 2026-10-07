#!/usr/bin/env python3
"""L4.3: token-length histogram of every rendered monitor prompt under the chat
templates of openai/gpt-oss-20b and Qwen/Qwen3-30B-A3B (tokenizers only).

  HF_HOME=<cache> <venv>/bin/python experiments/logit-curve-replication/data_export/token_lengths.py

Method: `apply_chat_template(messages, add_generation_prompt=True, tokenize=False)`
renders the exact string the pod will feed the model; the rendered strings are
then tokenised in batches with `add_special_tokens=False`, which is what
`apply_chat_template(tokenize=True)` does internally (checked on the probe).

Writes results/raw/<slug>/l4_token_lengths_<ts>.json (per-action counts for
both models) and results/figures/<slug>/l4_token_lengths.png.

(The first run, l4_token_lengths_20260910T052617Z.json, is WRONG: it took
`len()` of a transformers-5 BatchEncoding, i.e. 2 for every prompt. Kept,
append-only.)
"""
from __future__ import annotations

import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

HERE = Path(__file__).resolve()
SLUG = HERE.parents[1].name
REPO = HERE.parents[3]
RAW = REPO / "results" / "raw" / SLUG
FIG = REPO / "results" / "figures" / SLUG
DATA = HERE.parent / "data"
GPU = HERE.parents[1] / "gpu"
sys.path.insert(0, str(GPU))
import format_monitor_prompt as fmp  # noqa: E402

MODELS = ["openai/gpt-oss-20b", "Qwen/Qwen3-30B-A3B"]
TMPL_KWARGS = {"openai/gpt-oss-20b": {}, "Qwen/Qwen3-30B-A3B": {"enable_thinking": True}}


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "unknown"


def n_tokens_batched(tok, texts: list[str], bs: int = 64) -> list[int]:
    out: list[int] = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i: i + bs], add_special_tokens=False)
        out.extend(len(x) for x in enc["input_ids"])
        if (i // bs) % 40 == 0:
            print(f"    {i}/{len(texts)}", flush=True)
    return out


def main() -> None:
    manifest = json.loads((GPU / "manifest.json").read_text())
    trajs = fmp.load_export(DATA)
    template = fmp.load_template()
    toks = {m: AutoTokenizer.from_pretrained(m) for m in MODELS}

    # probe: the wrapper each template adds, and that tokenize=False + tok() == tokenize=True
    probe = {}
    for m, tk in toks.items():
        msgs = [{"role": "user", "content": "PROBE"}]
        rendered = tk.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, **TMPL_KWARGS[m])
        direct = tk.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, **TMPL_KWARGS[m])
        direct_ids = direct["input_ids"] if not isinstance(direct, list) else direct
        via_text = tk(rendered, add_special_tokens=False)["input_ids"]
        probe[m] = {"rendered": rendered, "n_tokens_wrapper_plus_probe": len(direct_ids),
                    "tokenize_true_equals_text_path": list(direct_ids) == list(via_text)}
        assert probe[m]["tokenize_true_equals_text_path"], m

    print("rendering prompts ...", flush=True)
    keys = [(a["traj_id"], a["action_idx"], a["side"], a["pilot"]) for a in manifest["actions"]]
    prompts = [fmp.build_messages(trajs[t], i, template) for t, i, _, _ in keys]
    counts: dict[str, list[int]] = {}
    for m, tk in toks.items():
        print(f"{m}: rendering chat template ...", flush=True)
        texts = [tk.apply_chat_template(p, add_generation_prompt=True, tokenize=False, **TMPL_KWARGS[m]) for p in prompts]
        print(f"{m}: tokenising ...", flush=True)
        counts[m] = n_tokens_batched(tk, texts)
    per_action = [{"traj_id": t, "action_idx": i, "side": s, "pilot": p, **{m: counts[m][k] for m in MODELS}}
                  for k, (t, i, s, p) in enumerate(keys)]
    stats = {}
    for m in MODELS:
        c = np.array(counts[m])
        pil = np.array([r[m] for r in per_action if r["pilot"]])
        stats[m] = {"n": int(len(c)), "median": float(np.median(c)), "p95": float(np.percentile(c, 95)),
                    "p99": float(np.percentile(c, 99)), "max": int(c.max()), "mean": float(c.mean()), "total": int(c.sum()),
                    "n_over_16k": int((c > 16384).sum()), "n_over_32k": int((c > 32768).sum()),
                    "n_over_64k": int((c > 65536).sum()), "n_over_128k": int((c > 131072).sum()),
                    "by_side": {s: {"median": float(np.median([r[m] for r in per_action if r["side"] == s])),
                                    "total": int(sum(r[m] for r in per_action if r["side"] == s))} for s in ("honest", "attack")},
                    "pilot": {"n": int(len(pil)), "median": float(np.median(pil)), "p95": float(np.percentile(pil, 95)),
                              "max": int(pil.max()), "total": int(pil.sum())}}
    out = {"script": str(HERE.relative_to(REPO)), "git_commit": git_commit(),
           "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(), "models": MODELS,
           "chat_template_kwargs": TMPL_KWARGS, "probe": probe, "stats": stats, "per_action": per_action}
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    p = RAW / f"l4_token_lengths_{ts}.json"
    p.write_text(json.dumps(out, indent=1))
    print(json.dumps(stats, indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    hi = max(max(counts[m]) for m in MODELS)
    bins = np.logspace(np.log10(max(1, min(min(counts[m]) for m in MODELS)) * 0.9), np.log10(hi * 1.1), 60)
    for m, color in zip(MODELS, ["#0072B2", "#E69F00"]):
        ax.hist(counts[m], bins=bins, histtype="step", lw=1.6, color=color,
                label=f"{m}: median {stats[m]['median']:.0f}, p95 {stats[m]['p95']:.0f}, max {stats[m]['max']}")
    for x, lbl in [(16384, "16k"), (32768, "32k"), (65536, "64k"), (131072, "128k")]:
        ax.axvline(x, color="#cccccc", lw=0.8, ls="--")
        ax.text(x, 1, f" {lbl}", fontsize=7, color="#888888", ha="left", va="bottom", transform=ax.get_xaxis_transform())
    ax.set(xscale="log", xlabel="prompt tokens after chat template (log scale)", ylabel="actions",
           title=f"Live-monitor prompt length, {len(per_action)} scorable actions")
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    fig.tight_layout(); fig.savefig(FIG / "l4_token_lengths.png", dpi=150)
    print(f"wrote {p.relative_to(REPO)} and results/figures/{SLUG}/l4_token_lengths.png")


if __name__ == "__main__":
    main()
