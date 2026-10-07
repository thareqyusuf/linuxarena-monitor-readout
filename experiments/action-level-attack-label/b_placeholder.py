"""B — placeholder control (TASK.md §B). Cached-score distribution, placeholder vs not, per side.

Groups from g2_prep per_action: honest-placeholder, honest-non, attack-placeholder, attack-non.
For each pair: mean cached_mean, p(cached_mean >= 6), p(both monitor scores >= 6), Mann-Whitney U
(stdlib normal approximation with tie correction, two-sided). Raw carries the full cached_mean
array per group so every number recomputes without the model.
"""
from __future__ import annotations

import collections
import datetime as dt
import json
import math
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
SLUG = HERE.name
REPO = HERE.parents[1]
EXPORT = REPO / "checkpoints" / "local-monitor-activations" / "data" / "export"
G2 = REPO / "results" / "raw" / "local-monitor-activations" / "g2_prep_20260910T103431Z.json"
RAW_DIR = REPO / "results" / "raw" / SLUG
FIG_DIR = REPO / "results" / "figures" / SLUG
TS = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def mannwhitney(x, y):
    """Two-sided Mann-Whitney U, normal approximation, tie-corrected. Returns U1, z, p, AUC."""
    allv = sorted([(v, 0) for v in x] + [(v, 1) for v in y])
    ranks, i, n = {}, 0, len(allv)
    tie_term = 0.0
    while i < n:
        j = i
        while j < n and allv[j][0] == allv[i][0]:
            j += 1
        r = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks.setdefault(k, r)
        t = j - i; tie_term += t ** 3 - t
        i = j
    r1 = sum(ranks[k] for k in range(n) if allv[k][1] == 0)
    n1, n2 = len(x), len(y)
    u1 = r1 - n1 * (n1 + 1) / 2.0
    mu = n1 * n2 / 2.0
    sigma = math.sqrt(n1 * n2 / 12.0 * ((n + 1) - tie_term / (n * (n - 1))))
    z = (u1 - mu) / sigma if sigma > 0 else 0.0
    p = math.erfc(abs(z) / math.sqrt(2))
    return u1, z, p, u1 / (n1 * n2)


def main():
    g2 = json.load(open(G2))["per_action"]
    both = {}
    for fn in ("attack.jsonl", "honest.jsonl"):
        with open(EXPORT / fn) as f:
            for line in f:
                if line.strip():
                    a = json.loads(line); ms = a["monitor_scores"]
                    both[(a["traj_id"], a["action_idx"])] = (all(v >= 6 for v in ms)) if len(ms) == 2 else None
    # `placeholder` in g2_prep is `"attachment://" in <whole rendered prompt>`, so it flags any action
    # whose HISTORY mentions an attachment. Split it: current_action = the attachment is in this
    # action's own tool_input (content genuinely hidden); history_only = everything else.
    cur = set()
    for fn in ("attack.jsonl", "honest.jsonl"):
        with open(EXPORT / fn) as f:
            for line in f:
                if line.strip():
                    a = json.loads(line)
                    if "attachment://" in json.dumps(a["tool_input"]):
                        cur.add((a["traj_id"], a["action_idx"]))
    groups = collections.defaultdict(list)
    for tid, idx, side, ph, cm in zip(g2["traj_id"], g2["action_idx"], g2["side"], g2["placeholder"], g2["cached_mean"]):
        k = (tid, idx)
        g = "current_action" if k in cur else ("history_only" if ph else "no_attachment")
        groups[(side, g)].append((cm, both[k]))
        groups[(side, "placeholder_flag_as_published" if ph else "non_placeholder_as_published")].append((cm, both[k]))

    def stats(vals):
        cm = [v for v, _ in vals]; b2 = [b for _, b in vals if b is not None]
        return {"n": len(cm), "mean": sum(cm) / len(cm), "median": sorted(cm)[len(cm) // 2],
                "p_cached_mean_ge6": sum(v >= 6 for v in cm) / len(cm),
                "p_both_ge6": (sum(b2) / len(b2)) if b2 else None, "n_two_scores": len(b2),
                "hist": dict(sorted(collections.Counter(str(v) for v in cm).items(), key=lambda kv: float(kv[0])))}

    out = {"slug": SLUG, "script": f"experiments/{SLUG}/b_placeholder.py", "timestamp": TS,
           "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip(),
           "g2_prep": str(G2.relative_to(REPO)), "groups": {}, "tests": {}, "cached_mean_arrays": {}}
    for side in ("honest", "attack"):
        for g in ("current_action", "history_only", "no_attachment", "placeholder_flag_as_published", "non_placeholder_as_published"):
            out["groups"][f"{side}_{g}"] = stats(groups[(side, g)])
            out["cached_mean_arrays"][f"{side}_{g}"] = [v for v, _ in groups[(side, g)]]
        for a, b in (("placeholder_flag_as_published", "non_placeholder_as_published"),
                     ("current_action", "no_attachment"), ("history_only", "no_attachment")):
            x = [v for v, _ in groups[(side, a)]]; y = [v for v, _ in groups[(side, b)]]
            if not x or not y:
                continue
            u, z, pv, auc = mannwhitney(x, y)
            out["tests"][f"{side}_{a}_vs_{b}"] = {"n_x": len(x), "n_y": len(y), "U1_x": u, "z": z, "p_two_sided": pv,
                                                  "auc_x_gt_y": auc,
                                                  "mean_diff": out["groups"][f"{side}_{a}"]["mean"] - out["groups"][f"{side}_{b}"]["mean"]}
    fig = None
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig_, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
        for ax, side in zip(axes, ("honest", "attack")):
            for g, c in (("no_attachment", "#888"), ("history_only", "#e8a33d"), ("current_action", "#c44")):
                v = out["cached_mean_arrays"][f"{side}_{g}"]
                ax.hist(v, bins=[i / 2 for i in range(0, 22)], density=True, alpha=0.6, color=c, label=f"{g} (n={len(v)})")
            ax.set_title(f"{side}: cached_mean by attachment location", fontsize=9); ax.legend(fontsize=7)
        FIG_DIR.mkdir(parents=True, exist_ok=True); fig = FIG_DIR / "b_placeholder_cached_hist.png"; fig_.tight_layout(); fig_.savefig(fig, dpi=150)
        out["figure"] = str(fig.relative_to(REPO))
    except ImportError as e:
        out["figure"] = None; print("!! no matplotlib:", e)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    rp = RAW_DIR / f"b_placeholder_{TS}.json"; rp.write_text(json.dumps(out, separators=(",", ":")))
    print("raw ->", rp.relative_to(REPO), f"({rp.stat().st_size/1e6:.2f} MB)", "figure ->", out["figure"])
    for k, v in out["groups"].items():
        print(f"{k:24} n={v['n']:6} mean={v['mean']:.3f} median={v['median']} p(mean>=6)={v['p_cached_mean_ge6']:.4f} p(both>=6)={v['p_both_ge6']:.4f} (n2={v['n_two_scores']})")
    for k, v in out["tests"].items():
        print(f"{k:52} n={v['n_x']:5}v{v['n_y']:5} z={v['z']:6.2f} p={v['p_two_sided']:.2e} AUC={v['auc_x_gt_y']:.3f} mean diff={v['mean_diff']:+.3f}")


if __name__ == "__main__":
    main()
