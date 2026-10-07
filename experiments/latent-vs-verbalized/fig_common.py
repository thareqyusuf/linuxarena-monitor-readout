"""Shared style for the four summary figures (FIGURES.md). Model-free; the fig_F*.py scripts read committed raws only."""
import json
from pathlib import Path
import numpy as np, matplotlib as mpl, matplotlib.pyplot as plt

mpl.rcParams.update({
    "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 100, "savefig.dpi": 200, "savefig.bbox": "tight",
})
# Okabe–Ito, fixed per scorer across all figures
C = {"integer": "#000000", "E": "#D55E00", "cached": "#0072B2",
     "score_probe": "#009E73", "label_probe": "#CC79A7", "text": "#999999", "dumb": "#E69F00"}
M = {"integer": ("o", ":"), "E": ("o", "-"), "cached": ("^", "-."),
     "score_probe": ("s", "-"), "label_probe": ("D", "--"), "text": ("x", "-"), "dumb": ("+", "-")}
HERE = Path(__file__).resolve().parent; SLUG = HERE.name; REPO = HERE.parents[1]
RAW = REPO / "results" / "raw" / SLUG; OUT = REPO / "results" / "figures" / SLUG / "summary"


def load(p):
    return json.load(open(p))


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"{name}.{ext}")
    print("saved", (OUT / f"{name}.png").relative_to(REPO), "+ .svg")
