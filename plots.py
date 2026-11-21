"""
plots.py
--------
Creates the figures for the report / slides from the files in results/.
Run after run_experiments.py.

Figures (results/figures/):
    convergence_<instance>.png  mean best-so-far value vs iteration, per neighbourhood,
                                with the greedy value and the optimum as reference lines
    runtime.png                 mean runtime per algorithm (log scale)
    gap.png                     best and mean gap to the optimum per algorithm
"""

import os
import json
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                     # no display needed
import matplotlib.pyplot as plt

import config

RESULTS_DIR = (
    os.path.join(config.RESULTS_DIR, "quick")
    if "--quick" in sys.argv else config.RESULTS_DIR
)

FIG_DIR = os.path.join(RESULTS_DIR, "figures")

# Fixed colour per entity (never by rank) — validated categorical slots 1-2
COLORS = {"Flip": "#2a78d6", "Swap": "#eb6834"}
GREEDY_COLOR = "#52514e"   # neutral ink for the reference algorithm
OPT_COLOR = "#0b0b0b"
GRID = "#e1e0d9"

plt.rcParams.update({
    "figure.dpi": 150, "font.size": 10, "axes.edgecolor": "#c3c2b7",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "lines.linewidth": 2,
})


def convergence_plot(inst, curves, optimum, greedy_value):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for nb, runs in curves.items():
        iters = [p[0] for p in runs[0]]
        vals = np.array([[p[1] for p in run] for run in runs])
        mean = vals.mean(axis=0)
        ax.plot(iters, mean, color=COLORS[nb], label=f"SA-{nb} (mean of {len(runs)} runs)")
        ax.fill_between(iters, vals.min(axis=0), vals.max(axis=0), color=COLORS[nb], alpha=0.12)
    ax.axhline(optimum, color=OPT_COLOR, ls=":", lw=1.5, label=f"Reference optimum = {optimum}")
    ax.axhline(greedy_value, color=GREEDY_COLOR, ls="--", lw=1.5,
               label=f"Greedy = {greedy_value}")
    # zoom on the interesting region (late iterations)
    lo = min(greedy_value, min(np.array([[p[1] for p in r] for r in runs])[:, -1].min()
                               for runs in curves.values()))
    ax.set_ylim(lo - 0.02 * optimum, optimum + 0.005 * optimum)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best value found so far")
    ax.set_title(f"SA convergence — {inst}", loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"convergence_{inst}.png"))
    plt.close(fig)


def _bar_color(alg):
    for nb, c in COLORS.items():
        if alg == f"SA-{nb}":
            return c
    return GREEDY_COLOR


def runtime_plot(summary):
    algs = ["Greedy-Ratio"] + [f"SA-{n}" for n in config.NEIGHBORHOODS]
    insts = list(summary["instance"].unique())
    fig, axes = plt.subplots(1, len(insts), figsize=(4 * len(insts), 3.8), sharey=True)
    for ax, inst in zip(np.atleast_1d(axes), insts):
        d = summary[summary["instance"] == inst].set_index("algorithm").loc[algs]
        ax.bar(range(len(algs)), d["mean_runtime_s"], color=[_bar_color(a) for a in algs],
               edgecolor="white", linewidth=2)
        for k, v in enumerate(d["mean_runtime_s"]):
            ax.text(k, v * 1.15, f"{v:.2g}s", ha="center", fontsize=8, color="#52514e")
        ax.set_yscale("log")
        ax.set_xticks(range(len(algs)), ["Greedy"] + algs[1:], rotation=20)
        ax.set_title(inst, loc="left")
    np.atleast_1d(axes)[0].set_ylabel("Mean runtime per run (s, log scale)")
    fig.suptitle("Efficiency: runtime", x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "runtime.png"))
    plt.close(fig)


def gap_plot(summary):
    algs = ["Greedy-Ratio"] + [f"SA-{n}" for n in config.NEIGHBORHOODS]
    insts = list(summary["instance"].unique())
    fig, axes = plt.subplots(1, len(insts), figsize=(4 * len(insts), 3.8), sharey=True)
    x = np.arange(len(algs))
    for ax, inst in zip(np.atleast_1d(axes), insts):
        d = summary[summary["instance"] == inst].set_index("algorithm").loc[algs]
        cols = [_bar_color(a) for a in algs]
        ax.bar(x - 0.2, d["best_gap_%"], 0.38, color=cols, edgecolor="white", linewidth=2,
               label="Best run")
        ax.bar(x + 0.2, d["mean_gap_%"], 0.38, color=cols, alpha=0.45, edgecolor="white",
               linewidth=2, hatch="//", label="Mean over runs")
        ax.set_xticks(x, ["Greedy"] + algs[1:], rotation=20)
        ax.set_title(inst, loc="left")
    np.atleast_1d(axes)[0].set_ylabel("Gap to optimum (%) — lower is better")
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Effectiveness: gap to the optimum", x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "gap.png"))
    plt.close(fig)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    greedy_df = pd.read_csv(os.path.join(RESULTS_DIR, "greedy_results.csv"))
    summary = pd.read_csv(os.path.join(RESULTS_DIR, "summary_table.csv"))

    with open(os.path.join(RESULTS_DIR, "convergence.json")) as f:
        conv = json.load(f)

    for inst, curves in conv["curves"].items():
        opt = conv["optima"][inst]
        g = greedy_df[(greedy_df["instance"] == inst) &
                      (greedy_df["algorithm"] == "Greedy-Ratio")]["value"].iloc[0]

        convergence_plot(inst, curves, opt, int(g))

    runtime_plot(summary)
    gap_plot(summary)
    print(f"Figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
