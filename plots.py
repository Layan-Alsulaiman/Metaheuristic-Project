"""
plots.py
--------
Creates the figures for the report / slides from the files in results/.
Run after run_experiments.py.

Figures (results/figures/):
    convergence_<instance>.png  mean best-so-far value vs iteration, per neighbourhood,
                                with the greedy value and the optimum as reference lines
    boxplot_<instance>.png      distribution of final SA values over the 20 runs
    runtime.png                 mean runtime per algorithm (log scale)
    gap.png                     best and mean gap to the optimum per algorithm
"""

import os
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                     # no display needed
import matplotlib.pyplot as plt

import config

FIG_DIR = os.path.join(config.RESULTS_DIR, "figures")

# Fixed colour per entity (never by rank) — validated categorical slots 1-3
COLORS = {"Flip": "#2a78d6", "Swap": "#eb6834", "Hybrid": "#1baf7a"}
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
    ax.axhline(optimum, color=OPT_COLOR, ls=":", lw=1.5, label=f"Optimum (DP) = {optimum}")
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


def boxplot(inst, sa_df, optimum, greedy_value):
    d = sa_df[sa_df["instance"] == inst]
    names = config.NEIGHBORHOODS_TO_TEST
    data = [d[d["neighborhood"] == nb]["value"].values for nb in names]
    fig, ax = plt.subplots(figsize=(6, 4))
    bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                    medianprops={"color": OPT_COLOR})
    for patch, nb in zip(bp["boxes"], names):
        patch.set_facecolor(COLORS[nb]); patch.set_alpha(0.55); patch.set_edgecolor(COLORS[nb])
    for k, vals in enumerate(data, start=1):            # show every run as a dot
        jitter = np.random.default_rng(0).uniform(-0.08, 0.08, len(vals))
        ax.scatter(np.full(len(vals), k) + jitter, vals, s=12, color=COLORS[names[k - 1]],
                   edgecolor="white", linewidth=0.5, zorder=3)
    ax.axhline(optimum, color=OPT_COLOR, ls=":", lw=1.5, label="Optimum (DP)")
    ax.axhline(greedy_value, color=GREEDY_COLOR, ls="--", lw=1.5, label="Greedy")
    ax.set_xticks(range(1, len(names) + 1), [f"SA-{n}" for n in names])
    ax.set_ylabel("Final objective value")
    ax.set_title(f"SA final values over {len(data[0])} runs — {inst}", loc="left")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"boxplot_{inst}.png"))
    plt.close(fig)


def _bar_color(alg):
    for nb, c in COLORS.items():
        if alg == f"SA-{nb}":
            return c
    return GREEDY_COLOR


def runtime_plot(summary):
    algs = ["Greedy-Ratio(modified)"] + [f"SA-{n}" for n in config.NEIGHBORHOODS_TO_TEST]
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
    algs = ["Greedy-Ratio(modified)"] + [f"SA-{n}" for n in config.NEIGHBORHOODS_TO_TEST]
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
    sa_df = pd.read_csv(os.path.join(config.RESULTS_DIR, "sa_runs.csv"))
    greedy_df = pd.read_csv(os.path.join(config.RESULTS_DIR, "greedy_results.csv"))
    summary = pd.read_csv(os.path.join(config.RESULTS_DIR, "summary_table.csv"))
    with open(os.path.join(config.RESULTS_DIR, "convergence.json")) as f:
        conv = json.load(f)

    for inst, curves in conv["curves"].items():
        opt = conv["optima"][inst]
        g = greedy_df[(greedy_df["instance"] == inst) &
                      (greedy_df["algorithm"] == "Greedy-Ratio(modified)")]["value"].iloc[0]
        convergence_plot(inst, curves, opt, int(g))
        boxplot(inst, sa_df, opt, int(g))
    runtime_plot(summary)
    gap_plot(summary)
    print(f"Figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
