"""
run_experiments.py
------------------
Main driver. Runs the full experimental protocol and writes all results.

For each instance in config.INSTANCES (name -> reference optimum):
    1. Run ratio Greedy (deterministic -> 1 run, timed over many repeats).
    2. Run Simulated Annealing N_RUNS times for each neighbourhood operator,
       each run with its own seed (BASE_SEED + run).

Outputs (folder results/, or results/quick/ with --quick):
    greedy_results.csv   one row per instance
    sa_runs.csv          one row per SA run
    summary_table.csv    the main comparison table (effectiveness + efficiency)
    convergence.json     best-so-far curves of every SA run (used by plots.py)

Usage:
    python run_experiments.py            # full experiment (20 runs)
    python run_experiments.py --quick    # 3 runs, 10k iterations (smoke test)
"""

import os
import sys
import json

import pandas as pd

import config
from helpers import (
    load_instance, gap_percent,
    summarize, Timer, total_value, total_weight
)
from greedy import greedy_by_ratio
from neighborhoods import NEIGHBORHOODS
from simulated_annealing import simulated_annealing


def run_greedy(inst, optimum):
    """Run ratio Greedy and average its runtime."""

    with Timer() as t:
        for _ in range(config.GREEDY_TIMING_REPEATS):
            x, value, weight = greedy_by_ratio(inst)

    assert value == total_value(x, inst)
    assert weight == total_weight(x, inst) <= inst.capacity

    gap = gap_percent(value, optimum)
    print(f"  Greedy-Ratio  value={value}  gap={gap:.3f}%")

    return {
        "instance": inst.name,
        "algorithm": "Greedy-Ratio",
        "value": value,
        "weight": weight,
        "capacity": inst.capacity,
        "optimum": optimum,
        "gap_%": gap,
        "runtime_s": t.elapsed / config.GREEDY_TIMING_REPEATS,
    }


def run_sa(inst, optimum, n_runs, sa_params):
    """Run SA n_runs times for each neighbourhood operator."""
    rows, curves = [], {}
    for nb_name in config.NEIGHBORHOODS:
        neighbor = NEIGHBORHOODS[nb_name]
        curves[nb_name] = []
        for r in range(n_runs):
            seed = config.BASE_SEED + r
            res = simulated_annealing(inst, neighbor, seed, **sa_params)
            rows.append({
                "instance": inst.name,
                "neighborhood": nb_name,
                "run": r + 1,
                "seed": seed,
                "value": res["value"],
                "weight": res["weight"],
                "optimum": optimum,
                "gap_%": gap_percent(res["value"], optimum),
                "runtime_s": res["runtime"],
                "T0": res["T0"],
                "acceptance_rate": res["acceptance_rate"],
            })
            curves[nb_name].append(res["history"])
        vals = [row["value"] for row in rows if row["neighborhood"] == nb_name]
        s = summarize(vals)
        print(f"  SA-{nb_name:<8} best={s['best']:.0f} mean={s['mean']:.1f} "
              f"worst={s['worst']:.0f} std={s['std']:.2f}")
    return rows, curves


def build_summary(greedy_df, sa_df):
    """One table comparing all algorithms on effectiveness and efficiency."""
    rows = []
    # Greedy: deterministic, so best = mean = worst
    for _, g in greedy_df.iterrows():
        rows.append({
            "instance": g["instance"], "algorithm": g["algorithm"], "runs": 1,
            "optimum": g["optimum"], "best": g["value"], "mean": g["value"],
            "worst": g["value"], "std": 0.0,
            "best_gap_%": g["gap_%"], "mean_gap_%": g["gap_%"],
            "optimum_hits": int(g["value"] == g["optimum"]),
            "mean_runtime_s": g["runtime_s"],
        })
    # SA: statistics over the independent runs
    for (inst, nb), grp in sa_df.groupby(["instance", "neighborhood"], sort=False):
        s = summarize(grp["value"])
        opt = grp["optimum"].iloc[0]
        rows.append({
            "instance": inst, "algorithm": f"SA-{nb}", "runs": len(grp),
            "optimum": opt, "best": s["best"], "mean": s["mean"],
            "worst": s["worst"], "std": s["std"],
            "best_gap_%": gap_percent(s["best"], opt),
            "mean_gap_%": gap_percent(s["mean"], opt),
            "optimum_hits": int((grp["value"] == opt).sum()),
            "mean_runtime_s": grp["runtime_s"].mean(),
        })
    return pd.DataFrame(rows)


def main():
    quick = "--quick" in sys.argv
    n_runs = 3 if quick else config.N_RUNS
    sa_params = {
        "max_iters": 10_000 if quick else config.MAX_ITERS,
        "p0": config.P0,
        "t_final": config.T_FINAL,
        "history_every": config.HISTORY_EVERY,
    }

    output_dir = (
        os.path.join(config.RESULTS_DIR, "quick")
        if quick else config.RESULTS_DIR
)
    os.makedirs(output_dir, exist_ok=True)

    greedy_rows, sa_rows, all_curves, optima = [], [], {}, {}

    for name, optimum in config.INSTANCES.items():
        inst = load_instance(config.DATA_DIR / f"{name}.txt")
        optima[inst.name] = optimum
        print(f"\n=== {inst.name} (n={inst.n}, C={inst.capacity}) "
              f"reference optimum = {optimum} ===")

        greedy_rows.append(run_greedy(inst, optimum))
        rows, curves = run_sa(inst, optimum, n_runs, sa_params)
        sa_rows += rows
        all_curves[inst.name] = curves

    greedy_df = pd.DataFrame(greedy_rows)
    sa_df = pd.DataFrame(sa_rows)
    summary = build_summary(greedy_df, sa_df)

    greedy_df.to_csv(os.path.join(output_dir, "greedy_results.csv"), index=False)
    sa_df.to_csv(os.path.join(output_dir, "sa_runs.csv"), index=False)
    summary.to_csv(os.path.join(output_dir, "summary_table.csv"), index=False)
    with open(os.path.join(output_dir, "convergence.json"), "w") as f:
        json.dump({"optima": optima, "curves": all_curves}, f)

    pd.set_option("display.width", 200)
    print("\n=== SUMMARY ===")
    print(summary.round(4).to_string(index=False))
    print(f"\nResults written to {output_dir}/")
    print("Now run: python plots.py" + (" --quick" if quick else ""))


if __name__ == "__main__":
    main()
