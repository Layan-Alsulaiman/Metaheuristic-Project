"""
run_experiments.py
------------------
Main driver. Runs the full experimental protocol and writes all results.

For each instance in config.INSTANCES:
    1. Compute the exact optimum with dynamic programming (best-known reference).
    2. Run the greedy variants (deterministic -> 1 run, timed over many repeats).
    3. Run Simulated Annealing N_RUNS times for each neighbourhood operator,
       each run with its own seed (BASE_SEED + run).

Outputs (folder results/):
    greedy_results.csv   one row per (instance, greedy variant)
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
from helpers import load_instance, dp_optimal, gap_percent, summarize, Timer
from greedy import GREEDY_VARIANTS
from neighborhoods import NEIGHBORHOODS
from simulated_annealing import simulated_annealing


def run_greedy(inst, optimum):
    """Run every greedy variant once; average its runtime over many repeats."""
    rows = []
    for name, algo in GREEDY_VARIANTS.items():
        with Timer() as t:
            for _ in range(config.GREEDY_TIMING_REPEATS):
                _, value, weight = algo(inst)
        rows.append({
            "instance": inst.name,
            "algorithm": name,
            "value": value,
            "weight": weight,
            "capacity": inst.capacity,
            "optimum": optimum,
            "gap_%": gap_percent(value, optimum),
            "runtime_s": t.elapsed / config.GREEDY_TIMING_REPEATS,
        })
        print(f"  {name:<24} value={value:<7} gap={rows[-1]['gap_%']:.3f}%")
    return rows


def run_sa(inst, optimum, n_runs, sa_params):
    """Run SA n_runs times for each neighbourhood operator."""
    rows, curves = [], {}
    for nb_name in config.NEIGHBORHOODS_TO_TEST:
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
    sa_params = dict(config.SA_PARAMS)
    if quick:
        sa_params["max_iters"] = 10_000

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    greedy_rows, sa_rows, all_curves, optima = [], [], {}, {}

    for size, filename in config.INSTANCES.items():
        inst = load_instance(os.path.join(config.DATA_DIR, filename))
        optimum, _ = dp_optimal(inst)
        optima[inst.name] = optimum
        print(f"\n=== {inst.name} ({size}: n={inst.n}, C={inst.capacity}) "
              f"optimum (DP) = {optimum} ===")

        greedy_rows += run_greedy(inst, optimum)
        rows, curves = run_sa(inst, optimum, n_runs, sa_params)
        sa_rows += rows
        all_curves[inst.name] = curves

    greedy_df = pd.DataFrame(greedy_rows)
    sa_df = pd.DataFrame(sa_rows)
    summary = build_summary(greedy_df, sa_df)

    greedy_df.to_csv(os.path.join(config.RESULTS_DIR, "greedy_results.csv"), index=False)
    sa_df.to_csv(os.path.join(config.RESULTS_DIR, "sa_runs.csv"), index=False)
    summary.to_csv(os.path.join(config.RESULTS_DIR, "summary_table.csv"), index=False)
    with open(os.path.join(config.RESULTS_DIR, "convergence.json"), "w") as f:
        json.dump({"optima": optima, "curves": all_curves}, f)

    pd.set_option("display.width", 200)
    print("\n=== SUMMARY ===")
    print(summary.round(4).to_string(index=False))
    print(f"\nResults written to {config.RESULTS_DIR}/  ->  now run: python plots.py")


if __name__ == "__main__":
    main()
