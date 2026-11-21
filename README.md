# 0/1 Knapsack: Simulated Annealing vs Greedy

CS1463 course project: a comparison of a metaheuristic (Simulated Annealing) and a greedy algorithm on the 0/1 knapsack problem.

## Requirements

- Python 3.8 or newer
- numpy, pandas, matplotlib

```bash
pip install -r requirements.txt
```

## How to run

```bash
python run_experiments.py          # full experiment: 2 instances x 3 neighbourhoods x 20 runs (~2.5 min)
python plots.py                    # builds the figures from the results
```

For a fast smoke test (3 runs, 10,000 iterations):

```bash
python run_experiments.py --quick
```

Python is interpreted, so there is no compile step.

## Project structure

| File | Role |
|------|------|
| `helpers.py` | Shared functions: load an instance, evaluate a solution, repair/fill, random initial solution, DP optimum, timer, statistics |
| `greedy.py` | **Algorithm B**: modified ratio greedy (main), plus ratio / value / weight variants |
| `neighborhoods.py` | Neighbourhood operators for SA: **Flip**, **Swap** (drop-add + fill), **Hybrid** (50/50) |
| `simulated_annealing.py` | **Algorithm A**: Simulated Annealing |
| `config.py` | All parameters (instances, runs, seeds, SA settings) |
| `run_experiments.py` | Runs everything and writes the CSV results |
| `plots.py` | Produces the figures in `results/figures/` |
| `data/` | The 20 instances from https://github.com/whuph/KP_data |

## Problem formulation

- **Encoding:** binary vector x ∈ {0,1}ⁿ, where xᵢ = 1 means item i is in the knapsack
- **Objective:** maximise Σ pᵢ xᵢ
- **Constraint:** Σ wᵢ xᵢ ≤ C
- **Constraint handling (SA):** repair. Any overweight solution drops its lowest value/weight items until it fits, so every solution SA visits is feasible.

## Experimental setup

| Setting | Value |
|---------|-------|
| Instances | `kp_sc_500` (medium, n=500, C=2518) and `kp_sc_1000` (large, n=1000, C=5068), strongly correlated |
| Best-known reference | Exact optimum computed by dynamic programming (`helpers.dp_optimal`) |
| SA runs | 20 independent runs per neighbourhood, seeds 1000–1019 |
| Stopping rule | 50,000 iterations per run |
| Initial temperature | Automatic: an average worsening move is accepted with probability p₀ = 0.8 |
| Cooling | Geometric, T ← αT, with α chosen so T reaches 0.01 at the last iteration |
| Initial solution | Random feasible solution |
| Greedy | Deterministic, so it runs once; its runtime is averaged over 50 repeats |

**Why strongly correlated instances?** We checked greedy against the DP optimum on all eight medium and large instances. Greedy is exactly optimal on 4 of the 8, and the uc_1000 instance is one of those. Only the `sc` pair has a gap to the optimum at both sizes, which gives the comparison something to measure.

## Outputs (`results/`)

- `summary_table.csv`: the main comparison table (best, mean, worst, std, gap %, optimum hits, runtime)
- `sa_runs.csv`: every individual SA run
- `greedy_results.csv`: all greedy variants
- `figures/`: convergence curves, box plots, gap chart and runtime chart
