# 0/1 Knapsack: Simulated Annealing vs Ratio Greedy

CS1463 project: compare one Greedy algorithm and one metaheuristic using the same two knapsack instances.

## Setup

Python 3.8+ and `pip install -r requirements.txt`.

## Run

```bash
python run_experiments.py --quick  # smoke test: 3 runs and 10,000 SA iterations
python plots.py --quick
python run_experiments.py          # final: 20 runs and 50,000 SA iterations
python plots.py
```

Quick test outputs are saved in `results/quick/`; full results in `results/`.

## Algorithms

- **Greedy:** take items in descending value-to-weight ratio when they fit.
- **Simulated Annealing:** binary solutions with Flip and Swap neighborhoods. Repair removes low-ratio items to keep candidates feasible. Swap also fills remaining capacity with randomly scanned feasible items.
- **Acceptance:** better/equal candidates are accepted; worse candidates may be accepted with probability `exp(delta / T)`.
- **Cooling:** geometric from an automatically estimated initial temperature to `T_FINAL`.

## Experimental setup

- Instances: `kp_sc_500` (500 items, capacity 2518), `kp_sc_1000` (1000 items, capacity 5068).
- Reference optimal values: 4078 and 8228,
  obtained from Symmetry (2025), Table 3.
  DOI: 10.3390/sym17091538.
- Independent SA seeds: 42 through 61, 20 runs per neighborhood per instance.
- Full run: 50,000 iterations; initial target acceptance probability 0.8, final temperature 0.01; record convergence every 500 iterations.
- Greedy is deterministic and run once for its value. Its runtime is averaged over 50 calls.
- Results include best and mean objective values, worst value, sample standard deviation, gap to reference optimum, and mean runtime.

## Files

- `config.py`: experiment settings and reference optimum values.
- `data/`: the 20 instances from [KP_data](https://github.com/whuph/KP_data) plus source-format notes (only the two above are used).
- `helpers.py`: dataset parsing, solution evaluation, repair/fill, random initial solutions, timing and statistics.
- `greedy.py`: ratio-based Greedy.
- `neighborhoods.py`: Flip and Swap operators for SA.
- `simulated_annealing.py`: SA implementation.
- `run_experiments.py`: executes the comparison and saves CSV/JSON results.
- `plots.py`: creates effectiveness, runtime, and two convergence figures.
- `requirements.txt`: Python dependencies.

## Interpreting the results

Gap (%) = `(reference optimum - found value) / reference optimum * 100`.
Lower gap is better; higher objective value is better. Greedy is a deterministic baseline, whereas SA is repeated with different seeds.
