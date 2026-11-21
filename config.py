"""
config.py
---------
All experiment settings in one place. Change values here, not in the code.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

# Exact optima previously computed separately
INSTANCES = {
    "kp_sc_500": 4078,
    "kp_sc_1000": 8228,
}

NEIGHBORHOODS = ("Flip", "Swap")

N_RUNS = 20
BASE_SEED = 42

MAX_ITERS = 50_000
P0 = 0.8
T_FINAL = 0.01
HISTORY_EVERY = 500

GREEDY_TIMING_REPEATS = 50