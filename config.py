"""
config.py
---------
All experiment settings in one place. Change values here, not in the code.
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# ---- Instances (medium + large) --------------------------------------
# Same instances are used for every algorithm.
INSTANCES = {
    "medium": "kp_sc_500.txt",    # 500 items, strongly correlated
    "large":  "kp_sc_1000.txt",   # 1000 items, strongly correlated
}

# ---- Simulated Annealing ---------------------------------------------
N_RUNS = 20                 # independent runs per (instance, neighbourhood)
BASE_SEED = 1000            # run r uses seed BASE_SEED + r
NEIGHBORHOODS_TO_TEST = ["Flip", "Swap", "Hybrid"]

SA_PARAMS = {
    "max_iters": 50_000,    # stopping rule: fixed iteration budget
    "p0": 0.8,              # initial acceptance prob. of an average worsening move
    "t_final": 0.01,        # final temperature
    "history_every": 500,   # convergence-curve resolution
}

# ---- Greedy -----------------------------------------------------------
GREEDY_TIMING_REPEATS = 50  # greedy is very fast: time it many times and average
