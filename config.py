"""
config.py
---------
Defines the selected datasets, SA parameters, number of independent
runs, and runtime measurement settings.

Keeping these settings in one file ensures that all experiments
use consistent parameters.
"""

from pathlib import Path

# project directories for loading data and saving results
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

# Exact optima
# Source: Symmetry (2025), Table 3.
# DOI: 10.3390/sym17091538
INSTANCES = {
    "kp_sc_500": 4078,
    "kp_sc_1000": 8228,
}

# the two neighborhood operators used to evaluate SA
NEIGHBORHOODS = ("Flip", "Swap")

# run SA 20 times per neighborhood and dataset using different seeds
# fixed seeds allow the experiments to be reproduced
N_RUNS = 20
BASE_SEED = 42

# SA parameters
MAX_ITERS = 50_000              # stop each SA run after this many iterations
P0 = 0.8                        # initial acceptance probability for worse moves      
T_FINAL = 0.01                  # final temperature
HISTORY_EVERY = 500             # record the best value every this many iterations

GREEDY_TIMING_REPEATS = 50      # repeat the greedy algorithm this many times to get a reliable runtime measurement