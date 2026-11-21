"""
helpers.py
----------
Shared utility functions used by every other module in the project.

Keeping these in one place means the greedy algorithm, the simulated
annealing algorithm and the experiment driver all read data, evaluate
solutions and measure time in exactly the same way (a fair comparison).

Contents
    1. Instance loading          -> load_instance()
    2. Solution evaluation        -> total_value(), total_weight(), is_feasible()
    3. Feasibility repair / fill  -> repair(), random_fill()
    4. Initial solutions          -> random_feasible_solution(), empty_solution()
    5. Exact reference (optimum)  -> dp_optimal()
    6. Timing                     -> Timer
    7. Statistics / reporting     -> summarize(), gap_percent()
"""

import os
import time
import random
from dataclasses import dataclass, field

import numpy as np


# =====================================================================
# 1. Instance loading
# =====================================================================

@dataclass
class KnapsackInstance:
    """A 0/1 knapsack instance.

    Attributes
    ----------
    name      : file name without extension, e.g. 'kp_uc_500'
    n         : number of items
    values    : list of profits  p_i
    weights   : list of weights  w_i
    capacity  : knapsack capacity C
    ratio_order : item indices sorted by value/weight ratio, DESCENDING
                  (best items first). Pre-computed once because both the
                  greedy algorithm and the SA repair step need it.
    """
    name: str
    n: int
    values: list
    weights: list
    capacity: int
    ratio_order: list = field(default_factory=list)


def load_instance(path):
    """Read a KP_data file (https://github.com/whuph/KP_data).

    File format:
        line 1         : N  (number of items)
        next N lines   : index  profit  weight
        last line      : capacity C
    """
    with open(path, "r") as f:
        tokens = [line.split() for line in f if line.strip()]

    n = int(tokens[0][0])
    values, weights = [], []
    for row in tokens[1:1 + n]:
        # row = [index, profit, weight]
        values.append(int(row[1]))
        weights.append(int(row[2]))
    capacity = int(tokens[1 + n][0])

    # Sort by ratio p/w descending (ties broken by larger value first)
    ratio_order = sorted(range(n),
                         key=lambda i: (values[i] / weights[i], values[i]),
                         reverse=True)

    name = os.path.splitext(os.path.basename(path))[0]
    return KnapsackInstance(name, n, values, weights, capacity, ratio_order)


# =====================================================================
# 2. Solution evaluation
#    A solution is encoded as a binary list x of length n:
#    x[i] = 1  -> item i is in the knapsack,  x[i] = 0 -> it is not.
# =====================================================================

def total_value(x, inst):
    """Objective function f(x) = sum p_i * x_i."""
    return sum(v for v, xi in zip(inst.values, x) if xi)


def total_weight(x, inst):
    """Constraint left-hand side  sum w_i * x_i."""
    return sum(w for w, xi in zip(inst.weights, x) if xi)


def is_feasible(x, inst):
    """True if the solution respects the capacity constraint."""
    return total_weight(x, inst) <= inst.capacity


# =====================================================================
# 3. Feasibility repair
# =====================================================================

def repair(x, weight, value, inst, protected=None):
    """Make an overweight solution feasible (in place).

    Strategy: remove selected items with the WORST value/weight ratio
    first, until the total weight fits the capacity. This keeps the
    most "efficient" items, so the loss in value is as small as possible.

    Parameters
    ----------
    x, weight, value : current solution and its (cached) weight / value
    protected        : optional item index that must not be removed
                       (e.g. the item the neighbour move just added)

    Returns
    -------
    (weight, value) after repair. x is modified in place.
    """
    if weight <= inst.capacity:
        return weight, value

    # walk from the worst ratio to the best
    for i in reversed(inst.ratio_order):
        if x[i] and i != protected:
            x[i] = 0
            weight -= inst.weights[i]
            value -= inst.values[i]
            if weight <= inst.capacity:
                break
    return weight, value


def random_fill(x, weight, value, inst, rng, exclude=None):
    """Add random unselected items while they fit (in place).

    Scans all items once, starting from a random position, so it costs
    O(n) and never adds `exclude` (e.g. the item a move just removed).
    Returns the updated (weight, value).
    """
    free = inst.capacity - weight
    if free <= 0:
        return weight, value
    start = rng.randrange(inst.n)
    for k in range(inst.n):
        i = (start + k) % inst.n
        if not x[i] and i != exclude and inst.weights[i] <= free:
            x[i] = 1
            free -= inst.weights[i]
            weight += inst.weights[i]
            value += inst.values[i]
    return weight, value


# =====================================================================
# 4. Initial solutions
# =====================================================================

def empty_solution(inst):
    """All-zero solution (always feasible)."""
    return [0] * inst.n


def random_feasible_solution(inst, rng):
    """Random feasible start: visit items in random order and add each
    one with probability 0.5 if it still fits.

    rng : a random.Random object (so every run is reproducible by seed)
    """
    x = [0] * inst.n
    weight = 0
    order = list(range(inst.n))
    rng.shuffle(order)
    for i in order:
        if rng.random() < 0.5 and weight + inst.weights[i] <= inst.capacity:
            x[i] = 1
            weight += inst.weights[i]
    return x


# =====================================================================
# 5. Exact reference: dynamic programming
#    The dataset does not publish optimal values, so we compute them.
#    Complexity O(n * C) — fast for these instances (n<=1000, C~5000).
# =====================================================================

def dp_optimal(inst):
    """Return (optimal_value, optimal_solution) via 0/1 knapsack DP."""
    n, C = inst.n, inst.capacity
    w = np.array(inst.weights, dtype=np.int64)
    v = np.array(inst.values, dtype=np.int64)

    dp = np.zeros(C + 1, dtype=np.int64)          # dp[c] = best value with capacity c
    take = np.zeros((n, C + 1), dtype=bool)       # take[i, c] = item i used at capacity c

    for i in range(n):
        wi, vi = w[i], v[i]
        if wi > C:
            continue
        # candidate value if item i is taken (vectorised, iterates c downwards implicitly)
        cand = dp[:C + 1 - wi] + vi
        better = cand > dp[wi:]
        take[i, wi:] = better
        dp[wi:] = np.where(better, cand, dp[wi:])

    # backtrack to recover the chosen items
    x = [0] * n
    c = C
    for i in range(n - 1, -1, -1):
        if take[i, c]:
            x[i] = 1
            c -= inst.weights[i]
    return int(dp[C]), x


# =====================================================================
# 6. Timing
# =====================================================================

class Timer:
    """Context manager that measures wall-clock time in seconds.

    Usage:
        with Timer() as t:
            do_something()
        print(t.elapsed)
    """
    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self._start
        return False


# =====================================================================
# 7. Statistics / reporting
# =====================================================================

def gap_percent(found, optimum):
    """Relative gap to the optimum in %:  (opt - found) / opt * 100."""
    return 100.0 * (optimum - found) / optimum


def summarize(values):
    """Basic descriptive statistics over a list of numbers."""
    a = np.asarray(values, dtype=float)
    return {
        "best": a.max(),
        "worst": a.min(),
        "mean": a.mean(),
        "median": float(np.median(a)),
        "std": a.std(ddof=1) if len(a) > 1 else 0.0,
    }


def make_rng(seed):
    """Create an independent, reproducible random generator."""
    return random.Random(seed)
