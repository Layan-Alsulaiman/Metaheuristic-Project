"""
helpers.py
----------
Contents
    1. Instance loading          -> load_instance()
    2. Solution evaluation        -> total_value(), total_weight()
    3. Feasibility repair / fill  -> repair(), random_fill()
    4. Initial solution           -> random_feasible_solution()
    5. Timing                     -> Timer
    6. Statistics / reporting     -> summarize(), gap_percent()
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
    name      : Dataset name 
    n         : number of items
    values    : list of profits  p_i
    weights   : list of weights  w_i
    capacity  : knapsack capacity C
    ratio_order : item indices sorted by descending value/weight ratio
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
    # read the file and split each line into columns, ignoring empty lines
    with open(path, "r") as f:
        tokens = [line.split() for line in f if line.strip()]

    # extract item profits and weights, and the knapsack capacity
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
# =====================================================================

def total_value(x, inst):
    """calculate the objective value: sum of profits of selected items."""
    return sum(v for v, xi in zip(inst.values, x) if xi)


def total_weight(x, inst):
    """Calculate the total weight of selected items to check feasibility."""
    return sum(w for w, xi in zip(inst.weights, x) if xi)


# =====================================================================
# 3. Feasibility repair
# =====================================================================

def repair(x, weight, value, inst, protected=None):
    """
    Make an overweight solution feasible.

    Strategy: remove selected items with the WORST value/weight ratio
    first, until the total weight fits the capacity.

    Parameters
    ----------
    x, weight, value : current solution and its (cached) weight / value
    protected        : optional item index that must not be removed

    Returns
    -------
    (weight, value) after repair. x is modified in place.
    """

    # no repair is needed if the solution is already feasible 
    if weight <= inst.capacity:
        return weight, value

    # start from the worst ratio to the best
    for i in reversed(inst.ratio_order):
        if x[i] and i != protected:
            x[i] = 0
            weight -= inst.weights[i]
            value -= inst.values[i]
            # stop removing items once the solution becomes feasible
            if weight <= inst.capacity:
                break
    return weight, value


def random_fill(x, weight, value, inst, rng, exclude=None):
    """
    Add random unselected items while they fit.

    Items are examined starting at a randomly
    selected index, introducing variation between candidate solutions.

    The excluded item is skipped to avoid immediately reversing
    the previous neighborhood move.

    Modifies x in place and returns the updated weight and value.
    """
    
    # calculate how much capacity is still available
    free = inst.capacity - weight
    if free <= 0:
        return weight, value
    # choose a random starting index
    start = rng.randrange(inst.n)
    for k in range(inst.n):
        i = (start + k) % inst.n
        # add the item only if it is unselected, allowed, and fits
        if not x[i] and i != exclude and inst.weights[i] <= free:
            x[i] = 1
            free -= inst.weights[i]
            weight += inst.weights[i]
            value += inst.values[i]
    return weight, value


# =====================================================================
# 4. Initial solution
# =====================================================================

def random_feasible_solution(inst, rng):
    """
    Generate a random, feasible initial solution for SA.

    Items are visited in a shuffled order, each with a 50% chance
    of being selected, provided it fits within the remaining capacity.
    """
    # start with an empty knapsack 
    x = [0] * inst.n
    weight = 0
    # shuffle the item indices to randomize the selection order
    order = list(range(inst.n))
    rng.shuffle(order)
    for i in order:
        if rng.random() < 0.5 and weight + inst.weights[i] <= inst.capacity:
            x[i] = 1
            weight += inst.weights[i]
    return x

# =====================================================================
# 5. Timing
# =====================================================================

class Timer:
    """    
    Measure elapsed execution time in seconds using a context manager.

    Used to compare the computational efficiency of Greedy and SA.
    """
    def __enter__(self):
        # record the start time
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        # calcuate the elapsed time
        self.elapsed = time.perf_counter() - self._start
        return False


# =====================================================================
# 6. Statistics / reporting
# =====================================================================

def gap_percent(found, optimum):
    """    
    Calculate the percentage gap between a solution and the
    published optimal objective value.

    A smaller gap indicates a better solution."""
    return 100.0 * (optimum - found) / optimum


def summarize(values):
    """Basic descriptive statistics over multiple runs."""
    a = np.asarray(values, dtype=float)
    return {
        "best": a.max(),
        "worst": a.min(),
        "mean": a.mean(),
        "median": float(np.median(a)),
        "std": a.std(ddof=1) if len(a) > 1 else 0.0,
    }


def make_rng(seed):
    """Create a random number generator initialized with a given seed."""
    return random.Random(seed)
