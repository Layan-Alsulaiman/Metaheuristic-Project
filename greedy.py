"""
greedy.py
---------
Algorithm B: greedy heuristics for the 0/1 knapsack problem.

Main algorithm used in the comparison:
    modified_greedy()  -> best of (ratio greedy, best single item).
                          This is the classic greedy with a guaranteed
                          approximation ratio of 1/2 of the optimum.

Extra variants (for the discussion / ablation table):
    greedy_by_ratio()  -> sort by value/weight, descending
    greedy_by_value()  -> sort by value, descending
    greedy_by_weight() -> sort by weight, ascending (lightest first)

All functions are deterministic: one run is enough.
They return (solution, value, weight).
"""

from helpers import empty_solution, total_value, total_weight


def _fill_in_order(inst, order):
    """Core greedy loop shared by every variant:
    scan the items in the given order and add each one if it still fits.
    """
    x = empty_solution(inst)
    weight = value = 0
    for i in order:
        if weight + inst.weights[i] <= inst.capacity:
            x[i] = 1
            weight += inst.weights[i]
            value += inst.values[i]
    return x, value, weight


def greedy_by_ratio(inst):
    """Take items with the highest value/weight ratio first."""
    return _fill_in_order(inst, inst.ratio_order)   # order pre-computed in helpers


def greedy_by_value(inst):
    """Take the most valuable items first."""
    order = sorted(range(inst.n), key=lambda i: inst.values[i], reverse=True)
    return _fill_in_order(inst, order)


def greedy_by_weight(inst):
    """Take the lightest items first."""
    order = sorted(range(inst.n), key=lambda i: inst.weights[i])
    return _fill_in_order(inst, order)


def modified_greedy(inst):
    """Ratio greedy, then compare with the single most valuable item
    that fits on its own; return whichever is better.

    Guarantee: value >= OPT / 2.
    """
    x, value, weight = greedy_by_ratio(inst)

    fitting = [i for i in range(inst.n) if inst.weights[i] <= inst.capacity]
    if fitting:
        best_item = max(fitting, key=lambda i: inst.values[i])
        if inst.values[best_item] > value:
            x = empty_solution(inst)
            x[best_item] = 1
            value, weight = inst.values[best_item], inst.weights[best_item]

    # sanity check: cached numbers must match a full re-evaluation
    assert value == total_value(x, inst) and weight == total_weight(x, inst)
    return x, value, weight


# Registry used by run_experiments.py
GREEDY_VARIANTS = {
    "Greedy-Ratio(modified)": modified_greedy,
    "Greedy-Ratio": greedy_by_ratio,
    "Greedy-Value": greedy_by_value,
    "Greedy-Weight": greedy_by_weight,
}
