"""
Greedy algorithm using value-to-weight ratio.
Strategy: Select items in descending order of value-to-weight ratio.
An item is selected only if it fits within the remaining capacity.
"""

from helpers import sort_by_ratio


def greedy_by_ratio(inst):
    # 1 = selected, 0 = not selected
    x = [0] * inst.n
    value = weight = 0

    for i in sort_by_ratio(inst.values, inst.weights):
        # select item if it fits in the knapsack
        if weight + inst.weights[i] <= inst.capacity:
            x[i] = 1
            weight += inst.weights[i]
            value += inst.values[i]

    return x, value, weight
