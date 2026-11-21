"""Greedy algorithm using value-to-weight ratio."""

def greedy_by_ratio(inst):
    x = [0] * inst.n
    value = weight = 0

    for i in inst.ratio_order:
        if weight + inst.weights[i] <= inst.capacity:
            x[i] = 1
            weight += inst.weights[i]
            value += inst.values[i]

    return x, value, weight