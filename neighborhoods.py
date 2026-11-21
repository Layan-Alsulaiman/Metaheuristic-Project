"""
neighborhoods.py
----------------
Neighbourhood (move) operators for Simulated Annealing on 0/1 knapsack.

Each operator takes the CURRENT state and returns a NEW candidate state
without modifying the current one. A "state" is a tuple:
        (x, weight, value)
where x is the binary list and weight/value are cached so we never
re-evaluate the whole solution (O(1)/O(k) per move instead of O(n)).

Operators
    N1  flip_neighbor : flip one random bit (add OR remove one item)
                        -> small step, good for fine local improvement
    N2  swap_neighbor : remove one selected item AND add one unselected item,
                        then fill any free space with random items that fit
                        -> bigger, sideways step; keeps the knapsack full
    N3  hybrid_neighbor: choose N1 or N2 at random each iteration (50/50)
                        -> combines both neighbourhoods

If a move makes the knapsack overweight, helpers.repair() removes the
lowest-ratio items (never the item just added) until it fits again.
"""

from helpers import repair, random_fill


def flip_neighbor(state, inst, rng):
    """N1 — Bit-flip: pick a random item i and toggle x[i]."""
    x, weight, value = state
    y = x[:]                                  # copy -> current state untouched
    i = rng.randrange(inst.n)

    if y[i]:                                  # remove item i
        y[i] = 0
        weight -= inst.weights[i]
        value -= inst.values[i]
    else:                                     # add item i (may overflow)
        y[i] = 1
        weight += inst.weights[i]
        value += inst.values[i]
        weight, value = repair(y, weight, value, inst, protected=i)

    return y, weight, value


def swap_neighbor(state, inst, rng):
    """N2 — Swap (1-out / 1-in): remove a random selected item and add a
    random unselected item. Falls back to a flip if all items are in
    or all items are out.
    """
    x, weight, value = state

    # sample candidates by rejection: cheaper than building index lists
    # every iteration. Bounded tries keep it safe on extreme solutions.
    i_in = i_out = None
    for _ in range(4 * inst.n):
        j = rng.randrange(inst.n)
        if x[j] and i_out is None:
            i_out = j
        elif not x[j] and i_in is None:
            i_in = j
        if i_in is not None and i_out is not None:
            break
    if i_in is None or i_out is None:
        return flip_neighbor(state, inst, rng)

    y = x[:]
    y[i_out] = 0
    y[i_in] = 1
    weight += inst.weights[i_in] - inst.weights[i_out]
    value += inst.values[i_in] - inst.values[i_out]
    weight, value = repair(y, weight, value, inst, protected=i_in)

    # A pure 1-for-1 swap can never increase the number of items, so the
    # search could never fill the knapsack. After the swap we therefore use
    # any leftover space by adding random items that still fit.
    weight, value = random_fill(y, weight, value, inst, rng, exclude=i_out)
    return y, weight, value


def hybrid_neighbor(state, inst, rng):
    """N3 — Hybrid: 50% flip, 50% swap."""
    if rng.random() < 0.5:
        return flip_neighbor(state, inst, rng)
    return swap_neighbor(state, inst, rng)


# Registry used by run_experiments.py
NEIGHBORHOODS = {
    "Flip": flip_neighbor,
    "Swap": swap_neighbor,
    "Hybrid": hybrid_neighbor,
}
