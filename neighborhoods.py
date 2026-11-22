"""
neighborhoods.py
----------------
Neighbourhood operators for Simulated Annealing on 0/1 knapsack.

Each operator takes the CURRENT state and returns a NEW candidate state
without modifying the current one. A "state" is a tuple:
        (x, weight, value)

Operators
    N1  flip_neighbor : flip one random bit (add OR remove one item)

    N2  swap_neighbor : remove one selected item AND add one unselected item,
                        then fill any free space with random items that fit
"""

from helpers import repair, random_fill


def flip_neighbor(state, inst, rng):
    """    
    Generate a neighbor by flipping one randomly selected bit.

    If the item is selected, remove it.
    Otherwise, add it and repair the solution if it exceeds capacity."""
    x, weight, value = state
    # copy the current solution to avoid modifying it in place
    y = x[:]    
    # randomly select an item to flip                              
    i = rng.randrange(inst.n)

    if y[i]:
        # remove the selected item i                                  
        y[i] = 0
        weight -= inst.weights[i]
        value -= inst.values[i]
    else:
        # add the unselected item i and repair if unfeasible                                       
        y[i] = 1
        weight += inst.weights[i]
        value += inst.values[i]
        weight, value = repair(y, weight, value, inst, protected=i)

    return y, weight, value


def swap_neighbor(state, inst, rng):
    """    
    Generate a neighbor by exchanging one selected item with one
    unselected item.

    If either type of item cannot be found, use Flip instead.

    After swapping, repair any excess weight and attempt to fill
    remaining capacity with additional items that fit.
    """
    x, weight, value = state

    # randomly select one item to remove and one to add
    i_in = i_out = None
    # limit the number of attempts to find both types of items to avoid infinite loops
    for _ in range(4 * inst.n):
        j = rng.randrange(inst.n)
        if x[j] and i_out is None:
            i_out = j
        elif not x[j] and i_in is None:
            i_in = j
        if i_in is not None and i_out is not None:
            break
    # If either type of item cannot be found, use Flip instead
    if i_in is None or i_out is None:
        return flip_neighbor(state, inst, rng)
    # copy the solutiona and perform the swap
    y = x[:]
    y[i_out] = 0
    y[i_in] = 1
    # update the weight and value to reflect the swap, then repair if needed
    weight += inst.weights[i_in] - inst.weights[i_out]
    value += inst.values[i_in] - inst.values[i_out]
    weight, value = repair(y, weight, value, inst, protected=i_in)

    # fill any remaining capacity with random items that fit
    weight, value = random_fill(y, weight, value, inst, rng, exclude=i_out)
    return y, weight, value

# maps neighborhood names to their corresponding functions
NEIGHBORHOODS = {
    "Flip": flip_neighbor,
    "Swap": swap_neighbor
}
