"""
Algorithm A: Simulated Annealing (SA) for the 0/1 knapsack problem.

    Encoding        : binary vector x = (x_1, ..., x_n) where x_i = 1 if item i is selected, else 0
    Objective       : maximise f(x) = sum p_i x_i
    Constraint      : sum w_i x_i <= C  (kept feasible at all times by repair)
    Initial solution: random feasible solution (helpers.random_feasible_solution)
    Neighbourhood   : one operator from neighborhoods.py (Flip / Swap)
    Acceptance      : accept if delta >= 0
                      else accept with probability exp(delta / T)
    Cooling         : geometric  T_{k+1} = alpha * T_k
    T0              : automatically calculated from the average loss of 200 random neighbors
    Stopping rule   : fixed number of iterations

"""

import math

from helpers import random_feasible_solution, total_value, total_weight, make_rng, Timer


def estimate_initial_temperature(state, inst, neighbor, rng, p0=0.8, samples=200):
    """
    Automatically calculate the starting temperature for SA.

    Generate 200 neighboring solutions and measure how much worse
    they are compared to the initial solution.

    Use the average loss to calculate T0, so that a move with this
    loss has an 80% chance of being accepted at the start.

    Formula:  p0 = exp(-mean_loss / T0)   ->   T0 = -mean_loss / ln(p0)
    """
    losses = []
    # sample a number of neighbors and record the losses
    for _ in range(samples):
        _, _, v_new = neighbor(state, inst, rng)
        delta = v_new - state[2]
        if delta < 0:
            losses.append(-delta)
    # compute the mean loss and derive T0 from it
    mean_loss = sum(losses) / len(losses) if losses else 1.0
    return -mean_loss / math.log(p0)


def simulated_annealing(inst, neighbor, seed, max_iters=50_000,
                        p0=0.8, t_final=0.01, history_every=500):
    """Run one SA run.

    Parameters
    ----------
    inst          : KnapsackInstance
    neighbor      : neighbourhood operator function
    seed          : random seed
    max_iters     : stopping criteria (number of iterations)
    p0            : initial acceptance probability for worse moves 
    t_final       : final temperature
    history_every : record the best value every this many iterations

    Returns
    -------
    dictionary with best solution, best value, runtime and convergence history
    """
    rng = make_rng(seed)

    with Timer() as timer:
        # ---- initial solution -----------------------------------------
        x = random_feasible_solution(inst, rng)
        # state = (solution, weight, value)
        current = (x, total_weight(x, inst), total_value(x, inst))
        best = current

        # ---- temperature schedule --------------------------------------
        # calculate T0 
        T0 = estimate_initial_temperature(current, inst, neighbor, rng, p0)
        # calculate alpha so that the temperature reaches t_final after max_iters iterations
        alpha = (t_final / T0) ** (1.0 / max_iters) 
        T = T0
        # record the intial solution for the convergence history
        history = [(0, best[2])]
        # count how many moves were accepted to calculate the acceptance rate
        accepted = 0

        # ---- main loop -------------------------------------------------
        # repeat the search until the stopping criteria is met
        for k in range(1, max_iters + 1):
            # generate a candidate using the selected neighborhood operator
            candidate = neighbor(current, inst, rng)
            # calculate the change in value between the candidate and the current solution
            delta = candidate[2] - current[2]
            # accept the candidate if it is better
            # worse candidates are accepted with a probability
            if delta >= 0 or rng.random() < math.exp(delta / T):
                current = candidate
                accepted += 1
                # update the best solution if the candidate is better than the best found so far
                if current[2] > best[2]:
                    best = current
            # reduce the temperature after each iteration 
            T *= alpha
            # record the best value for the convergence history
            if k % history_every == 0:
                history.append((k, best[2]))

    # extract the best solution and verify that it is feasible and consistent
    x_best, w_best, v_best = best
    assert v_best == total_value(x_best, inst)
    assert w_best == total_weight(x_best, inst) <= inst.capacity

    # return the results in a dictionary
    return {
        "solution": x_best,
        "value": v_best,
        "weight": w_best,
        "runtime": timer.elapsed,
        "history": history,
        "T0": T0,
        "alpha": alpha,
        "acceptance_rate": accepted / max_iters,
    }
