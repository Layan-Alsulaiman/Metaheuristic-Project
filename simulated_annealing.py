"""
Algorithm A: Simulated Annealing (SA) for the 0/1 knapsack problem.

Design summary
    Encoding        : binary vector x in {0,1}^n
    Objective       : maximise f(x) = sum p_i x_i
    Constraint      : sum w_i x_i <= C  (kept feasible at all times by repair)
    Initial solution: random feasible solution (helpers.random_feasible_solution)
    Neighbourhood   : one operator from neighborhoods.py (Flip / Swap)
    Acceptance      : Metropolis rule
                        accept if delta >= 0
                        else accept with probability exp(delta / T)
    Cooling         : geometric  T_{k+1} = alpha * T_k
    T0              : automatic — chosen so that an average worsening move
                      is accepted with probability `p0` at the start
    Stopping rule   : fixed number of iterations (same budget for every run)

"""

import math

from helpers import random_feasible_solution, total_value, total_weight, make_rng, Timer


def estimate_initial_temperature(state, inst, neighbor, rng, p0=0.8, samples=200):
    """Pick T0 so that the mean worsening move is accepted with prob. p0.

    From  p0 = exp(-mean_loss / T0)   ->   T0 = -mean_loss / ln(p0)
    """
    losses = []
    for _ in range(samples):
        _, _, v_new = neighbor(state, inst, rng)
        delta = v_new - state[2]
        if delta < 0:
            losses.append(-delta)
    mean_loss = sum(losses) / len(losses) if losses else 1.0
    return -mean_loss / math.log(p0)


def simulated_annealing(inst, neighbor, seed, max_iters=50_000,
                        p0=0.8, t_final=0.01, history_every=500):
    """Run one independent SA run.

    Parameters
    ----------
    inst          : KnapsackInstance
    neighbor      : neighbourhood function (see neighborhoods.py)
    seed          : random seed (different per run -> independent runs)
    max_iters     : stopping rule (number of iterations)
    p0            : initial acceptance probability for a mean worsening move
    t_final       : final temperature; alpha is derived from T0, t_final, max_iters
    history_every : record best-so-far every this many iterations (for plots)

    Returns
    -------
    dict with best solution, best value, runtime and convergence history
    """
    rng = make_rng(seed)

    with Timer() as timer:
        # ---- initial solution -----------------------------------------
        x = random_feasible_solution(inst, rng)
        current = (x, total_weight(x, inst), total_value(x, inst))
        best = current

        # ---- temperature schedule --------------------------------------
        T0 = estimate_initial_temperature(current, inst, neighbor, rng, p0)
        alpha = (t_final / T0) ** (1.0 / max_iters)   # reaches t_final at the end
        T = T0

        history = [(0, best[2])]
        accepted = 0

        # ---- main loop -------------------------------------------------
        for k in range(1, max_iters + 1):
            candidate = neighbor(current, inst, rng)
            delta = candidate[2] - current[2]

            if delta >= 0 or rng.random() < math.exp(delta / T):
                current = candidate
                accepted += 1
                if current[2] > best[2]:
                    best = current

            T *= alpha
            if k % history_every == 0:
                history.append((k, best[2]))

    x_best, w_best, v_best = best
    # safety check: the cached numbers must be exact and feasible
    assert v_best == total_value(x_best, inst)
    assert w_best == total_weight(x_best, inst) <= inst.capacity

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
