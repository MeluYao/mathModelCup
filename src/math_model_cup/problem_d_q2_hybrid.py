"""Minimum viable candidate-pool and ALNS hybrid for Q2."""

from __future__ import annotations

from dataclasses import replace
from time import perf_counter
from typing import Dict, Mapping, Tuple

from .problem_d_q2 import Q2Data, Q2Solution, TripPlan
from .problem_d_q2_alns import solve_alns
from .problem_d_q2_candidates import generate_initial_candidates, solve_candidate_method
from .problem_d_q2_geometry import ArcGeometry


def solve_hybrid(
    data: Q2Data,
    arcs: Mapping[Tuple[str, str], ArcGeometry],
    seed: int = 0,
    iterations: int = 10_000,
    rounds: int = 2,
    time_limit_s: float = 30.0,
) -> Q2Solution:
    started = perf_counter()
    if rounds < 1:
        raise ValueError("rounds must be at least one")
    pool: Dict[Tuple[object, ...], TripPlan] = {
        plan.signature: plan for plan in generate_initial_candidates(data, arcs)
    }
    current = solve_candidate_method(
        data, arcs, time_limit_s=time_limit_s, candidates=tuple(pool.values())
    )
    objective_history = [current.objective]
    for round_index in range(rounds):
        improved = solve_alns(
            data,
            arcs,
            seed=seed + round_index,
            iterations=iterations,
            initial_solution=current,
            candidate_pool=tuple(pool.values()),
        )
        for trip in improved.trips:
            pool[trip.plan.signature] = trip.plan
        refreshed = solve_candidate_method(
            data,
            arcs,
            time_limit_s=time_limit_s,
            candidates=tuple(pool.values()),
        )
        current = min((current, improved, refreshed), key=lambda solution: solution.objective)
        objective_history.append(current.objective)

    diagnostics = dict(current.diagnostics)
    diagnostics.update(
        {
            "rounds_completed": rounds,
            "seed": seed,
            "iterations_per_round": iterations,
            "candidate_time_limit_s": time_limit_s,
            "final_candidate_count": len(pool),
            "objective_history": objective_history,
        }
    )
    return replace(
        current,
        method="hybrid",
        runtime_s=perf_counter() - started,
        diagnostics=diagnostics,
    )
