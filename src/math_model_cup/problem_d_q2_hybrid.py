"""Minimum viable candidate-pool and ALNS hybrid for Q2."""

from __future__ import annotations

from dataclasses import replace
from time import perf_counter
from typing import Dict, Mapping, Tuple

from .problem_d_q2 import Q2Data, Q2Solution, TripPlan
from .problem_d_q2_alns import solve_alns
from .problem_d_q2_candidates import generate_initial_candidates, solve_candidate_method
from .problem_d_q2_geometry import ArcGeometry
from .problem_d_q2_incumbent import (
    ensure_valid_initial_solution,
    finalize_seeded_solution,
    solution_key,
)
from .problem_d_q2_urgent import PUBLISHED_MODE


def solve_hybrid(
    data: Q2Data,
    arcs: Mapping[Tuple[str, str], ArcGeometry],
    seed: int = 0,
    iterations: int = 10_000,
    rounds: int = 2,
    time_limit_s: float = 30.0,
    *,
    initial_solution: Q2Solution | None = None,
    evaluation_mode: str = PUBLISHED_MODE,
) -> Q2Solution:
    started = perf_counter()
    ensure_valid_initial_solution(
        data, arcs, initial_solution, evaluation_mode=evaluation_mode
    )
    if rounds < 1:
        raise ValueError("rounds must be at least one")
    pool: Dict[Tuple[object, ...], TripPlan] = {
        plan.signature: plan for plan in generate_initial_candidates(data, arcs)
    }
    if initial_solution is not None:
        for trip in initial_solution.trips:
            pool[trip.plan.signature] = trip.plan
    current = solve_candidate_method(
        data,
        arcs,
        time_limit_s=time_limit_s,
        candidates=tuple(pool.values()),
        initial_solution=initial_solution,
        evaluation_mode=evaluation_mode,
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
            evaluation_mode=evaluation_mode,
        )
        for trip in improved.trips:
            pool[trip.plan.signature] = trip.plan
        refreshed = solve_candidate_method(
            data,
            arcs,
            time_limit_s=time_limit_s,
            candidates=tuple(pool.values()),
            initial_solution=current,
            evaluation_mode=evaluation_mode,
        )
        current = min(
            (current, improved, refreshed),
            key=lambda solution: solution_key(
                solution, data=data, evaluation_mode=evaluation_mode
            ),
        )
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
            "evaluation_mode": evaluation_mode,
        }
    )
    return finalize_seeded_solution(
        "hybrid",
        current,
        initial_solution,
        started=started,
        native_source="hybrid_search",
        diagnostics=diagnostics,
        data=data,
        evaluation_mode=evaluation_mode,
    )
