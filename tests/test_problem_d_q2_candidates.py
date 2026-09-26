from __future__ import annotations

from dataclasses import replace

import pytest

from math_model_cup.problem_d_q2 import InfeasibleQ2Error
from math_model_cup.problem_d_q2_candidates import (
    generate_initial_candidates,
    prune_dominated_candidates,
    solve_candidate_method,
)
from math_model_cup.problem_d_q2_schedule import schedule_trips_greedy
from math_model_cup.problem_d_q2_validation import validate_q2_solution

from test_problem_d_q2_schedule import schedule_case


def test_initial_candidates_cover_every_box(schedule_case) -> None:
    data, arcs, _ = schedule_case

    candidates = generate_initial_candidates(data, arcs, max_stops=2)

    covered = {box_id for candidate in candidates for box_id in candidate.box_ids}
    assert covered == set(data.boxes)
    assert len(prune_dominated_candidates(candidates)) <= len(candidates)


def test_candidate_solver_covers_each_box_once(schedule_case) -> None:
    data, arcs, _ = schedule_case

    solution = solve_candidate_method(data, arcs, time_limit_s=5)

    delivered = [record.box_id for record in solution.deliveries]
    assert sorted(delivered) == sorted(data.boxes)
    assert len(delivered) == len(set(delivered))
    assert validate_q2_solution(data, arcs, solution).is_valid
    assert solution.solver_status in {"FEASIBLE", "OPTIMAL"}
    assert not solution.diagnostics.get("fallback")


def test_candidate_solver_uses_provided_solution_as_native_incumbent(schedule_case) -> None:
    data, arcs, plans = schedule_case
    seed = schedule_trips_greedy(data, plans, method="grouped_seed")

    solution = solve_candidate_method(
        data,
        arcs,
        time_limit_s=5,
        candidates=(),
        initial_solution=seed,
    )

    assert solution.objective <= seed.objective
    assert solution.diagnostics["seed_objective"] == seed.objective
    assert solution.diagnostics["seed_candidate_count"] == len(seed.trips)
    assert solution.diagnostics["incumbent_source"] in {
        "candidate_search",
        "provided_seed",
    }


def test_candidate_solver_rejects_invalid_provided_solution(schedule_case) -> None:
    data, arcs, plans = schedule_case
    seed = schedule_trips_greedy(data, plans, method="grouped_seed")
    invalid_seed = replace(seed, deliveries=())

    with pytest.raises(InfeasibleQ2Error, match="invalid initial solution"):
        solve_candidate_method(data, arcs, time_limit_s=5, initial_solution=invalid_seed)
