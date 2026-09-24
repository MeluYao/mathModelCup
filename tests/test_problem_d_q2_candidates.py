from __future__ import annotations

from math_model_cup.problem_d_q2_candidates import (
    generate_initial_candidates,
    prune_dominated_candidates,
    solve_candidate_method,
)
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
