from __future__ import annotations

from dataclasses import replace

from math_model_cup.problem_d_q2_schedule import schedule_trips_greedy
from math_model_cup.problem_d_q2_validation import validate_q2_solution

from test_problem_d_q2_schedule import schedule_case


def test_validator_accepts_recomputed_valid_solution(schedule_case) -> None:
    data, arcs, plans = schedule_case
    solution = schedule_trips_greedy(data, plans, method="valid")

    report = validate_q2_solution(data, arcs, solution)

    assert report.is_valid
    assert report.issues == tuple()


def test_validator_rejects_duplicate_box_delivery(schedule_case) -> None:
    data, arcs, plans = schedule_case
    solution = schedule_trips_greedy(data, plans, method="valid")
    broken = replace(
        solution,
        deliveries=solution.deliveries + (solution.deliveries[0],),
    )

    report = validate_q2_solution(data, arcs, broken)

    assert not report.is_valid
    assert "duplicate_box" in {issue.code for issue in report.issues}
