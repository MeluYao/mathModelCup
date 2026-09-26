from __future__ import annotations

from dataclasses import replace

from math_model_cup.problem_d_q2 import TripDraft
from math_model_cup.problem_d_q2_physics import evaluate_trip
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


def test_validator_recomputes_alternative_range_energy_calibration(schedule_case) -> None:
    data, arcs, plans = schedule_case
    explicit_baseline_plans = tuple(
        evaluate_trip(
            data,
            arcs,
            TripDraft(plan.model_id, plan.stops),
            range_energy_fraction=1.0,
        )
        for plan in plans
    )
    alternative_plans = tuple(
        evaluate_trip(
            data,
            arcs,
            TripDraft(plan.model_id, plan.stops),
            range_energy_fraction=0.8,
        )
        for plan in plans
    )
    solution = schedule_trips_greedy(data, alternative_plans, method="alternative")

    default_report = validate_q2_solution(data, arcs, solution)
    alternative_report = validate_q2_solution(
        data,
        arcs,
        solution,
        range_energy_fraction=0.8,
    )

    assert "energy_mismatch" in {issue.code for issue in default_report.issues}
    assert alternative_report.is_valid
    assert [plan.energy_kwh for plan in explicit_baseline_plans] == [
        plan.energy_kwh for plan in plans
    ]
    assert [plan.duration_s for plan in alternative_plans] == [
        plan.duration_s for plan in plans
    ]
