from __future__ import annotations

from dataclasses import replace

from math_model_cup.problem_d_q1 import MethodSolution, solve_dynamic_programming
from math_model_cup.problem_d_q1_validation import (
    ENERGY_TOLERANCE_KWH,
    audit_solution_independently,
    independent_trip_breakdown,
)

from test_problem_d_q1 import make_problem


def test_independent_trip_recalculation_matches_solution_values() -> None:
    data = make_problem()
    solution = solve_dynamic_programming(data)

    audit = independent_trip_breakdown(data, solution.trips[0])

    assert abs(audit.total_energy_kwh - solution.trips[0].energy_kwh) <= ENERGY_TOLERANCE_KWH
    assert abs(audit.total_time_s - solution.trips[0].time_s) <= 1e-6
    assert audit.return_payload_kg == 0.0
    assert audit.descent_energy_kwh == 0.0
    assert audit.constraints_pass


def test_independent_solution_audit_checks_coverage_constraints_and_sums() -> None:
    data = make_problem()
    solution = solve_dynamic_programming(data)

    result = audit_solution_independently(data, solution)

    assert result.all_passed
    assert result.unique_box_count == len(data.boxes)
    assert result.total_assignment_count == len(data.boxes)
    assert abs(result.recomputed_energy_kwh - solution.total_energy_kwh) <= ENERGY_TOLERANCE_KWH
    assert abs(result.recomputed_time_s - solution.total_time_s) <= 1e-6


def test_independent_audit_detects_duplicate_and_missing_box() -> None:
    data = make_problem()
    solution = solve_dynamic_programming(data)
    first, second = solution.trips
    corrupted_second = replace(second, box_ids=(first.box_ids[0],) + second.box_ids[1:])
    corrupted = MethodSolution(
        method="corrupted",
        trips=(first, corrupted_second),
        runtime_s=0.0,
    )

    result = audit_solution_independently(data, corrupted)

    assert not result.all_passed
    assert not result.coverage_pass
    assert result.total_assignment_count == len(data.boxes)
    assert result.unique_box_count < len(data.boxes)
