from __future__ import annotations

import pytest

from math_model_cup.problem_d_q2_alns import solve_alns
from math_model_cup.problem_d_q2_hybrid import solve_hybrid
from math_model_cup.problem_d_q2_milp import solve_integrated_milp
from math_model_cup.problem_d_q2_validation import validate_q2_solution

from test_problem_d_q2_schedule import schedule_case


def test_integrated_milp_returns_valid_solution(schedule_case) -> None:
    data, arcs, _ = schedule_case

    solution = solve_integrated_milp(data, arcs, time_limit_s=10, max_stops=3)

    assert validate_q2_solution(data, arcs, solution).is_valid
    assert solution.method == "integrated_milp"
    assert "backend" in solution.diagnostics
    assert solution.solver_status in {"FEASIBLE", "OPTIMAL"}
    assert not solution.diagnostics.get("fallback")


def test_alns_is_reproducible_and_valid(schedule_case) -> None:
    data, arcs, _ = schedule_case

    first = solve_alns(data, arcs, seed=7, iterations=100)
    second = solve_alns(data, arcs, seed=7, iterations=100)

    assert first.objective == pytest.approx(second.objective)
    assert validate_q2_solution(data, arcs, first).is_valid
    assert first.method == "alns"
    assert first.solver_status in {"FEASIBLE", "OPTIMAL"}
    assert not first.diagnostics.get("fallback")
    assert first.diagnostics["initialization"] == "independent_resource_aware"


def test_hybrid_returns_valid_solution_and_records_rounds(schedule_case) -> None:
    data, arcs, _ = schedule_case

    solution = solve_hybrid(data, arcs, seed=11, iterations=100, rounds=2)

    assert validate_q2_solution(data, arcs, solution).is_valid
    assert solution.method == "hybrid"
    assert solution.diagnostics["rounds_completed"] == 2
    assert solution.solver_status in {"FEASIBLE", "OPTIMAL"}
    assert not solution.diagnostics.get("fallback")
