from __future__ import annotations

import pytest
from dataclasses import replace

import math_model_cup.problem_d_q2_alns as alns_module
from math_model_cup.problem_d_q2 import InfeasibleQ2Error
from math_model_cup.problem_d_q2_alns import solve_alns
from math_model_cup.problem_d_q2_hybrid import solve_hybrid
from math_model_cup.problem_d_q2_milp import solve_integrated_milp
from math_model_cup.problem_d_q2_schedule import schedule_trips_greedy
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


def test_integrated_milp_keeps_seed_candidates_and_records_provenance(schedule_case) -> None:
    data, arcs, plans = schedule_case
    seed = schedule_trips_greedy(data, plans, method="grouped_seed")

    solution = solve_integrated_milp(
        data,
        arcs,
        time_limit_s=5,
        max_candidates=1,
        initial_solution=seed,
    )

    assert solution.objective <= seed.objective
    assert solution.diagnostics["seed_candidate_count"] == len(seed.trips)
    assert solution.diagnostics["seed_objective"] == seed.objective
    assert solution.diagnostics["lexicographic_phases_completed"] >= 1
    assert solution.diagnostics["incumbent_source"] in {
        "integrated_cp_sat",
        "provided_seed",
    }


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


def test_alns_records_whether_provided_seed_was_improved(schedule_case) -> None:
    data, arcs, plans = schedule_case
    seed = schedule_trips_greedy(data, plans, method="grouped_seed")

    solution = solve_alns(data, arcs, seed=7, iterations=10, initial_solution=seed)

    assert solution.objective <= seed.objective
    assert solution.diagnostics["seed_objective"] == seed.objective
    assert solution.diagnostics["incumbent_source"] in {"alns_search", "provided_seed"}
    assert solution.diagnostics["seed_retained"] is (
        solution.objective == seed.objective
    )


def test_alns_rejects_move_when_both_repairs_fail(schedule_case, monkeypatch) -> None:
    data, arcs, plans = schedule_case
    seed = schedule_trips_greedy(data, plans, method="grouped_seed")

    def fail(*args, **kwargs):
        raise InfeasibleQ2Error("forced repair failure")

    monkeypatch.setattr(alns_module, "_repair_plans", fail)
    monkeypatch.setattr(alns_module, "schedule_trips_greedy", fail)

    solution = solve_alns(data, arcs, seed=7, iterations=1, initial_solution=seed)

    assert solution.objective == seed.objective
    assert solution.diagnostics["feasible_moves"] == 0
    assert solution.diagnostics["rejected_infeasible_moves"] == 1


def test_alns_accepts_lexicographically_better_candidate_unconditionally(
    schedule_case,
) -> None:
    data, _, plans = schedule_case
    current = schedule_trips_greedy(data, plans, method="current")
    candidate = replace(
        current,
        objective=(
            current.objective[0] - 1e-6,
            current.objective[1] + 1e9,
            current.objective[2] + 1e9,
            current.objective[3] + 100,
        ),
    )

    assert alns_module._accept_candidate(current, candidate, 1e-12, lambda: 1.0)


def test_hybrid_returns_valid_solution_and_records_rounds(schedule_case) -> None:
    data, arcs, _ = schedule_case

    solution = solve_hybrid(data, arcs, seed=11, iterations=100, rounds=2)

    assert validate_q2_solution(data, arcs, solution).is_valid
    assert solution.method == "hybrid"
    assert solution.diagnostics["rounds_completed"] == 2
    assert solution.solver_status in {"FEASIBLE", "OPTIMAL"}
    assert not solution.diagnostics.get("fallback")


def test_hybrid_passes_provided_seed_through_native_rounds(schedule_case) -> None:
    data, arcs, plans = schedule_case
    seed = schedule_trips_greedy(data, plans, method="grouped_seed")

    solution = solve_hybrid(
        data,
        arcs,
        seed=11,
        iterations=10,
        rounds=1,
        time_limit_s=5,
        initial_solution=seed,
    )

    assert solution.objective <= seed.objective
    assert solution.diagnostics["seed_objective"] == seed.objective
    assert solution.diagnostics["incumbent_source"] in {"hybrid_search", "provided_seed"}
