from __future__ import annotations

from pathlib import Path

from math_model_cup.problem_d_q2 import load_q2_data
from math_model_cup.problem_d_q2_geometry import build_arc_matrix
from math_model_cup.problem_d_q2_grouped import solve_grouped_local_search
from math_model_cup.problem_d_q2_hybrid import solve_hybrid
from math_model_cup.problem_d_q2_milp import solve_integrated_milp
from math_model_cup.problem_d_q2_urgent import URGENT_PRIORITY_MODE, all_box_lateness
from math_model_cup.problem_d_q2_urgent import urgent_objective_vector
from math_model_cup.problem_d_q2_validation import validate_q2_solution


def test_grouped_seed_delivers_all_real_boxes_on_time() -> None:
    data = load_q2_data(Path("D题"))
    arcs = build_arc_matrix(data)

    solution = solve_grouped_local_search(
        data,
        arcs,
        seed=20260924,
        iterations=20,
        evaluation_mode=URGENT_PRIORITY_MODE,
    )

    report = validate_q2_solution(
        data,
        arcs,
        solution,
        evaluation_mode=URGENT_PRIORITY_MODE,
    )
    assert report.is_valid
    assert all_box_lateness(data, solution) == (0, 0.0)


def test_hybrid_uses_urgent_lexicographic_objective() -> None:
    data = load_q2_data(Path("D题"))
    arcs = build_arc_matrix(data)
    seed_solution = solve_grouped_local_search(
        data,
        arcs,
        seed=20260924,
        iterations=10,
        evaluation_mode=URGENT_PRIORITY_MODE,
    )

    solution = solve_hybrid(
        data,
        arcs,
        seed=20260924,
        iterations=2,
        rounds=1,
        time_limit_s=2.0,
        initial_solution=seed_solution,
        evaluation_mode=URGENT_PRIORITY_MODE,
    )

    assert validate_q2_solution(
        data,
        arcs,
        solution,
        evaluation_mode=URGENT_PRIORITY_MODE,
    ).is_valid
    assert solution.objective == urgent_objective_vector(data, solution)


def test_cpsat_uses_all_deadlines_and_urgent_objective() -> None:
    data = load_q2_data(Path("D题"))
    arcs = build_arc_matrix(data)
    seed_solution = solve_grouped_local_search(
        data,
        arcs,
        seed=20260924,
        iterations=10,
        evaluation_mode=URGENT_PRIORITY_MODE,
    )

    solution = solve_integrated_milp(
        data,
        arcs,
        time_limit_s=2.0,
        max_candidates=200,
        initial_solution=seed_solution,
        evaluation_mode=URGENT_PRIORITY_MODE,
    )

    assert validate_q2_solution(
        data,
        arcs,
        solution,
        evaluation_mode=URGENT_PRIORITY_MODE,
    ).is_valid
    assert solution.objective == urgent_objective_vector(data, solution)
