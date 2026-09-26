from __future__ import annotations

from pathlib import Path

from math_model_cup.problem_d_q2 import load_q2_data
from math_model_cup.problem_d_q2_geometry import build_arc_matrix
from math_model_cup.problem_d_q2_grouped import solve_grouped_local_search
from math_model_cup.problem_d_q2_urgent import URGENT_PRIORITY_MODE, all_box_lateness
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
