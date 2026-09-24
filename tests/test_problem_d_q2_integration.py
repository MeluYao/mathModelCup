from __future__ import annotations

import json

import pandas as pd

from math_model_cup.problem_d_q2_candidates import solve_candidate_method
from math_model_cup.problem_d_q2_reporting import write_q2_outputs

from test_problem_d_q2_schedule import schedule_case


def test_reporting_writes_submission_source_tables(tmp_path, schedule_case) -> None:
    data, arcs, _ = schedule_case
    solution = solve_candidate_method(data, arcs, time_limit_s=5)

    write_q2_outputs(tmp_path, data, arcs, {solution.method: solution})

    method_dir = tmp_path / solution.method
    expected_method_files = {
        "trips.csv",
        "deliveries.csv",
        "aircraft_timeline.csv",
        "battery_timeline.csv",
        "summary.json",
        "validation.csv",
        "constraint_check.csv",
    }
    assert expected_method_files == {path.name for path in method_dir.iterdir()}
    assert (tmp_path / "method_comparison.csv").exists()
    assert (tmp_path / "summary.md").exists()

    comparison = pd.read_csv(tmp_path / "method_comparison.csv")
    assert comparison.loc[0, "validation"] == "PASS"
    assert comparison.loc[0, "delivered_box_count"] == len(data.boxes)
    summary = json.loads((method_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["validation"] == "PASS"
    assert summary["objective"]["trip_count"] == len(solution.trips)
