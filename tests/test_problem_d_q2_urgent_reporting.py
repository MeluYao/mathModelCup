from __future__ import annotations

import pandas as pd

from math_model_cup.problem_d_q2_urgent_reporting import (
    run_hybrid_repetitions,
    run_urgent_methods,
)

from test_problem_d_q2_schedule import schedule_case


def test_method_table_contains_only_cpsat_and_hybrid(tmp_path, schedule_case) -> None:
    data, arcs, _ = schedule_case

    solutions = run_urgent_methods(
        data,
        arcs,
        tmp_path,
        seed=20260924,
        iterations=1,
        time_limit_s=1.0,
    )

    rows = pd.read_csv(tmp_path / "method_comparison.csv")
    assert set(solutions) == {"integrated_milp", "hybrid"}
    assert set(rows["method"]) == {"integrated_milp", "hybrid"}
    assert (rows["late_box_count"] == 0).all()
    assert {
        "urgent_weighted_delivery",
        "energy_kwh",
        "trip_count",
        "makespan_s",
        "minimum_return_soc_percent",
    } <= set(rows.columns)

    repetitions = {}
    frame = run_hybrid_repetitions(
        data,
        arcs,
        solutions["hybrid"],
        seeds=(1, 2),
        iterations=0,
        time_limit_s=1.0,
        solutions_out=repetitions,
    )
    assert set(repetitions) == {1, 2}
    assert len(frame) == 2
