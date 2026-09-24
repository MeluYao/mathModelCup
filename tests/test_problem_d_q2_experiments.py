from __future__ import annotations

from math_model_cup.problem_d_q2_candidates import generate_initial_candidates
from math_model_cup.problem_d_q2_experiments import (
    create_paper_figures,
    run_alns_repetitions,
    run_sensitivity_analysis,
)
from math_model_cup.problem_d_q2_schedule import construct_resource_aware_boxwise_solution

from test_problem_d_q2_schedule import schedule_case


def test_repeated_experiments_sensitivity_and_figures(tmp_path, schedule_case) -> None:
    data, arcs, _ = schedule_case
    pool = generate_initial_candidates(data, arcs, max_stops=2)
    baseline = construct_resource_aware_boxwise_solution(data, pool, "candidate")

    repetitions, convergence = run_alns_repetitions(
        data,
        arcs,
        seeds=(3, 7),
        iterations=5,
        candidate_pool=pool,
    )
    sensitivity = run_sensitivity_analysis(data, arcs, pool)
    figure_dir = tmp_path / "figures"
    created = create_paper_figures(
        figure_dir,
        data,
        {"candidate": baseline},
        repetitions,
        convergence,
        sensitivity,
    )

    assert set(repetitions["seed"]) == {3, 7}
    assert (repetitions["validation"] == "PASS").all()
    assert not convergence.empty
    assert set(convergence.groupby("seed")["iteration"].max()) == {5}
    assert set(sensitivity["factor"]) == {
        "reserve_ratio",
        "charge_time_multiplier",
        "resource_fraction",
        "deadline_multiplier",
    }
    assert sensitivity.groupby("factor").size().min() >= 3
    reserve_counts = sensitivity[
        sensitivity["factor"] == "reserve_ratio"
    ].sort_values("level")["candidate_count"]
    assert reserve_counts.is_monotonic_decreasing
    assert len(created) == 14
    assert all(path.exists() and path.stat().st_size > 0 for path in created)
