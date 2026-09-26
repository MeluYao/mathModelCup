from __future__ import annotations

from dataclasses import replace

import math_model_cup.problem_d_q2_experiments as q2_experiments
from math_model_cup.problem_d_q2_candidates import generate_initial_candidates
from math_model_cup.problem_d_q2_experiments import (
    _save_figure,
    _recalibrate_solution,
    _sensitivity_result,
    create_paper_figures,
    run_alns_repetitions,
    run_sensitivity_analysis,
)
import matplotlib.pyplot as plt
from PIL import Image
from math_model_cup.problem_d_q2_schedule import construct_resource_aware_boxwise_solution
from math_model_cup.problem_d_q2_validation import validate_q2_solution

from test_problem_d_q2_schedule import schedule_case


def test_save_figure_retries_transient_windows_io_error(tmp_path, monkeypatch) -> None:
    figure = plt.figure()
    original_savefig = figure.savefig
    attempts = 0

    def flaky_savefig(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError(22, "transient file handle state")
        return original_savefig(*args, **kwargs)

    monkeypatch.setattr(figure, "savefig", flaky_savefig)

    paths = _save_figure(figure, tmp_path, "retry")

    assert attempts == 3
    assert all(path.exists() and path.stat().st_size > 0 for path in paths)


def test_repeated_experiments_sensitivity_and_figures(
    tmp_path, schedule_case, monkeypatch
) -> None:
    data, arcs, _ = schedule_case
    pool = generate_initial_candidates(data, arcs, max_stops=2)
    baseline = construct_resource_aware_boxwise_solution(data, pool, "candidate")
    recalibrated_baseline = _recalibrate_solution(
        data,
        arcs,
        baseline,
        reserve_ratio=0.20,
        range_energy_fraction=0.8,
    )

    assert recalibrated_baseline.objective[1] < baseline.objective[1]

    repetitions, convergence = run_alns_repetitions(
        data,
        arcs,
        seeds=(3, 7),
        iterations=5,
        candidate_pool=pool,
    )
    sensitivity = run_sensitivity_analysis(data, arcs, pool)
    figure_dir = tmp_path / "figures"
    sensitivity_titles = []
    original_save_figure = q2_experiments._save_figure

    def capture_sensitivity_titles(fig, output_dir, stem, **kwargs):
        if stem == "fig05_sensitivity":
            sensitivity_titles.extend(
                axis.get_title() for axis in fig.axes if axis.get_visible()
            )
        return original_save_figure(fig, output_dir, stem, **kwargs)

    monkeypatch.setattr(q2_experiments, "_save_figure", capture_sensitivity_titles)
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
    assert repetitions["incumbent_source"].isin(
        {"provided_seed", "alns_search"}
    ).all()
    assert not convergence.empty
    assert set(convergence.groupby("seed")["iteration"].max()) == {5}
    assert set(sensitivity["factor"]) == {
        "reserve_ratio",
        "charge_time_multiplier",
        "resource_fraction",
        "deadline_multiplier",
        "range_energy_fraction",
    }
    energy_calibration = sensitivity[
        sensitivity["factor"] == "range_energy_fraction"
    ].sort_values("level")
    assert set(energy_calibration["level"]) == {0.8, 1.0}
    assert energy_calibration["feasible"].all()
    assert set(energy_calibration["status"]) == {"FEASIBLE"}
    assert energy_calibration.iloc[0]["objective_primary"] <= energy_calibration.iloc[1]["objective_primary"]
    assert energy_calibration.iloc[0]["energy_kwh"] < energy_calibration.iloc[1]["energy_kwh"]
    assert sensitivity[
        sensitivity["factor"] != "range_energy_fraction"
    ].groupby("factor").size().min() >= 3
    reserve_counts = sensitivity[
        sensitivity["factor"] == "reserve_ratio"
    ].sort_values("level")["reference_candidate_count"]
    assert reserve_counts.is_monotonic_decreasing
    assert len(created) == 14
    assert all(path.exists() and path.stat().st_size > 0 for path in created)
    with Image.open(figure_dir / "fig05_sensitivity.png") as sensitivity_figure:
        assert sensitivity_figure.height > sensitivity_figure.width
    assert set(sensitivity_titles) == {
        "Charge Time Multiplier",
        "Deadline Multiplier",
        "Range Energy Fraction",
        "Reserve Ratio",
        "Resource Fraction",
    }


def test_sensitivity_reports_search_failure_without_claiming_infeasibility(
    schedule_case,
) -> None:
    data, arcs, plans = schedule_case
    no_aircraft = replace(data, aircraft_units=tuple())

    result = _sensitivity_result(
        "resource_fraction",
        0.0,
        no_aircraft,
        arcs,
        plans,
    )

    assert not result["feasible"]
    assert result["status"] == "NO_FEASIBLE_SOLUTION_FOUND"
    assert result["error"]

    solver_error = _sensitivity_result(
        "range_energy_fraction",
        0.0,
        data,
        arcs,
        plans,
        range_energy_fraction=0.0,
    )
    assert not solver_error["feasible"]
    assert solver_error["status"] == "SOLVER_ERROR"


def test_recalibration_rejects_an_invalid_better_reschedule(
    schedule_case,
    monkeypatch,
) -> None:
    data, arcs, plans = schedule_case
    baseline = construct_resource_aware_boxwise_solution(data, plans, "candidate")
    invalid = replace(
        baseline,
        deliveries=baseline.deliveries + (baseline.deliveries[0],),
        objective=(0.0, 0.0, 0.0, 0),
    )
    monkeypatch.setattr(q2_experiments, "schedule_trips_greedy", lambda *args, **kwargs: invalid)

    recalibrated = _recalibrate_solution(
        data,
        arcs,
        baseline,
        reserve_ratio=0.20,
        range_energy_fraction=0.8,
    )

    assert validate_q2_solution(
        data,
        arcs,
        recalibrated,
        range_energy_fraction=0.8,
    ).is_valid
