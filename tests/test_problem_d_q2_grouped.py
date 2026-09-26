from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path

import pytest

from math_model_cup.problem_d_q2 import load_q2_data
from math_model_cup.problem_d_q2_geometry import build_arc_matrix
from math_model_cup.problem_d_q2_grouped import (
    build_grouped_initial_plans,
    solve_grouped_local_search,
)
from math_model_cup.problem_d_q2_schedule import schedule_trips_greedy
from math_model_cup.problem_d_q2_validation import validate_q2_solution
from test_problem_d_q2_schedule import schedule_case


def test_grouped_solver_propagates_range_energy_calibration(schedule_case) -> None:
    data, original_arcs, _ = schedule_case
    arcs = dict(original_arcs)
    template = original_arcs[("O01", "S001")]
    arcs[("S001", "S002")] = replace(
        template,
        origin_id="S001",
        destination_id="S002",
    )
    arcs[("S002", "S001")] = replace(
        template,
        origin_id="S002",
        destination_id="S001",
    )

    baseline = solve_grouped_local_search(
        data,
        arcs,
        seed=7,
        iterations=0,
        objective_order="published",
    )
    alternative = solve_grouped_local_search(
        data,
        arcs,
        seed=7,
        iterations=0,
        objective_order="published",
        range_energy_fraction=0.8,
    )
    explicit_baseline = solve_grouped_local_search(
        data,
        arcs,
        seed=7,
        iterations=0,
        objective_order="published",
        range_energy_fraction=1.0,
    )

    assert alternative.diagnostics["range_energy_fraction"] == pytest.approx(0.8)
    assert alternative.objective[2] < baseline.objective[2]
    assert explicit_baseline.objective == pytest.approx(baseline.objective)
    assert validate_q2_solution(
        data,
        arcs,
        alternative,
        range_energy_fraction=0.8,
    ).is_valid


@pytest.mark.skipif(
    not os.environ.get("D_PROBLEM_DIR"), reason="D_PROBLEM_DIR is not configured"
)
def test_grouped_initial_plans_cover_real_data_with_feasible_multibox_trips() -> None:
    data = load_q2_data(Path(os.environ["D_PROBLEM_DIR"]))
    arcs = build_arc_matrix(data)

    plans = build_grouped_initial_plans(data, arcs)
    covered = [box_id for plan in plans for box_id in plan.box_ids]

    assert sorted(covered) == sorted(data.boxes)
    assert len(covered) == len(set(covered))
    assert len(plans) < len(data.boxes)
    assert any(len(plan.box_ids) > 1 for plan in plans)
    assert all(stop.box_ids for plan in plans for stop in plan.stops)

    solution = schedule_trips_greedy(data, plans, method="grouped_initial")
    assert validate_q2_solution(data, arcs, solution).is_valid


@pytest.mark.skipif(
    not os.environ.get("D_PROBLEM_DIR"), reason="D_PROBLEM_DIR is not configured"
)
def test_grouped_local_search_returns_compact_reproducible_valid_solution() -> None:
    data = load_q2_data(Path(os.environ["D_PROBLEM_DIR"]))
    arcs = build_arc_matrix(data)

    first = solve_grouped_local_search(
        data,
        arcs,
        seed=20260924,
        iterations=200,
        method="grouped_test",
    )
    second = solve_grouped_local_search(
        data,
        arcs,
        seed=20260924,
        iterations=200,
        method="grouped_test",
    )

    assert first.objective == pytest.approx(second.objective)
    assert len(first.trips) <= 22
    assert any(len(trip.plan.stops) > 1 for trip in first.trips)
    assert all(
        record.delivery_time_s <= data.boxes[record.box_id].expected_time_s + 1e-7
        for record in first.deliveries
    )
    assert all(stop.box_ids for trip in first.trips for stop in trip.plan.stops)
    assert validate_q2_solution(data, arcs, first).is_valid
    assert first.diagnostics["initial_trip_count"] < len(first.trips)
    assert first.diagnostics["accepted_repairs"] > 0
    assert first.diagnostics["accepted_local_moves"] > 0

    compact = solve_grouped_local_search(
        data,
        arcs,
        seed=20260924,
        iterations=200,
        method="grouped_compact_test",
        objective_order="trip_count",
    )
    assert len(compact.trips) <= 15
    assert any(len(trip.plan.stops) > 1 for trip in compact.trips)
    assert validate_q2_solution(data, arcs, compact).is_valid
    assert compact.diagnostics["objective_order"] == "trip_count"
