from __future__ import annotations

from dataclasses import replace

from math_model_cup.problem_d_q1 import Aircraft, Box, ProblemData, ServiceGeometry
from math_model_cup.problem_d_q1_analysis import (
    ParetoPoint,
    best_under_trip_budget,
    pareto_dynamic_programming,
    pareto_for_trip_count,
    pareto_solution,
    select_anchor,
)

from test_problem_d_q1 import make_aircraft, make_geometry


def make_tradeoff_problem() -> ProblemData:
    fast = replace(
        make_aircraft(model_id="F", max_payload_kg=10.0),
        cruise_speed_mps=30.0,
        usable_energy_kwh=10.0,
        empty_range_m=10_000.0,
        full_range_m=8_000.0,
        preparation_time_s=30.0,
    )
    efficient = replace(
        make_aircraft(model_id="E", max_payload_kg=5.0),
        cruise_speed_mps=10.0,
        usable_energy_kwh=2.0,
        empty_range_m=100_000.0,
        full_range_m=80_000.0,
        preparation_time_s=30.0,
    )
    boxes = tuple(
        Box(
            box_id=f"S001-X-{index:02d}",
            service_id="S001",
            material_type="X",
            mass_kg=5.0,
            volume_m3=0.02,
        )
        for index in range(1, 3)
    )
    return ProblemData(
        aircraft={"E": efficient, "F": fast},
        boxes=boxes,
        geometries={"S001": make_geometry(distance_m=1_000.0)},
        material_types=("X",),
    )


def test_pareto_dp_retains_trip_time_energy_tradeoffs() -> None:
    data = make_tradeoff_problem()

    points = pareto_dynamic_programming(data)

    assert {point.trip_count for point in points} == {1, 2}
    fewest = select_anchor(points, "trips")
    lowest_energy = select_anchor(points, "energy")
    fastest = select_anchor(points, "time")
    assert fewest.trip_count == 1
    assert lowest_energy.trip_count == 2
    assert fastest.trip_count == 1
    assert lowest_energy.total_energy_kwh < fewest.total_energy_kwh


def test_fixed_minimum_trip_front_and_budget_selection() -> None:
    data = make_tradeoff_problem()
    points = pareto_dynamic_programming(data)
    minimum_trips = min(point.trip_count for point in points)

    fixed = pareto_for_trip_count(points, minimum_trips)
    budget = best_under_trip_budget(points, minimum_trips + 1, "energy")
    solution = pareto_solution(data, budget, method="pareto_energy")

    assert fixed
    assert all(point.trip_count == minimum_trips for point in fixed)
    assert budget.trip_count == 2
    assert solution.trip_count == 2
    assert len({box_id for trip in solution.trips for box_id in trip.box_ids}) == 2


def test_dominated_points_are_not_returned() -> None:
    data = make_tradeoff_problem()
    points = pareto_dynamic_programming(data)

    for left in points:
        for right in points:
            if left is right:
                continue
            assert not (
                right.trip_count <= left.trip_count
                and right.total_time_s <= left.total_time_s
                and right.total_energy_kwh <= left.total_energy_kwh
            )
