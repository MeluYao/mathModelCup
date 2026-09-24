from __future__ import annotations

import os
from pathlib import Path

import pytest

from math_model_cup.problem_d_q1 import (
    Aircraft,
    Box,
    ProblemData,
    ServiceGeometry,
    generate_candidates,
    safe_payload,
    solve_dynamic_programming,
    solve_greedy,
    solve_milp,
    trip_energy,
    validate_solution,
)


def make_aircraft(
    *,
    model_id: str = "T",
    max_payload_kg: float = 10.0,
    volume_capacity_m3: float = 0.10,
) -> Aircraft:
    return Aircraft(
        model_id=model_id,
        name="test",
        empty_mass_kg=20.0,
        max_payload_kg=max_payload_kg,
        volume_capacity_m3=volume_capacity_m3,
        cruise_speed_mps=10.0,
        empty_range_m=100_000.0,
        full_range_m=80_000.0,
        usable_energy_kwh=10.0,
        default_reserve_ratio=0.20,
        preparation_time_s=60.0,
        load_time_per_box_s=5.0,
        handoff_base_time_s=20.0,
        handoff_time_per_box_s=5.0,
        climb_speed_mps=2.0,
        descent_speed_mps=2.0,
        climb_efficiency=0.72,
    )


def make_geometry(*, distance_m: float = 1000.0) -> ServiceGeometry:
    return ServiceGeometry(
        service_id="S001",
        distance_m=distance_m,
        peak_ground_m=120.0,
        cruise_altitude_m=170.0,
        center_climb_m=70.0,
        service_climb_m=40.0,
    )


def make_problem() -> ProblemData:
    aircraft = make_aircraft()
    boxes = tuple(
        Box(
            box_id=f"S001-X-{index:02d}",
            service_id="S001",
            material_type="X",
            mass_kg=5.0,
            volume_m3=0.02,
        )
        for index in range(1, 5)
    )
    return ProblemData(
        aircraft={aircraft.model_id: aircraft},
        boxes=boxes,
        geometries={"S001": make_geometry()},
        material_types=("X",),
    )


def test_safe_payload_returns_nominal_capacity_when_full_load_is_safe() -> None:
    aircraft = make_aircraft(max_payload_kg=25.0)
    geometry = make_geometry(distance_m=1000.0)

    assert safe_payload(aircraft, geometry, reserve_ratio=0.20) == 25.0


def test_trip_energy_increases_with_outbound_payload() -> None:
    aircraft = make_aircraft(max_payload_kg=25.0)
    geometry = make_geometry(distance_m=5000.0)

    assert trip_energy(aircraft, geometry, 20.0) > trip_energy(aircraft, geometry, 0.0)


def test_candidate_generation_enforces_mass_volume_and_reserve() -> None:
    data = make_problem()
    candidates = generate_candidates(data, "S001", reserve_ratio=0.20)

    assert candidates
    for candidate in candidates:
        aircraft = data.aircraft[candidate.model_id]
        assert candidate.mass_kg <= aircraft.max_payload_kg + 1e-9
        assert candidate.volume_m3 <= aircraft.volume_capacity_m3 + 1e-12
        assert candidate.return_soc_percent >= 20.0 - 1e-9


def test_all_three_solvers_return_valid_two_trip_solution() -> None:
    data = make_problem()
    solutions = (
        solve_greedy(data),
        solve_dynamic_programming(data),
        solve_milp(data),
    )

    for solution in solutions:
        validate_solution(data, solution.trips, reserve_ratio=0.20)
        assert solution.trip_count == 2
        assert len({box_id for trip in solution.trips for box_id in trip.box_ids}) == 4


def test_dp_and_milp_match_lexicographic_objective() -> None:
    data = make_problem()

    dynamic = solve_dynamic_programming(data)
    mixed_integer = solve_milp(data)

    assert dynamic.trip_count == mixed_integer.trip_count
    assert abs(dynamic.total_time_s - mixed_integer.total_time_s) < 1e-6
    assert abs(dynamic.total_energy_kwh - mixed_integer.total_energy_kwh) < 1e-6


@pytest.mark.skipif(
    not os.environ.get("D_PROBLEM_DIR"), reason="D_PROBLEM_DIR is not configured"
)
def test_real_data_all_methods_cover_80_boxes_and_exact_methods_match() -> None:
    from math_model_cup.problem_d_q1 import load_problem_data

    data = load_problem_data(Path(os.environ["D_PROBLEM_DIR"]))
    assert len(data.boxes) == 80

    greedy = solve_greedy(data)
    dynamic = solve_dynamic_programming(data)
    mixed_integer = solve_milp(data)

    for solution in (greedy, dynamic, mixed_integer):
        validate_solution(data, solution.trips)
        assert len({box_id for trip in solution.trips for box_id in trip.box_ids}) == 80
    assert dynamic.trip_count == mixed_integer.trip_count
    assert abs(dynamic.total_time_s - mixed_integer.total_time_s) < 1e-5
    assert abs(dynamic.total_energy_kwh - mixed_integer.total_energy_kwh) < 1e-5
