from __future__ import annotations

import math

from math_model_cup.problem_d_q1 import (
    GRAVITY_MPS2,
    JOULES_PER_KWH,
    generate_candidates,
    solve_dynamic_programming,
    trip_energy,
    trip_energy_breakdown,
)

from test_problem_d_q1 import make_aircraft, make_geometry, make_problem


def test_energy_breakdown_uses_loaded_outbound_and_empty_return() -> None:
    aircraft = make_aircraft(max_payload_kg=25.0)
    geometry = make_geometry(distance_m=5_000.0)
    payload = 20.0

    result = trip_energy_breakdown(aircraft, geometry, payload)

    expected_outbound_climb = (
        (aircraft.empty_mass_kg + payload)
        * GRAVITY_MPS2
        * geometry.center_climb_m
        / (JOULES_PER_KWH * aircraft.climb_efficiency)
    )
    expected_return_climb = (
        aircraft.empty_mass_kg
        * GRAVITY_MPS2
        * geometry.service_climb_m
        / (JOULES_PER_KWH * aircraft.climb_efficiency)
    )
    assert math.isclose(result.outbound_climb_kwh, expected_outbound_climb)
    assert math.isclose(result.return_climb_kwh, expected_return_climb)
    assert result.return_payload_kg == 0.0
    assert result.descent_energy_kwh == 0.0
    assert math.isclose(result.total_kwh, trip_energy(aircraft, geometry, payload))


def test_alternative_range_calibration_scales_horizontal_energy_only() -> None:
    aircraft = make_aircraft(max_payload_kg=25.0)
    geometry = make_geometry(distance_m=5_000.0)

    primary = trip_energy_breakdown(aircraft, geometry, 20.0)
    alternative = trip_energy_breakdown(
        aircraft, geometry, 20.0, range_energy_fraction=0.8
    )

    assert math.isclose(
        alternative.horizontal_kwh, 0.8 * primary.horizontal_kwh, rel_tol=1e-12
    )
    assert math.isclose(alternative.climb_kwh, primary.climb_kwh, rel_tol=1e-12)
    assert alternative.total_kwh < primary.total_kwh


def test_invalid_range_energy_fraction_is_rejected() -> None:
    aircraft = make_aircraft()
    geometry = make_geometry()

    for fraction in (0.0, -0.1, 1.1):
        try:
            trip_energy_breakdown(
                aircraft, geometry, 5.0, range_energy_fraction=fraction
            )
        except ValueError:
            pass
        else:
            raise AssertionError(f"range_energy_fraction={fraction} should fail")


def test_range_calibration_propagates_through_candidates_and_solver() -> None:
    data = make_problem()
    primary_candidates = generate_candidates(data, "S001", range_energy_fraction=1.0)
    alternative_candidates = generate_candidates(data, "S001", range_energy_fraction=0.8)
    primary = solve_dynamic_programming(data, range_energy_fraction=1.0)
    alternative = solve_dynamic_programming(data, range_energy_fraction=0.8)

    primary_energy = {
        (candidate.model_id, candidate.counts): candidate.energy_kwh
        for candidate in primary_candidates
    }
    alternative_energy = {
        (candidate.model_id, candidate.counts): candidate.energy_kwh
        for candidate in alternative_candidates
    }
    common = set(primary_energy) & set(alternative_energy)
    assert common
    assert all(alternative_energy[key] < primary_energy[key] for key in common)
    assert alternative.total_energy_kwh < primary.total_energy_kwh
