"""Independent physical and accounting validation for Problem D, question 1.

This module deliberately re-implements the physical equations instead of
calling the candidate generator's energy and time functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Tuple

from .problem_d_q1 import MethodSolution, ProblemData, TripResult


MASS_TOLERANCE_KG = 1e-7
VOLUME_TOLERANCE_M3 = 1e-9
ENERGY_TOLERANCE_KWH = 1e-8
TIME_TOLERANCE_S = 1e-6
SOC_TOLERANCE_PERCENT = 1e-6
_GRAVITY_MPS2 = 9.81
_JOULES_PER_KWH = 3_600_000.0


@dataclass(frozen=True)
class IndependentTripAudit:
    trip_id: str
    service_id: str
    model_id: str
    box_count: int
    mass_kg: float
    volume_m3: float
    outbound_payload_kg: float
    return_payload_kg: float
    outbound_equivalent_range_m: float
    return_equivalent_range_m: float
    outbound_horizontal_kwh: float
    return_horizontal_kwh: float
    outbound_climb_kwh: float
    return_climb_kwh: float
    descent_energy_kwh: float
    total_energy_kwh: float
    flight_time_s: float
    ground_time_s: float
    total_time_s: float
    return_soc_percent: float
    mass_error_kg: float
    volume_error_m3: float
    energy_error_kwh: float
    time_error_s: float
    soc_error_percent: float
    same_service_pass: bool
    mass_capacity_pass: bool
    volume_capacity_pass: bool
    reserve_pass: bool
    constraints_pass: bool


@dataclass(frozen=True)
class IndependentSolutionAudit:
    trip_audits: Tuple[IndependentTripAudit, ...]
    total_assignment_count: int
    unique_box_count: int
    expected_box_count: int
    coverage_pass: bool
    constraints_pass: bool
    recomputed_energy_kwh: float
    reported_energy_kwh: float
    energy_sum_pass: bool
    recomputed_time_s: float
    reported_time_s: float
    time_sum_pass: bool
    all_passed: bool


def _independent_equivalent_range_m(aircraft, payload_kg: float) -> float:
    load_fraction = payload_kg / aircraft.max_payload_kg
    return aircraft.empty_range_m - (
        aircraft.empty_range_m - aircraft.full_range_m
    ) * load_fraction**1.5


def independent_trip_breakdown(
    data: ProblemData,
    trip: TripResult,
    reserve_ratio: float = 0.20,
    *,
    range_energy_fraction: float = 1.0,
) -> IndependentTripAudit:
    boxes_by_id = {box.box_id: box for box in data.boxes}
    boxes = tuple(boxes_by_id[box_id] for box_id in trip.box_ids)
    aircraft = data.aircraft[trip.model_id]
    geometry = data.geometries[trip.service_id]
    mass_kg = sum(box.mass_kg for box in boxes)
    volume_m3 = sum(box.volume_m3 for box in boxes)
    box_count = len(boxes)

    outbound_range = _independent_equivalent_range_m(aircraft, mass_kg)
    return_range = _independent_equivalent_range_m(aircraft, 0.0)
    outbound_horizontal = (
        range_energy_fraction
        * aircraft.usable_energy_kwh
        * geometry.distance_m
        / outbound_range
    )
    return_horizontal = (
        range_energy_fraction
        * aircraft.usable_energy_kwh
        * geometry.distance_m
        / return_range
    )
    outbound_climb = (
        (aircraft.empty_mass_kg + mass_kg)
        * _GRAVITY_MPS2
        * geometry.center_climb_m
        / (_JOULES_PER_KWH * aircraft.climb_efficiency)
    )
    return_climb = (
        aircraft.empty_mass_kg
        * _GRAVITY_MPS2
        * geometry.return_climb_m
        / (_JOULES_PER_KWH * aircraft.climb_efficiency)
    )
    descent_energy = 0.0
    total_energy = (
        outbound_horizontal
        + return_horizontal
        + outbound_climb
        + return_climb
        + descent_energy
    )

    flight_time = (
        geometry.center_climb_m / aircraft.climb_speed_mps
        + geometry.distance_m / aircraft.cruise_speed_mps
        + geometry.outbound_descent_m / aircraft.descent_speed_mps
        + geometry.return_climb_m / aircraft.climb_speed_mps
        + geometry.distance_m / aircraft.cruise_speed_mps
        + geometry.return_descent_m / aircraft.descent_speed_mps
    )
    ground_time = (
        aircraft.preparation_time_s
        + box_count * aircraft.load_time_per_box_s
        + aircraft.handoff_base_time_s
        + box_count * aircraft.handoff_time_per_box_s
    )
    total_time = flight_time + ground_time
    return_soc = 100.0 * (1.0 - total_energy / aircraft.usable_energy_kwh)

    mass_error = mass_kg - trip.mass_kg
    volume_error = volume_m3 - trip.volume_m3
    energy_error = total_energy - trip.energy_kwh
    time_error = total_time - trip.time_s
    soc_error = return_soc - trip.return_soc_percent
    same_service = all(box.service_id == trip.service_id for box in boxes)
    mass_capacity = mass_kg <= aircraft.max_payload_kg + MASS_TOLERANCE_KG
    volume_capacity = (
        volume_m3 <= aircraft.volume_capacity_m3 + VOLUME_TOLERANCE_M3
    )
    reserve = return_soc >= reserve_ratio * 100.0 - SOC_TOLERANCE_PERCENT
    constraints_pass = (
        same_service
        and mass_capacity
        and volume_capacity
        and reserve
        and abs(mass_error) <= MASS_TOLERANCE_KG
        and abs(volume_error) <= VOLUME_TOLERANCE_M3
        and abs(energy_error) <= ENERGY_TOLERANCE_KWH
        and abs(time_error) <= TIME_TOLERANCE_S
        and abs(soc_error) <= SOC_TOLERANCE_PERCENT
    )
    return IndependentTripAudit(
        trip_id=trip.trip_id,
        service_id=trip.service_id,
        model_id=trip.model_id,
        box_count=box_count,
        mass_kg=mass_kg,
        volume_m3=volume_m3,
        outbound_payload_kg=mass_kg,
        return_payload_kg=0.0,
        outbound_equivalent_range_m=outbound_range,
        return_equivalent_range_m=return_range,
        outbound_horizontal_kwh=outbound_horizontal,
        return_horizontal_kwh=return_horizontal,
        outbound_climb_kwh=outbound_climb,
        return_climb_kwh=return_climb,
        descent_energy_kwh=descent_energy,
        total_energy_kwh=total_energy,
        flight_time_s=flight_time,
        ground_time_s=ground_time,
        total_time_s=total_time,
        return_soc_percent=return_soc,
        mass_error_kg=mass_error,
        volume_error_m3=volume_error,
        energy_error_kwh=energy_error,
        time_error_s=time_error,
        soc_error_percent=soc_error,
        same_service_pass=same_service,
        mass_capacity_pass=mass_capacity,
        volume_capacity_pass=volume_capacity,
        reserve_pass=reserve,
        constraints_pass=constraints_pass,
    )


def audit_solution_independently(
    data: ProblemData,
    solution: MethodSolution,
    reserve_ratio: float = 0.20,
    *,
    range_energy_fraction: float = 1.0,
) -> IndependentSolutionAudit:
    audits = tuple(
        independent_trip_breakdown(
            data,
            trip,
            reserve_ratio,
            range_energy_fraction=range_energy_fraction,
        )
        for trip in solution.trips
    )
    assigned = [box_id for trip in solution.trips for box_id in trip.box_ids]
    expected = {box.box_id for box in data.boxes}
    coverage_pass = len(assigned) == len(set(assigned)) and set(assigned) == expected
    recomputed_energy = sum(audit.total_energy_kwh for audit in audits)
    recomputed_time = sum(audit.total_time_s for audit in audits)
    energy_sum_pass = (
        abs(recomputed_energy - solution.total_energy_kwh) <= ENERGY_TOLERANCE_KWH
    )
    time_sum_pass = abs(recomputed_time - solution.total_time_s) <= TIME_TOLERANCE_S
    constraints_pass = all(audit.constraints_pass for audit in audits)
    return IndependentSolutionAudit(
        trip_audits=audits,
        total_assignment_count=len(assigned),
        unique_box_count=len(set(assigned)),
        expected_box_count=len(expected),
        coverage_pass=coverage_pass,
        constraints_pass=constraints_pass,
        recomputed_energy_kwh=recomputed_energy,
        reported_energy_kwh=solution.total_energy_kwh,
        energy_sum_pass=energy_sum_pass,
        recomputed_time_s=recomputed_time,
        reported_time_s=solution.total_time_s,
        time_sum_pass=time_sum_pass,
        all_passed=(
            coverage_pass and constraints_pass and energy_sum_pass and time_sum_pass
        ),
    )


def representative_trips(solution: MethodSolution) -> Mapping[str, TripResult]:
    """Select the four paper-required representative sorties deterministically."""
    by_service = {
        service_id: [trip for trip in solution.trips if trip.service_id == service_id]
        for service_id in {trip.service_id for trip in solution.trips}
    }
    near_heavy = max(by_service["S001"], key=lambda trip: trip.mass_kg)
    far_candidates = by_service.get("S003", []) + by_service.get("S008", [])
    far_heavy = max(far_candidates, key=lambda trip: trip.mass_kg)
    b_candidates = [trip for trip in solution.trips if trip.model_id == "B"]
    b_25 = min(
        b_candidates,
        key=lambda trip: (abs(trip.mass_kg - 25.0), trip.trip_id),
    )
    minimum_soc = min(
        solution.trips, key=lambda trip: (trip.return_soc_percent, trip.trip_id)
    )
    return {
        "S001近距离重载架次": near_heavy,
        "S003或S008远距离重载架次": far_heavy,
        "B型25kg架次": b_25,
        "最低返航SOC架次": minimum_soc,
    }
