"""Time, energy, charging, and objective calculations for Q2 trips."""

from __future__ import annotations

from typing import Mapping, Tuple

from .problem_d_q1 import GRAVITY_MPS2, JOULES_PER_KWH
from .problem_d_q2 import (
    AircraftModel,
    InfeasibleQ2Error,
    Q2Data,
    Q2Solution,
    TripDraft,
    TripPlan,
)
from .problem_d_q2_geometry import ArcGeometry


NUMERIC_TOLERANCE = 1e-9


def equivalent_range_m(model: AircraftModel, payload_kg: float) -> float:
    if payload_kg < -NUMERIC_TOLERANCE or payload_kg > model.max_payload_kg + NUMERIC_TOLERANCE:
        raise InfeasibleQ2Error("payload mass is outside model capacity")
    payload = min(max(payload_kg, 0.0), model.max_payload_kg)
    fraction = payload / model.max_payload_kg
    return model.empty_range_m - (model.empty_range_m - model.full_range_m) * fraction**1.5


def leg_time_s(model: AircraftModel, arc: ArcGeometry) -> float:
    return (
        arc.climb_m / model.climb_speed_mps
        + arc.distance_m / model.cruise_speed_mps
        + arc.descent_m / model.descent_speed_mps
    )


def leg_energy_kwh(
    model: AircraftModel,
    arc: ArcGeometry,
    payload_kg: float,
    *,
    range_energy_fraction: float = 1.0,
) -> float:
    if not 0.0 < range_energy_fraction <= 1.0:
        raise ValueError("range_energy_fraction must be in (0, 1]")
    horizontal = (
        range_energy_fraction
        * model.usable_energy_kwh
        * arc.distance_m
        / equivalent_range_m(model, payload_kg)
    )
    climb = (
        (model.empty_mass_kg + payload_kg)
        * GRAVITY_MPS2
        * arc.climb_m
        / (JOULES_PER_KWH * model.climb_efficiency)
    )
    return horizontal + climb


def charge_time_s(end_soc: float, full_charge_time_s: float) -> float:
    """Return seconds to charge a battery from fractional SOC to 100%."""
    if not 0.0 <= end_soc <= 1.0:
        raise ValueError("end_soc must be in [0, 1]")
    if full_charge_time_s < 0.0:
        raise ValueError("full_charge_time_s must be non-negative")
    if end_soc < 0.90:
        return full_charge_time_s * (
            0.65 * (0.90 - end_soc) / 0.90 + 0.35
        )
    return full_charge_time_s * 0.35 * (1.0 - end_soc) / 0.10


def evaluate_trip(
    data: Q2Data,
    arcs: Mapping[Tuple[str, str], ArcGeometry],
    draft: TripDraft,
    reserve_ratio: float = 0.20,
    *,
    range_energy_fraction: float = 1.0,
) -> TripPlan:
    """Evaluate a complete multi-stop trip and reject any physical violation."""
    if not draft.stops:
        raise InfeasibleQ2Error("a trip must contain at least one service stop")
    if draft.model_id not in data.aircraft_models:
        raise InfeasibleQ2Error(f"unknown aircraft model {draft.model_id}")
    if not 0.0 <= reserve_ratio < 1.0:
        raise ValueError("reserve_ratio must be in [0, 1)")
    model = data.aircraft_models[draft.model_id]

    service_ids = tuple(stop.service_id for stop in draft.stops)
    if len(service_ids) != len(set(service_ids)):
        raise InfeasibleQ2Error("a service may appear only once in a trip")

    ordered_box_ids = tuple(box_id for stop in draft.stops for box_id in stop.box_ids)
    if not ordered_box_ids:
        raise InfeasibleQ2Error("a trip must carry at least one box")
    if len(ordered_box_ids) != len(set(ordered_box_ids)):
        raise InfeasibleQ2Error("a box may appear only once in a trip")

    for stop in draft.stops:
        if stop.service_id == "O01" or stop.service_id not in data.nodes:
            raise InfeasibleQ2Error(f"invalid service stop {stop.service_id}")
        if not stop.box_ids:
            raise InfeasibleQ2Error("each visited service must receive at least one box")
        for box_id in stop.box_ids:
            if box_id not in data.boxes:
                raise InfeasibleQ2Error(f"unknown box {box_id}")
            if data.boxes[box_id].service_id != stop.service_id:
                raise InfeasibleQ2Error(f"box {box_id} is assigned to the wrong service")

    total_mass = sum(data.boxes[box_id].mass_kg for box_id in ordered_box_ids)
    total_volume = sum(data.boxes[box_id].volume_m3 for box_id in ordered_box_ids)
    if total_mass > model.max_payload_kg + NUMERIC_TOLERANCE:
        raise InfeasibleQ2Error("payload mass exceeds model capacity")
    if total_volume > model.volume_capacity_m3 + NUMERIC_TOLERANCE:
        raise InfeasibleQ2Error("payload volume exceeds model capacity")

    elapsed = model.preparation_time_s + len(ordered_box_ids) * model.load_time_per_box_s
    payload = total_mass
    energy = 0.0
    leg_payloads = []
    delivery_offsets = {}
    origin_id = "O01"

    for stop in draft.stops:
        arc_key = (origin_id, stop.service_id)
        if arc_key not in arcs:
            raise InfeasibleQ2Error(f"missing arc {origin_id}->{stop.service_id}")
        arc = arcs[arc_key]
        leg_payloads.append(payload)
        elapsed += leg_time_s(model, arc)
        energy += leg_energy_kwh(
            model,
            arc,
            payload,
            range_energy_fraction=range_energy_fraction,
        )
        elapsed += model.handoff_base_time_s + len(stop.box_ids) * model.handoff_time_per_box_s
        for box_id in stop.box_ids:
            delivery_offsets[box_id] = elapsed
            payload -= data.boxes[box_id].mass_kg
        origin_id = stop.service_id

    if abs(payload) > 1e-7:
        raise AssertionError("all trip payload must be delivered before return")
    return_key = (origin_id, "O01")
    if return_key not in arcs:
        raise InfeasibleQ2Error(f"missing arc {origin_id}->O01")
    return_arc = arcs[return_key]
    leg_payloads.append(0.0)
    elapsed += leg_time_s(model, return_arc)
    energy += leg_energy_kwh(
        model,
        return_arc,
        0.0,
        range_energy_fraction=range_energy_fraction,
    )

    energy_limit = (1.0 - reserve_ratio) * model.usable_energy_kwh
    if energy > energy_limit + NUMERIC_TOLERANCE:
        raise InfeasibleQ2Error("trip energy exceeds reserve-constrained capacity")
    return_soc_percent = 100.0 * (1.0 - energy / model.usable_energy_kwh)

    return TripPlan(
        model_id=draft.model_id,
        stops=draft.stops,
        box_ids=ordered_box_ids,
        total_mass_kg=total_mass,
        total_volume_m3=total_volume,
        duration_s=elapsed,
        energy_kwh=energy,
        return_soc_percent=return_soc_percent,
        leg_payloads_kg=tuple(leg_payloads),
        delivery_offsets_s=delivery_offsets,
    )


def objective_vector(data: Q2Data, solution: Q2Solution) -> Tuple[float, float, float, int]:
    if not solution.deliveries:
        return (0.0, 0.0, 0.0, 0)
    total_weight = sum(data.boxes[record.box_id].priority_weight for record in solution.deliveries)
    weighted_delivery = sum(
        data.boxes[record.box_id].priority_weight
        * record.delivery_time_s
        / data.boxes[record.box_id].expected_time_s
        for record in solution.deliveries
    ) / total_weight
    makespan = max(trip.return_time_s for trip in solution.trips)
    total_energy = sum(trip.plan.energy_kwh for trip in solution.trips)
    return (weighted_delivery, makespan, total_energy, len(solution.trips))
