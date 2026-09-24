"""Independent validation for complete question 2 solutions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Mapping, Tuple

from .problem_d_q2 import Q2Data, Q2Solution, TripDraft
from .problem_d_q2_geometry import ArcGeometry
from .problem_d_q2_physics import charge_time_s, evaluate_trip, objective_vector


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str
    reference: str = ""


@dataclass(frozen=True)
class ValidationReport:
    issues: Tuple[ValidationIssue, ...]
    checked_trip_count: int
    checked_box_count: int

    @property
    def is_valid(self) -> bool:
        return not self.issues


def _overlap_issues(solution: Q2Solution, resource: str) -> List[ValidationIssue]:
    grouped = {}
    for trip in solution.trips:
        if resource == "aircraft":
            key = trip.aircraft_id
            end = trip.return_time_s
        else:
            key = trip.battery_id
            end = trip.battery_ready_time_s
        grouped.setdefault(key, []).append((trip.start_time_s, end, trip.trip_id))
    issues = []
    for resource_id, intervals in grouped.items():
        intervals.sort()
        for left, right in zip(intervals, intervals[1:]):
            if right[0] < left[1] - 1e-7:
                issues.append(
                    ValidationIssue(
                        f"{resource}_overlap",
                        f"{resource_id} overlaps {left[2]} and {right[2]}",
                        resource_id,
                    )
                )
    return issues


def validate_q2_solution(
    data: Q2Data,
    arcs: Mapping[Tuple[str, str], ArcGeometry],
    solution: Q2Solution,
    reserve_ratio: float = 0.20,
) -> ValidationReport:
    issues: List[ValidationIssue] = []
    aircraft = {unit.aircraft_id: unit for unit in data.aircraft_units}
    batteries = {battery.battery_id: battery for battery in data.batteries}
    expected_delivery_times = {}

    for trip in solution.trips:
        if trip.aircraft_id not in aircraft:
            issues.append(ValidationIssue("unknown_aircraft", "unknown aircraft", trip.trip_id))
            continue
        if trip.battery_id not in batteries:
            issues.append(ValidationIssue("unknown_battery", "unknown battery", trip.trip_id))
            continue
        if aircraft[trip.aircraft_id].model_id != trip.plan.model_id:
            issues.append(ValidationIssue("aircraft_model", "aircraft model mismatch", trip.trip_id))
        if batteries[trip.battery_id].model_id != trip.plan.model_id:
            issues.append(ValidationIssue("battery_model", "battery model mismatch", trip.trip_id))
        try:
            recomputed = evaluate_trip(
                data,
                arcs,
                TripDraft(trip.plan.model_id, trip.plan.stops),
                reserve_ratio,
            )
        except Exception as error:
            issues.append(ValidationIssue("trip_infeasible", str(error), trip.trip_id))
            continue
        if abs(recomputed.energy_kwh - trip.plan.energy_kwh) > 1e-8:
            issues.append(ValidationIssue("energy_mismatch", "trip energy mismatch", trip.trip_id))
        if abs(recomputed.duration_s - trip.plan.duration_s) > 1e-6:
            issues.append(ValidationIssue("duration_mismatch", "trip duration mismatch", trip.trip_id))
        if abs(trip.return_time_s - (trip.start_time_s + recomputed.duration_s)) > 1e-6:
            issues.append(ValidationIssue("return_time", "return time mismatch", trip.trip_id))
        expected_ready = trip.return_time_s + charge_time_s(
            recomputed.return_soc_percent / 100.0,
            batteries[trip.battery_id].full_charge_time_s,
        )
        if abs(trip.battery_ready_time_s - expected_ready) > 1e-6:
            issues.append(ValidationIssue("battery_ready", "battery ready time mismatch", trip.trip_id))
        for box_id in recomputed.box_ids:
            expected_delivery_times[box_id] = (
                trip.trip_id,
                data.boxes[box_id].service_id,
                trip.start_time_s + recomputed.delivery_offsets_s[box_id],
            )

    delivered_ids = [record.box_id for record in solution.deliveries]
    if len(delivered_ids) != len(set(delivered_ids)):
        issues.append(ValidationIssue("duplicate_box", "a box is delivered more than once"))
    missing = sorted(set(data.boxes) - set(delivered_ids))
    extra = sorted(set(delivered_ids) - set(data.boxes))
    if missing:
        issues.append(ValidationIssue("missing_box", f"missing boxes: {missing}"))
    if extra:
        issues.append(ValidationIssue("unknown_box", f"unknown boxes: {extra}"))

    for record in solution.deliveries:
        expected = expected_delivery_times.get(record.box_id)
        if expected is None:
            continue
        if record.trip_id != expected[0] or record.service_id != expected[1]:
            issues.append(ValidationIssue("delivery_reference", "delivery reference mismatch", record.box_id))
        if abs(record.delivery_time_s - expected[2]) > 1e-6:
            issues.append(ValidationIssue("delivery_time", "delivery time mismatch", record.box_id))
        deadline = data.boxes[record.box_id].hard_deadline_s
        if deadline is not None and record.delivery_time_s > deadline + 1e-7:
            issues.append(ValidationIssue("hard_deadline", "hard deadline violated", record.box_id))

    issues.extend(_overlap_issues(solution, "aircraft"))
    issues.extend(_overlap_issues(solution, "battery"))

    if solution.trips and not issues:
        recomputed_objective = objective_vector(data, solution)
        if any(
            abs(float(left) - float(right)) > 1e-7
            for left, right in zip(recomputed_objective, solution.objective)
        ):
            issues.append(ValidationIssue("objective_mismatch", "objective vector mismatch"))

    return ValidationReport(
        issues=tuple(issues),
        checked_trip_count=len(solution.trips),
        checked_box_count=len(delivered_ids),
    )
