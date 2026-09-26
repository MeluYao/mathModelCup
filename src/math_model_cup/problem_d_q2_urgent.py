"""Urgent-first objective and deadline semantics for Problem D, question 2."""

from __future__ import annotations

from typing import Mapping, Protocol

from .problem_d_q2 import Q2Box, Q2Data, Q2Solution


URGENT_PRIORITY_MODE = "urgent_priority"
PUBLISHED_MODE = "published"


class _DataWithBoxes(Protocol):
    boxes: Mapping[str, Q2Box]


def is_urgent_box(box: Q2Box) -> bool:
    """Return whether a box belongs to the first-batch-or-medical urgent set."""
    return box.first_batch or box.material_type == "医疗物资"


def required_deadline_s(box: Q2Box) -> float:
    """Return the hard deadline used by the urgent-priority model."""
    if is_urgent_box(box) and box.hard_deadline_s is not None:
        return box.hard_deadline_s
    return box.expected_time_s


def urgent_lateness(
    data: _DataWithBoxes,
    solution: Q2Solution,
) -> tuple[int, float]:
    """Count late urgent boxes and their total lateness in seconds."""
    count = 0
    total_s = 0.0
    for delivery in solution.deliveries:
        box = data.boxes[delivery.box_id]
        if not is_urgent_box(box):
            continue
        lateness_s = max(0.0, delivery.delivery_time_s - required_deadline_s(box))
        if lateness_s > 1e-9:
            count += 1
            total_s += lateness_s
    return count, total_s


def all_box_lateness(
    data: _DataWithBoxes,
    solution: Q2Solution,
) -> tuple[int, float]:
    """Count all late boxes and total lateness under required deadlines."""
    count = 0
    total_s = 0.0
    for delivery in solution.deliveries:
        box = data.boxes[delivery.box_id]
        lateness_s = max(0.0, delivery.delivery_time_s - required_deadline_s(box))
        if lateness_s > 1e-9:
            count += 1
            total_s += lateness_s
    return count, total_s


def urgent_objective_vector(
    data: Q2Data | _DataWithBoxes,
    solution: Q2Solution,
) -> tuple[float, float, float, float]:
    """Return ``(urgent timing, energy, trip count, makespan)``."""
    delivery_times = {
        delivery.box_id: delivery.delivery_time_s for delivery in solution.deliveries
    }
    urgent_boxes = [box for box in data.boxes.values() if is_urgent_box(box)]
    total_weight = sum(box.priority_weight for box in urgent_boxes)
    if not urgent_boxes or total_weight <= 0.0:
        urgent_timing = 0.0
    else:
        urgent_timing = sum(
            box.priority_weight
            * delivery_times[box.box_id]
            / box.expected_time_s
            for box in urgent_boxes
        ) / total_weight
    energy_kwh = sum(trip.plan.energy_kwh for trip in solution.trips)
    makespan_s = max((trip.return_time_s for trip in solution.trips), default=0.0)
    return urgent_timing, energy_kwh, float(len(solution.trips)), makespan_s


def urgent_search_key(
    data: Q2Data | _DataWithBoxes,
    solution: Q2Solution,
) -> tuple[float, ...]:
    """Return the search key ``(late count, lateness, Z, E, K, M)``."""
    late_count, lateness_s = all_box_lateness(data, solution)
    return (
        float(late_count),
        lateness_s,
        *urgent_objective_vector(data, solution),
    )
