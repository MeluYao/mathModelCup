"""Feasible grouped construction and local improvement for question 2."""

from __future__ import annotations

from dataclasses import replace
import random
from time import perf_counter
from typing import Mapping, Sequence, Tuple

from .problem_d_q2 import (
    InfeasibleQ2Error,
    Q2Box,
    Q2Data,
    Q2Solution,
    TripDraft,
    TripPlan,
    TripStop,
)
from .problem_d_q2_geometry import ArcGeometry
from .problem_d_q2_physics import evaluate_trip
from .problem_d_q2_schedule import (
    construct_resource_aware_boxwise_solution,
    schedule_trips_greedy,
)
from .problem_d_q2_urgent import (
    PUBLISHED_MODE,
    URGENT_PRIORITY_MODE,
    is_urgent_box,
    required_deadline_s,
    urgent_search_key,
)


def _best_plan(
    data: Q2Data,
    arcs: Mapping[Tuple[str, str], ArcGeometry],
    stops: Sequence[TripStop],
    *,
    preferred_model_id: str | None = None,
    reserve_ratio: float = 0.20,
    range_energy_fraction: float = 1.0,
) -> TripPlan | None:
    candidates = []
    for model_id in sorted(data.aircraft_models):
        try:
            plan = evaluate_trip(
                data,
                arcs,
                TripDraft(model_id, tuple(stops)),
                reserve_ratio=reserve_ratio,
                range_energy_fraction=range_energy_fraction,
            )
        except InfeasibleQ2Error:
            continue
        candidates.append(
            (
                plan.energy_kwh,
                0 if model_id == preferred_model_id else 1,
                plan.duration_s,
                model_id,
                plan,
            )
        )
    return min(candidates)[-1] if candidates else None


def _feasible_plans(
    data: Q2Data,
    arcs: Mapping[Tuple[str, str], ArcGeometry],
    stops: Sequence[TripStop],
    reserve_ratio: float = 0.20,
    range_energy_fraction: float = 1.0,
) -> tuple[TripPlan, ...]:
    plans = []
    for model_id in sorted(data.aircraft_models):
        try:
            plans.append(
                evaluate_trip(
                    data,
                    arcs,
                    TripDraft(model_id, tuple(stops)),
                    reserve_ratio=reserve_ratio,
                    range_energy_fraction=range_energy_fraction,
                )
            )
        except InfeasibleQ2Error:
            continue
    return tuple(plans)


def _box_order(box: Q2Box) -> tuple[float, float, float, str]:
    deadline = box.hard_deadline_s
    return (
        deadline if deadline is not None else float("inf"),
        -box.mass_kg,
        -box.volume_m3,
        box.box_id,
    )


def _build_pure_ffd_plans(
    data: Q2Data,
    arcs: Mapping[Tuple[str, str], ArcGeometry],
    reserve_ratio: float = 0.20,
    range_energy_fraction: float = 1.0,
) -> tuple[TripPlan, ...]:
    plans = []
    service_ids = sorted({box.service_id for box in data.boxes.values()})
    for service_id in service_ids:
        boxes = sorted(
            (box for box in data.boxes.values() if box.service_id == service_id),
            key=_box_order,
        )
        groups: list[list[str]] = []
        for box in boxes:
            for group in groups:
                stop = TripStop(service_id, tuple(group + [box.box_id]))
                if _best_plan(
                    data,
                    arcs,
                    (stop,),
                    reserve_ratio=reserve_ratio,
                    range_energy_fraction=range_energy_fraction,
                ) is not None:
                    group.append(box.box_id)
                    break
            else:
                groups.append([box.box_id])
        for group in groups:
            plan = _best_plan(
                data,
                arcs,
                (TripStop(service_id, tuple(group)),),
                reserve_ratio=reserve_ratio,
                range_energy_fraction=range_energy_fraction,
            )
            if plan is None:
                raise InfeasibleQ2Error(f"FFD produced an infeasible group at {service_id}")
            plans.append(plan)
    return tuple(plans)


def build_grouped_initial_plans(
    data: Q2Data,
    arcs: Mapping[Tuple[str, str], ArcGeometry],
    reserve_ratio: float = 0.20,
    range_energy_fraction: float = 1.0,
    evaluation_mode: str = PUBLISHED_MODE,
) -> tuple[TripPlan, ...]:
    """Build a resource-safe hard-deadline core plus FFD soft-demand trips."""
    singleton_candidates = []
    for box in data.boxes.values():
        for model_id in sorted(data.aircraft_models):
            try:
                singleton_candidates.append(
                    evaluate_trip(
                        data,
                        arcs,
                        TripDraft(
                            model_id,
                            (TripStop(box.service_id, (box.box_id,)),),
                        ),
                        reserve_ratio=reserve_ratio,
                        range_energy_fraction=range_energy_fraction,
                    )
                )
            except InfeasibleQ2Error:
                continue
    boxwise = construct_resource_aware_boxwise_solution(
        data,
        singleton_candidates,
        method="grouped_hard_core",
        evaluation_mode=evaluation_mode,
    )
    hard_plan_by_box = {
        trip.plan.box_ids[0]: trip.plan
        for trip in boxwise.trips
        if (
            evaluation_mode == URGENT_PRIORITY_MODE
            or data.boxes[trip.plan.box_ids[0]].hard_deadline_s is not None
        )
    }
    plans = list(hard_plan_by_box.values())
    service_ids = sorted({box.service_id for box in data.boxes.values()})
    for service_id in service_ids:
        boxes = sorted(
            (
                box
                for box in data.boxes.values()
                if box.service_id == service_id
                and evaluation_mode != URGENT_PRIORITY_MODE
                and box.hard_deadline_s is None
            ),
            key=_box_order,
        )
        groups: list[list[str]] = []
        for box in boxes:
            for group in groups:
                candidate_ids = tuple(group + [box.box_id])
                if _best_plan(
                    data,
                    arcs,
                    (TripStop(service_id, candidate_ids),),
                    reserve_ratio=reserve_ratio,
                    range_energy_fraction=range_energy_fraction,
                ) is not None:
                    group.append(box.box_id)
                    break
            else:
                singleton = (TripStop(service_id, (box.box_id,)),)
                if _best_plan(
                    data,
                    arcs,
                    singleton,
                    reserve_ratio=reserve_ratio,
                    range_energy_fraction=range_energy_fraction,
                ) is None:
                    raise InfeasibleQ2Error(
                        f"no aircraft model can carry box {box.box_id}"
                    )
                groups.append([box.box_id])

        for group in groups:
            plan = _best_plan(
                data,
                arcs,
                (TripStop(service_id, tuple(group)),),
                reserve_ratio=reserve_ratio,
                range_energy_fraction=range_energy_fraction,
            )
            if plan is None:
                raise AssertionError("FFD accepted a group that is no longer feasible")
            plans.append(plan)
    return tuple(plans)


def _ordered_merged_stops(
    data: Q2Data,
    arcs: Mapping[Tuple[str, str], ArcGeometry],
    left: TripPlan,
    right: TripPlan,
) -> tuple[TripStop, ...]:
    grouped: dict[str, list[str]] = {}
    for stop in left.stops + right.stops:
        grouped.setdefault(stop.service_id, []).extend(stop.box_ids)

    def stop_deadline(service_id: str) -> float:
        deadlines = [
            data.boxes[box_id].hard_deadline_s
            for box_id in grouped[service_id]
            if data.boxes[box_id].hard_deadline_s is not None
        ]
        return min(deadlines, default=float("inf"))

    hard = sorted(
        (service_id for service_id in grouped if stop_deadline(service_id) < float("inf")),
        key=lambda service_id: (stop_deadline(service_id), service_id),
    )
    soft = [service_id for service_id in grouped if service_id not in hard]
    ordered = list(hard)
    previous = ordered[-1] if ordered else "O01"
    while soft:
        next_service = min(
            soft,
            key=lambda service_id: (
                arcs[(previous, service_id)].distance_m,
                service_id,
            ),
        )
        ordered.append(next_service)
        soft.remove(next_service)
        previous = next_service
    return tuple(
        TripStop(service_id, tuple(sorted(grouped[service_id])))
        for service_id in ordered
    )


def _ordered_stops(
    data: Q2Data,
    arcs: Mapping[Tuple[str, str], ArcGeometry],
    stops: Sequence[TripStop],
) -> tuple[TripStop, ...]:
    """Merge duplicate services and order hard-deadline stops before near neighbours."""
    grouped: dict[str, list[str]] = {}
    for stop in stops:
        if stop.box_ids:
            grouped.setdefault(stop.service_id, []).extend(stop.box_ids)
    if not grouped:
        return ()

    def stop_deadline(service_id: str) -> float:
        deadlines = [
            data.boxes[box_id].hard_deadline_s
            for box_id in grouped[service_id]
            if data.boxes[box_id].hard_deadline_s is not None
        ]
        return min(deadlines, default=float("inf"))

    hard = sorted(
        (service_id for service_id in grouped if stop_deadline(service_id) < float("inf")),
        key=lambda service_id: (stop_deadline(service_id), service_id),
    )
    soft = [service_id for service_id in grouped if service_id not in hard]
    ordered = list(hard)
    previous = ordered[-1] if ordered else "O01"
    while soft:
        next_service = min(
            soft,
            key=lambda service_id: (
                arcs[(previous, service_id)].distance_m,
                service_id,
            ),
        )
        ordered.append(next_service)
        soft.remove(next_service)
        previous = next_service
    return tuple(
        TripStop(service_id, tuple(sorted(grouped[service_id])))
        for service_id in ordered
    )


def _grouped_quality(
    data: Q2Data,
    solution: Q2Solution,
    objective_order: str = "timeliness",
    evaluation_mode: str = PUBLISHED_MODE,
) -> tuple[float, ...]:
    if evaluation_mode == URGENT_PRIORITY_MODE:
        return urgent_search_key(data, solution)
    if evaluation_mode != PUBLISHED_MODE:
        raise ValueError(f"unknown Q2 evaluation mode: {evaluation_mode}")
    hard_violations = sum(
        1
        for record in solution.deliveries
        if data.boxes[record.box_id].hard_deadline_s is not None
        and record.delivery_time_s
        > float(data.boxes[record.box_id].hard_deadline_s) + 1e-7
    )
    weighted_tardiness = sum(
        data.boxes[record.box_id].priority_weight
        * max(0.0, record.delivery_time_s - data.boxes[record.box_id].expected_time_s)
        for record in solution.deliveries
        if data.boxes[record.box_id].hard_deadline_s is None
    )
    components = {
        "H": float(hard_violations),
        "W": weighted_tardiness,
        "M": solution.objective[1],
        "E": solution.objective[2],
        "K": float(solution.objective[3]),
        "Z1": solution.objective[0],
    }
    orders = {
        "published": ("H", "Z1", "M", "E", "K", "W"),
        "timeliness": ("H", "W", "M", "E", "K", "Z1"),
        "trip_count": ("H", "K", "W", "M", "E", "Z1"),
    }
    if objective_order not in orders:
        raise ValueError(f"unknown grouped objective order: {objective_order}")
    return tuple(components[name] for name in orders[objective_order])


def _schedule_search_state(
    data: Q2Data,
    plans: Sequence[TripPlan],
    method: str,
    evaluation_mode: str = PUBLISHED_MODE,
) -> Q2Solution:
    return schedule_trips_greedy(
        data,
        plans,
        method=method,
        enforce_hard_deadlines=False,
        evaluation_mode=evaluation_mode,
    )


def _tardy_box_ids(
    data: Q2Data,
    solution: Q2Solution,
    evaluation_mode: str = PUBLISHED_MODE,
) -> tuple[str, ...]:
    tardy = []
    for record in solution.deliveries:
        box = data.boxes[record.box_id]
        limit = (
            required_deadline_s(box)
            if evaluation_mode == URGENT_PRIORITY_MODE
            else box.hard_deadline_s
        )
        is_hard = limit is not None
        if limit is None:
            limit = box.expected_time_s
        delay = record.delivery_time_s - float(limit)
        if delay > 1e-7:
            tardy.append(
                (
                    0
                    if evaluation_mode == URGENT_PRIORITY_MODE and is_urgent_box(box)
                    else 1 if evaluation_mode == URGENT_PRIORITY_MODE else 0 if is_hard else 1,
                    -box.priority_weight * delay,
                    record.box_id,
                )
            )
    return tuple(item[-1] for item in sorted(tardy))


def _remove_box_stops(plan: TripPlan, box_id: str) -> tuple[TripStop, ...]:
    stops = []
    for stop in plan.stops:
        remaining = tuple(item for item in stop.box_ids if item != box_id)
        if remaining:
            stops.append(TripStop(stop.service_id, remaining))
    return tuple(stops)


def _repair_tardy_boxes(
    data: Q2Data,
    arcs: Mapping[Tuple[str, str], ArcGeometry],
    plans: list[TripPlan],
    current: Q2Solution,
    method: str,
    max_steps: int,
    reserve_ratio: float,
    range_energy_fraction: float,
    objective_order: str,
    evaluation_mode: str = PUBLISHED_MODE,
) -> tuple[list[TripPlan], Q2Solution, int]:
    current_quality = _grouped_quality(
        data, current, objective_order, evaluation_mode
    )
    accepted = 0
    for _ in range(max_steps):
        tardy_ids = _tardy_box_ids(data, current, evaluation_mode)
        if not tardy_ids:
            break
        best = None
        for box_id in tardy_ids:
            source_index = next(
                index for index, plan in enumerate(plans) if box_id in plan.box_ids
            )
            source_stops = _remove_box_stops(plans[source_index], box_id)
            source_variants: tuple[TripPlan | None, ...]
            if source_stops:
                source_variants = _feasible_plans(
                    data,
                    arcs,
                    source_stops,
                    reserve_ratio=reserve_ratio,
                    range_energy_fraction=range_energy_fraction,
                )
            else:
                source_variants = (None,)
            box = data.boxes[box_id]
            singleton_variants = _feasible_plans(
                data,
                arcs,
                (TripStop(box.service_id, (box_id,)),),
                reserve_ratio=reserve_ratio,
                range_energy_fraction=range_energy_fraction,
            )
            for source in source_variants:
                for singleton in singleton_variants:
                    trial_plans = [
                        plan for index, plan in enumerate(plans) if index != source_index
                    ]
                    if source is not None:
                        trial_plans.append(source)
                    trial_plans.append(singleton)
                    trial = _schedule_search_state(
                        data, trial_plans, method, evaluation_mode
                    )
                    quality = _grouped_quality(
                        data, trial, objective_order, evaluation_mode
                    )
                    if quality < current_quality and (best is None or quality < best[0]):
                        best = (quality, trial_plans, trial)
        if best is None:
            break
        current_quality, plans, current = best
        accepted += 1
    return plans, current, accepted


def _merge_improving_plans(
    data: Q2Data,
    arcs: Mapping[Tuple[str, str], ArcGeometry],
    plans: list[TripPlan],
    current: Q2Solution,
    method: str,
    max_steps: int,
    max_stops: int,
    reserve_ratio: float,
    range_energy_fraction: float,
    objective_order: str,
    evaluation_mode: str = PUBLISHED_MODE,
) -> tuple[list[TripPlan], Q2Solution, int]:
    current_quality = _grouped_quality(
        data, current, objective_order, evaluation_mode
    )
    accepted = 0
    for _ in range(max_steps):
        candidates = []
        for left_index in range(len(plans)):
            for right_index in range(left_index + 1, len(plans)):
                stops = _ordered_merged_stops(
                    data,
                    arcs,
                    plans[left_index],
                    plans[right_index],
                )
                if len(stops) > max_stops:
                    continue
                for merged in _feasible_plans(
                    data,
                    arcs,
                    stops,
                    reserve_ratio=reserve_ratio,
                    range_energy_fraction=range_energy_fraction,
                ):
                    saving = (
                        plans[left_index].energy_kwh
                        + plans[right_index].energy_kwh
                        - merged.energy_kwh
                    )
                    candidates.append(
                        (
                            -saving,
                            merged.duration_s,
                            merged.signature,
                            left_index,
                            right_index,
                            merged,
                        )
                    )
        if not candidates:
            break
        candidates.sort()
        improved = False
        for _, _, _, left_index, right_index, merged in candidates:
            trial_plans = [
                plan
                for index, plan in enumerate(plans)
                if index not in (left_index, right_index)
            ]
            trial_plans.append(merged)
            trial = _schedule_search_state(
                data, trial_plans, method, evaluation_mode
            )
            quality = _grouped_quality(
                data, trial, objective_order, evaluation_mode
            )
            if quality < current_quality:
                plans = trial_plans
                current = trial
                current_quality = quality
                accepted += 1
                improved = True
                break
        if not improved:
            break
    return plans, current, accepted


def _polish_retypes(
    data: Q2Data,
    arcs: Mapping[Tuple[str, str], ArcGeometry],
    plans: list[TripPlan],
    current: Q2Solution,
    method: str,
    reserve_ratio: float,
    range_energy_fraction: float,
    objective_order: str,
    evaluation_mode: str = PUBLISHED_MODE,
    max_passes: int = 4,
) -> tuple[list[TripPlan], Q2Solution, int]:
    """Change aircraft models when global resource timing improves."""
    current_quality = _grouped_quality(
        data, current, objective_order, evaluation_mode
    )
    accepted = 0
    for _ in range(max_passes):
        best = None
        for index, plan in enumerate(plans):
            for variant in _feasible_plans(
                data,
                arcs,
                plan.stops,
                reserve_ratio=reserve_ratio,
                range_energy_fraction=range_energy_fraction,
            ):
                if variant.model_id == plan.model_id:
                    continue
                trial_plans = list(plans)
                trial_plans[index] = variant
                trial = _schedule_search_state(
                    data, trial_plans, method, evaluation_mode
                )
                quality = _grouped_quality(
                    data, trial, objective_order, evaluation_mode
                )
                if quality < current_quality and (best is None or quality < best[0]):
                    best = (quality, trial_plans, trial)
        if best is None:
            break
        current_quality, plans, current = best
        accepted += 1
    return plans, current, accepted


def _local_search_moves(
    data: Q2Data,
    arcs: Mapping[Tuple[str, str], ArcGeometry],
    plans: list[TripPlan],
    current: Q2Solution,
    method: str,
    reserve_ratio: float,
    range_energy_fraction: float,
    max_stops: int,
    seed: int,
    iterations: int,
    objective_order: str,
    evaluation_mode: str = PUBLISHED_MODE,
) -> tuple[list[TripPlan], Q2Solution, int]:
    """Reference-style improving moves with full physics and resource decoding."""
    rng = random.Random(seed)
    current_quality = _grouped_quality(
        data, current, objective_order, evaluation_mode
    )
    accepted = 0

    def best_plan(stops: Sequence[TripStop], preferred: str | None = None) -> TripPlan | None:
        ordered = _ordered_stops(data, arcs, stops)
        if not ordered or len(ordered) > max_stops:
            return None
        return _best_plan(
            data,
            arcs,
            ordered,
            preferred_model_id=preferred,
            reserve_ratio=reserve_ratio,
            range_energy_fraction=range_energy_fraction,
        )

    def try_accept(trial_plans: list[TripPlan]) -> bool:
        nonlocal plans, current, current_quality, accepted
        trial = _schedule_search_state(data, trial_plans, method, evaluation_mode)
        quality = _grouped_quality(data, trial, objective_order, evaluation_mode)
        if quality < current_quality:
            plans = trial_plans
            current = trial
            current_quality = quality
            accepted += 1
            return True
        return False

    for _ in range(iterations):
        if not plans:
            break
        trial_plans = list(plans)
        move = rng.random()

        if move < 0.28 and len(trial_plans) >= 2:  # move one complete service group
            source_index = rng.randrange(len(trial_plans))
            source = trial_plans[source_index]
            stop_index = rng.randrange(len(source.stops))
            moved_stop = source.stops[stop_index]
            destination_index = rng.choice(
                [index for index in range(len(trial_plans)) if index != source_index]
            )
            destination = trial_plans[destination_index]
            new_destination = best_plan(
                destination.stops + (moved_stop,), destination.model_id
            )
            if new_destination is None:
                continue
            remaining_stops = tuple(
                stop for index, stop in enumerate(source.stops) if index != stop_index
            )
            new_source = (
                best_plan(remaining_stops, source.model_id) if remaining_stops else None
            )
            if remaining_stops and new_source is None:
                continue
            rebuilt = [
                plan
                for index, plan in enumerate(trial_plans)
                if index not in (source_index, destination_index)
            ]
            if new_source is not None:
                rebuilt.append(new_source)
            rebuilt.append(new_destination)
            try_accept(rebuilt)

        elif move < 0.46 and len(trial_plans) >= 2:  # move one box
            source_index = rng.randrange(len(trial_plans))
            source = trial_plans[source_index]
            box_id = rng.choice(source.box_ids)
            box = data.boxes[box_id]
            destination_index = rng.choice(
                [index for index in range(len(trial_plans)) if index != source_index]
            )
            destination = trial_plans[destination_index]
            destination_stops = destination.stops + (
                TripStop(box.service_id, (box_id,)),
            )
            new_destination = best_plan(destination_stops, destination.model_id)
            if new_destination is None:
                continue
            remaining_stops = _remove_box_stops(source, box_id)
            new_source = (
                best_plan(remaining_stops, source.model_id) if remaining_stops else None
            )
            if remaining_stops and new_source is None:
                continue
            rebuilt = [
                plan
                for index, plan in enumerate(trial_plans)
                if index not in (source_index, destination_index)
            ]
            if new_source is not None:
                rebuilt.append(new_source)
            rebuilt.append(new_destination)
            try_accept(rebuilt)

        elif move < 0.66 and len(trial_plans) >= 2:  # merge two trips
            left_index, right_index = rng.sample(range(len(trial_plans)), 2)
            merged = best_plan(
                trial_plans[left_index].stops + trial_plans[right_index].stops
            )
            if merged is None:
                continue
            rebuilt = [
                plan
                for index, plan in enumerate(trial_plans)
                if index not in (left_index, right_index)
            ]
            rebuilt.append(merged)
            try_accept(rebuilt)

        elif move < 0.84:  # split a route or a large same-service group
            source_index = rng.randrange(len(trial_plans))
            source = trial_plans[source_index]
            if len(source.stops) >= 2:
                cut = rng.randrange(1, len(source.stops))
                left_stops = source.stops[:cut]
                right_stops = source.stops[cut:]
            elif len(source.box_ids) >= 4:
                stop = source.stops[0]
                ordered_boxes = sorted(
                    stop.box_ids,
                    key=lambda box_id: (
                        data.boxes[box_id].hard_deadline_s
                        if data.boxes[box_id].hard_deadline_s is not None
                        else float("inf"),
                        box_id,
                    ),
                )
                cut = len(ordered_boxes) // 2
                left_stops = (TripStop(stop.service_id, tuple(ordered_boxes[:cut])),)
                right_stops = (TripStop(stop.service_id, tuple(ordered_boxes[cut:])),)
            else:
                continue
            left = best_plan(left_stops)
            right = best_plan(right_stops)
            if left is None or right is None:
                continue
            rebuilt = [
                plan for index, plan in enumerate(trial_plans) if index != source_index
            ]
            rebuilt.extend((left, right))
            try_accept(rebuilt)

        else:  # route reordering or aircraft retyping
            source_index = rng.randrange(len(trial_plans))
            source = trial_plans[source_index]
            if len(source.stops) >= 2:
                shuffled = list(source.stops)
                rng.shuffle(shuffled)
                variant = _best_plan(
                    data,
                    arcs,
                    tuple(shuffled),
                    preferred_model_id=source.model_id,
                    reserve_ratio=reserve_ratio,
                    range_energy_fraction=range_energy_fraction,
                )
            else:
                variants = [
                    variant
                    for variant in _feasible_plans(
                        data,
                        arcs,
                        source.stops,
                        reserve_ratio=reserve_ratio,
                        range_energy_fraction=range_energy_fraction,
                    )
                    if variant.model_id != source.model_id
                ]
                variant = rng.choice(variants) if variants else None
            if variant is None:
                continue
            trial_plans[source_index] = variant
            try_accept(trial_plans)

    return plans, current, accepted


def solve_grouped_local_search(
    data: Q2Data,
    arcs: Mapping[Tuple[str, str], ArcGeometry],
    *,
    seed: int = 20260924,
    iterations: int = 200,
    method: str = "grouped_local_search",
    max_stops: int = 4,
    reserve_ratio: float = 0.20,
    range_energy_fraction: float = 1.0,
    objective_order: str = "timeliness",
    evaluation_mode: str = PUBLISHED_MODE,
) -> Q2Solution:
    """Repair and improve pure FFD trips with full resource simulation."""
    if iterations < 0:
        raise ValueError("iterations must be non-negative")
    if max_stops < 1:
        raise ValueError("max_stops must be positive")
    if objective_order not in {"published", "timeliness", "trip_count"}:
        raise ValueError(
            "objective_order must be 'published', 'timeliness', or 'trip_count'"
        )
    started = perf_counter()
    plans = list(
        _build_pure_ffd_plans(
            data,
            arcs,
            reserve_ratio=reserve_ratio,
            range_energy_fraction=range_energy_fraction,
        )
    )
    initial_trip_count = len(plans)
    current = _schedule_search_state(data, plans, method, evaluation_mode)
    plans, current, accepted_repairs = _repair_tardy_boxes(
        data,
        arcs,
        plans,
        current,
        method,
        max_steps=max(1, iterations // 2),
        reserve_ratio=reserve_ratio,
        range_energy_fraction=range_energy_fraction,
        objective_order=objective_order,
        evaluation_mode=evaluation_mode,
    )
    plans, current, accepted_merges = _merge_improving_plans(
        data,
        arcs,
        plans,
        current,
        method,
        max_steps=max(1, iterations // 2),
        max_stops=max_stops,
        reserve_ratio=reserve_ratio,
        range_energy_fraction=range_energy_fraction,
        objective_order=objective_order,
        evaluation_mode=evaluation_mode,
    )
    base_plans = list(plans)
    base_solution = current
    best_branch = (
        _grouped_quality(data, current, objective_order, evaluation_mode),
        plans,
        current,
        0,
        0,
        0,
        seed,
    )
    for branch_seed in (seed, seed + 1, seed + 2):
        branch_plans = list(base_plans)
        branch_solution = base_solution
        branch_plans, branch_solution, moves_a = _local_search_moves(
            data,
            arcs,
            branch_plans,
            branch_solution,
            method,
            reserve_ratio,
            range_energy_fraction,
            max_stops,
            branch_seed,
            iterations,
            objective_order,
            evaluation_mode,
        )
        branch_plans, branch_solution, retypes_a = _polish_retypes(
            data,
            arcs,
            branch_plans,
            branch_solution,
            method,
            reserve_ratio,
            range_energy_fraction,
            objective_order,
            evaluation_mode,
        )
        branch_plans, branch_solution, moves_b = _local_search_moves(
            data,
            arcs,
            branch_plans,
            branch_solution,
            method,
            reserve_ratio,
            range_energy_fraction,
            max_stops,
            branch_seed + 100,
            max(200, iterations // 2),
            objective_order,
            evaluation_mode,
        )
        branch_plans, branch_solution, retypes_b = _polish_retypes(
            data,
            arcs,
            branch_plans,
            branch_solution,
            method,
            reserve_ratio,
            range_energy_fraction,
            objective_order,
            evaluation_mode,
        )
        branch_plans, branch_solution, branch_merges = _merge_improving_plans(
            data,
            arcs,
            branch_plans,
            branch_solution,
            method,
            max_steps=max(1, iterations // 4),
            max_stops=max_stops,
            reserve_ratio=reserve_ratio,
            range_energy_fraction=range_energy_fraction,
            objective_order=objective_order,
            evaluation_mode=evaluation_mode,
        )
        branch_quality = _grouped_quality(
            data, branch_solution, objective_order, evaluation_mode
        )
        if branch_quality < best_branch[0]:
            best_branch = (
                branch_quality,
                branch_plans,
                branch_solution,
                moves_a + moves_b,
                retypes_a + retypes_b,
                branch_merges,
                branch_seed,
            )
    (
        current_quality,
        plans,
        current,
        accepted_local_moves,
        accepted_retypes,
        final_merges,
        best_seed,
    ) = best_branch
    accepted_merges += final_merges

    if current_quality[0] > 0:
        fallback_plans = list(
            build_grouped_initial_plans(
                data,
                arcs,
                reserve_ratio=reserve_ratio,
                range_energy_fraction=range_energy_fraction,
                evaluation_mode=evaluation_mode,
            )
        )
        fallback = schedule_trips_greedy(
            data,
            fallback_plans,
            method=method,
            evaluation_mode=evaluation_mode,
        )
        fallback_plans, fallback, fallback_merges = _merge_improving_plans(
            data,
            arcs,
            fallback_plans,
            fallback,
            method,
            max_steps=max(1, iterations // 2),
            max_stops=max_stops,
            reserve_ratio=reserve_ratio,
            range_energy_fraction=range_energy_fraction,
            objective_order=objective_order,
            evaluation_mode=evaluation_mode,
        )
        if (
            _grouped_quality(data, fallback, objective_order, evaluation_mode)
            < current_quality
        ):
            plans = fallback_plans
            current = fallback
            current_quality = _grouped_quality(
                data, current, objective_order, evaluation_mode
            )
            accepted_merges = fallback_merges
            accepted_local_moves = 0
            accepted_retypes = 0

    diagnostics = dict(current.diagnostics)
    diagnostics.update(
        {
            "initialization": "pure_ffd_then_tardiness_repair",
            "initial_trip_count": initial_trip_count,
            "accepted_repairs": accepted_repairs,
            "accepted_merges": accepted_merges,
            "accepted_local_moves": accepted_local_moves,
            "accepted_retypes": accepted_retypes,
            "best_local_seed": best_seed,
            "grouped_quality": current_quality,
            "incumbent_source": "grouped_local_search",
            "reserve_ratio": reserve_ratio,
            "range_energy_fraction": range_energy_fraction,
            "objective_order": objective_order,
            "evaluation_mode": evaluation_mode,
        }
    )
    return replace(
        current,
        method=method,
        runtime_s=perf_counter() - started,
        diagnostics=diagnostics,
    )
