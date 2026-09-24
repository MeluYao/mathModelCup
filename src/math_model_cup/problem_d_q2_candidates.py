"""Candidate-trip generation and set-partitioning solver for question 2."""

from __future__ import annotations

from dataclasses import replace
from math import inf
from time import perf_counter
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csc_matrix, vstack

from .problem_d_q1 import load_problem_data, solve_dynamic_programming
from .problem_d_q2 import (
    InfeasibleQ2Error,
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
    schedule_trips_cp_sat,
)


def _try_plan(
    data: Q2Data,
    arcs: Mapping[Tuple[str, str], ArcGeometry],
    draft: TripDraft,
) -> TripPlan | None:
    try:
        return evaluate_trip(data, arcs, draft)
    except InfeasibleQ2Error:
        return None


def _service_distance(
    arcs: Mapping[Tuple[str, str], ArcGeometry],
    left_service_id: str,
    right_service_id: str,
) -> float:
    arc = arcs.get((left_service_id, right_service_id))
    return arc.distance_m if arc is not None else inf


def _insert_box_stops(
    stops: Tuple[TripStop, ...],
    service_id: str,
    box_id: str,
    max_stops: int,
) -> Tuple[Tuple[TripStop, ...], ...]:
    existing = next(
        (index for index, stop in enumerate(stops) if stop.service_id == service_id),
        None,
    )
    if existing is not None:
        updated = list(stops)
        stop = updated[existing]
        updated[existing] = TripStop(service_id, stop.box_ids + (box_id,))
        return (tuple(updated),)
    if len(stops) >= max_stops:
        return ()
    new_stop = TripStop(service_id, (box_id,))
    return tuple(
        stops[:position] + (new_stop,) + stops[position:]
        for position in range(len(stops) + 1)
    )


def _q1_seed_plans(
    data: Q2Data,
    arcs: Mapping[Tuple[str, str], ArcGeometry],
) -> List[TripPlan]:
    try:
        q1_data = load_problem_data(data.problem_dir)
        q1_solution = solve_dynamic_programming(q1_data)
    except (FileNotFoundError, ValueError):
        return []
    plans = []
    for trip in q1_solution.trips:
        plan = _try_plan(
            data,
            arcs,
            TripDraft(
                trip.model_id,
                (TripStop(trip.service_id, tuple(trip.box_ids)),),
            ),
        )
        if plan is not None:
            plans.append(plan)
    return plans


def generate_initial_candidates(
    data: Q2Data,
    arcs: Mapping[Tuple[str, str], ArcGeometry],
    max_stops: int = 4,
) -> Tuple[TripPlan, ...]:
    """Build a compact, guaranteed-covering initial trip pool."""
    if max_stops < 1:
        raise ValueError("max_stops must be positive")
    candidates: List[TripPlan] = []

    for box in data.boxes.values():
        for model_id in sorted(data.aircraft_models):
            plan = _try_plan(
                data,
                arcs,
                TripDraft(model_id, (TripStop(box.service_id, (box.box_id,)),)),
            )
            if plan is not None:
                candidates.append(plan)

    seed_plans = _q1_seed_plans(data, arcs)
    candidates.extend(seed_plans)

    frontier: List[TripPlan] = []
    if max_stops >= 2:
        box_ids = tuple(sorted(data.boxes))
        pair_keys = set()
        for left_id in box_ids:
            left = data.boxes[left_id]
            neighbors = sorted(
                (right_id for right_id in box_ids if right_id != left_id),
                key=lambda right_id: (
                    _service_distance(
                        arcs, left.service_id, data.boxes[right_id].service_id
                    ),
                    right_id,
                ),
            )[:10]
            for right_id in neighbors:
                pair_key = tuple(sorted((left_id, right_id)))
                if pair_key in pair_keys:
                    continue
                pair_keys.add(pair_key)
                right = data.boxes[right_id]
                for model_id in sorted(data.aircraft_models):
                    if left.service_id == right.service_id:
                        stop_orders = (
                            (TripStop(left.service_id, pair_key),),
                        )
                    else:
                        stop_orders = (
                            (
                                TripStop(left.service_id, (left_id,)),
                                TripStop(right.service_id, (right_id,)),
                            ),
                            (
                                TripStop(right.service_id, (right_id,)),
                                TripStop(left.service_id, (left_id,)),
                            ),
                        )
                    for stops in stop_orders:
                        plan = _try_plan(data, arcs, TripDraft(model_id, stops))
                        if plan is not None:
                            frontier.append(plan)
        frontier = list(prune_dominated_candidates(frontier))
        candidates.extend(frontier)

    for box_count in range(3, max_stops + 1):
        expanded: List[TripPlan] = []
        for base in frontier:
            used_boxes = set(base.box_ids)
            extensions = sorted(
                (box_id for box_id in data.boxes if box_id not in used_boxes),
                key=lambda box_id: (
                    min(
                        _service_distance(
                            arcs, stop.service_id, data.boxes[box_id].service_id
                        )
                        for stop in base.stops
                    ),
                    box_id,
                ),
            )[:5]
            for box_id in extensions:
                box = data.boxes[box_id]
                for stops in _insert_box_stops(
                    base.stops, box.service_id, box_id, max_stops
                ):
                    plan = _try_plan(data, arcs, TripDraft(base.model_id, stops))
                    if plan is not None:
                        expanded.append(plan)
        frontier = list(prune_dominated_candidates(expanded))
        if len(frontier) > 1_000:
            frontier = sorted(
                frontier,
                key=lambda plan: (
                    _candidate_cost(data, plan),
                    plan.duration_s,
                    plan.signature,
                ),
            )[:1_000]
        candidates.extend(frontier)

    return prune_dominated_candidates(candidates)


def prune_dominated_candidates(
    candidates: Iterable[TripPlan],
) -> Tuple[TripPlan, ...]:
    """Remove exact duplicates and dominated orderings for the same box cover."""
    by_signature: Dict[Tuple[object, ...], TripPlan] = {}
    for candidate in candidates:
        current = by_signature.get(candidate.signature)
        if current is None or (
            candidate.duration_s,
            candidate.energy_kwh,
        ) < (
            current.duration_s,
            current.energy_kwh,
        ):
            by_signature[candidate.signature] = candidate

    grouped: Dict[Tuple[str, frozenset[str]], List[TripPlan]] = {}
    for candidate in by_signature.values():
        grouped.setdefault(
            (candidate.model_id, frozenset(candidate.box_ids)), []
        ).append(candidate)

    kept = []
    for group in grouped.values():
        for candidate in group:
            dominated = False
            for other in group:
                if other is candidate:
                    continue
                offsets_better = all(
                    other.delivery_offsets_s[box_id]
                    <= candidate.delivery_offsets_s[box_id] + 1e-9
                    for box_id in candidate.box_ids
                )
                if (
                    offsets_better
                    and other.duration_s <= candidate.duration_s + 1e-9
                    and other.energy_kwh <= candidate.energy_kwh + 1e-12
                    and (
                        other.duration_s < candidate.duration_s - 1e-9
                        or other.energy_kwh < candidate.energy_kwh - 1e-12
                        or any(
                            other.delivery_offsets_s[box_id]
                            < candidate.delivery_offsets_s[box_id] - 1e-9
                            for box_id in candidate.box_ids
                        )
                    )
                ):
                    dominated = True
                    break
            if not dominated:
                kept.append(candidate)
    return tuple(sorted(kept, key=lambda plan: plan.signature))


def _candidate_cost(data: Q2Data, candidate: TripPlan) -> float:
    timely = sum(
        data.boxes[box_id].priority_weight
        * candidate.delivery_offsets_s[box_id]
        / data.boxes[box_id].expected_time_s
        for box_id in candidate.box_ids
    )
    return (
        1_000_000.0 * timely
        + 100.0 * candidate.duration_s / 3600.0
        + 10.0 * candidate.energy_kwh
        + 1.0
    )


def _select_candidates(
    data: Q2Data,
    candidates: Sequence[TripPlan],
    no_good_cuts: Sequence[Sequence[int]],
    time_limit_s: float,
) -> Tuple[Tuple[TripPlan, ...], object]:
    box_ids = tuple(sorted(data.boxes))
    box_index = {box_id: index for index, box_id in enumerate(box_ids)}
    cover = np.zeros((len(box_ids), len(candidates)), dtype=float)
    for column, candidate in enumerate(candidates):
        for box_id in candidate.box_ids:
            cover[box_index[box_id], column] = 1.0
    rows = [csc_matrix(cover)]
    lower = [np.ones(len(box_ids))]
    upper = [np.ones(len(box_ids))]
    for cut in no_good_cuts:
        row = np.zeros((1, len(candidates)), dtype=float)
        row[0, list(cut)] = 1.0
        rows.append(csc_matrix(row))
        lower.append(np.array([-inf]))
        upper.append(np.array([len(cut) - 1.0]))
    matrix = vstack(rows, format="csc")
    constraint = LinearConstraint(matrix, np.concatenate(lower), np.concatenate(upper))
    result = milp(
        c=np.array([_candidate_cost(data, candidate) for candidate in candidates]),
        integrality=np.ones(len(candidates)),
        bounds=Bounds(np.zeros(len(candidates)), np.ones(len(candidates))),
        constraints=constraint,
        options={"time_limit": float(time_limit_s), "mip_rel_gap": 0.0},
    )
    if result.x is None:
        raise InfeasibleQ2Error(f"candidate MILP failed: {result.message}")
    selected_indices = tuple(index for index, value in enumerate(result.x) if value > 0.5)
    return tuple(candidates[index] for index in selected_indices), (result, selected_indices)


def _resource_aware_incumbent(
    data: Q2Data,
    candidates: Sequence[TripPlan],
    *,
    started: float,
    diagnostics: Mapping[str, object] | None = None,
) -> Q2Solution:
    incumbent = construct_resource_aware_boxwise_solution(data, candidates, "candidate")
    merged_diagnostics = dict(incumbent.diagnostics)
    merged_diagnostics.update({"candidate_count": len(candidates)})
    merged_diagnostics.update(diagnostics or {})
    return replace(
        incumbent,
        runtime_s=perf_counter() - started,
        diagnostics=merged_diagnostics,
    )


def solve_candidate_method(
    data: Q2Data,
    arcs: Mapping[Tuple[str, str], ArcGeometry],
    time_limit_s: float = 300.0,
    *,
    candidates: Sequence[TripPlan] | None = None,
) -> Q2Solution:
    started = perf_counter()
    pool = tuple(candidates) if candidates is not None else generate_initial_candidates(data, arcs)
    if not pool:
        raise InfeasibleQ2Error("candidate pool is empty")
    incumbent = _resource_aware_incumbent(data, pool, started=started)
    cuts: List[Tuple[int, ...]] = []
    last_error = ""
    for iteration in range(10):
        remaining = max(1.0, time_limit_s - (perf_counter() - started))
        selected, details = _select_candidates(data, pool, cuts, remaining)
        result, selected_indices = details
        try:
            scheduled = schedule_trips_cp_sat(
                data,
                selected,
                method="candidate",
                time_limit_s=remaining,
            )
        except InfeasibleQ2Error as error:
            last_error = str(error)
            cuts.append(selected_indices)
            continue
        diagnostics = dict(scheduled.diagnostics)
        diagnostics.update(
            {
                "candidate_count": len(pool),
                "selected_candidate_count": len(selected),
                "set_partition_status": int(result.status),
                "set_partition_message": result.message,
                "set_partition_objective": float(result.fun),
                "no_good_cut_count": len(cuts),
            }
        )
        exact = replace(
            scheduled,
            runtime_s=perf_counter() - started,
            diagnostics=diagnostics,
        )
        if exact.objective < incumbent.objective:
            return exact
        return _resource_aware_incumbent(
            data,
            pool,
            started=started,
            diagnostics={
                "selected_candidate_count": len(selected),
                "set_partition_status": int(result.status),
                "set_partition_message": result.message,
                "set_partition_objective": float(result.fun),
                "no_good_cut_count": len(cuts),
                "incumbent_source": "resource_aware_bootstrap",
            },
        )
    return _resource_aware_incumbent(
        data,
        pool,
        started=started,
        diagnostics={
            "no_good_cut_count": len(cuts),
            "incumbent_source": "resource_aware_bootstrap",
            "exact_search_failure": last_error,
        },
    )
