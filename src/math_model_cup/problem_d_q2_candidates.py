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
    BatteryUnit,
    InfeasibleQ2Error,
    Q2Data,
    Q2Solution,
    TripDraft,
    TripExecution,
    TripPlan,
    TripStop,
)
from .problem_d_q2_geometry import ArcGeometry
from .problem_d_q2_physics import charge_time_s, evaluate_trip
from .problem_d_q2_schedule import _make_solution, schedule_trips_cp_sat


def _try_plan(
    data: Q2Data,
    arcs: Mapping[Tuple[str, str], ArcGeometry],
    draft: TripDraft,
) -> TripPlan | None:
    try:
        return evaluate_trip(data, arcs, draft)
    except InfeasibleQ2Error:
        return None


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
    max_stops: int = 3,
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

    if max_stops >= 2:
        for left_index, left in enumerate(seed_plans):
            for right in seed_plans[left_index + 1 :]:
                if left.model_id != right.model_id:
                    continue
                if left.stops[0].service_id == right.stops[0].service_id:
                    continue
                for stops in (
                    (left.stops[0], right.stops[0]),
                    (right.stops[0], left.stops[0]),
                ):
                    plan = _try_plan(data, arcs, TripDraft(left.model_id, stops))
                    if plan is not None:
                        candidates.append(plan)

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


def _resource_aware_single_box_fallback(
    data: Q2Data,
    candidates: Sequence[TripPlan],
    *,
    started: float,
    failure_reason: str,
) -> Q2Solution:
    alternatives: Dict[str, List[TripPlan]] = {box_id: [] for box_id in data.boxes}
    for candidate in candidates:
        if len(candidate.box_ids) == 1:
            alternatives[candidate.box_ids[0]].append(candidate)
    if any(not plans for plans in alternatives.values()):
        raise InfeasibleQ2Error("single-box fallback lacks a candidate for some box")

    aircraft_available = {unit.aircraft_id: 0.0 for unit in data.aircraft_units}
    battery_available = {battery.battery_id: 0.0 for battery in data.batteries}
    battery_by_id: Dict[str, BatteryUnit] = {
        battery.battery_id: battery for battery in data.batteries
    }
    ordered_box_ids = sorted(
        data.boxes,
        key=lambda box_id: (
            data.boxes[box_id].hard_deadline_s
            if data.boxes[box_id].hard_deadline_s is not None
            else inf,
            data.boxes[box_id].expected_time_s,
            -data.boxes[box_id].priority_weight,
            box_id,
        ),
    )
    executions = []
    for index, box_id in enumerate(ordered_box_ids, start=1):
        choices = []
        box = data.boxes[box_id]
        for plan in alternatives[box_id]:
            for unit in data.aircraft_units:
                if unit.model_id != plan.model_id:
                    continue
                for battery in data.batteries:
                    if battery.model_id != plan.model_id:
                        continue
                    start_time = max(
                        aircraft_available[unit.aircraft_id],
                        battery_available[battery.battery_id],
                    )
                    delivery_time = start_time + plan.delivery_offsets_s[box_id]
                    if box.hard_deadline_s is not None and delivery_time > box.hard_deadline_s + 1e-9:
                        continue
                    choices.append(
                        (
                            delivery_time,
                            start_time + plan.duration_s,
                            plan.energy_kwh,
                            start_time,
                            unit.aircraft_id,
                            battery.battery_id,
                            plan,
                        )
                    )
        if not choices:
            raise InfeasibleQ2Error(f"single-box fallback misses hard deadline for {box_id}")
        _, return_time, _, start_time, aircraft_id, battery_id, plan = min(choices)
        battery_ready = return_time + charge_time_s(
            plan.return_soc_percent / 100.0,
            battery_by_id[battery_id].full_charge_time_s,
        )
        aircraft_available[aircraft_id] = return_time
        battery_available[battery_id] = battery_ready
        executions.append(
            TripExecution(
                trip_id=f"T{index:03d}",
                plan=plan,
                aircraft_id=aircraft_id,
                battery_id=battery_id,
                start_time_s=start_time,
                return_time_s=return_time,
                battery_ready_time_s=battery_ready,
            )
        )
    return _make_solution(
        data,
        "candidate",
        executions,
        perf_counter() - started,
        "FEASIBLE_FALLBACK",
        diagnostics={
            "candidate_count": len(candidates),
            "fallback": "resource_aware_single_box",
            "fallback_reason": failure_reason,
        },
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
        return replace(
            scheduled,
            runtime_s=perf_counter() - started,
            diagnostics=diagnostics,
        )
    return _resource_aware_single_box_fallback(
        data,
        pool,
        started=started,
        failure_reason=f"candidate scheduling failed after 10 cuts: {last_error}",
    )
