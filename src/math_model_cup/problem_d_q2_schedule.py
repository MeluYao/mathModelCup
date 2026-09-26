"""Greedy and CP-SAT resource scheduling for fixed Q2 trip plans."""

from __future__ import annotations

from dataclasses import replace
from math import ceil, floor, inf
from time import perf_counter
from typing import Dict, Iterable, List, Sequence, Tuple

from ortools.sat.python import cp_model

from .problem_d_q2 import (
    BatteryUnit,
    DeliveryRecord,
    InfeasibleQ2Error,
    Q2Data,
    Q2Solution,
    TripExecution,
    TripPlan,
)
from .problem_d_q2_physics import charge_time_s, objective_vector
from .problem_d_q2_urgent import (
    PUBLISHED_MODE,
    URGENT_PRIORITY_MODE,
    is_urgent_box,
    required_deadline_s,
    urgent_objective_vector,
)


def _deadline(data: Q2Data, box_id: str, evaluation_mode: str) -> float | None:
    box = data.boxes[box_id]
    if evaluation_mode == URGENT_PRIORITY_MODE:
        return required_deadline_s(box)
    if evaluation_mode != PUBLISHED_MODE:
        raise ValueError(f"unknown Q2 evaluation mode: {evaluation_mode}")
    return box.hard_deadline_s


def _latest_start(
    data: Q2Data,
    plan: TripPlan,
    evaluation_mode: str = PUBLISHED_MODE,
) -> float:
    limits = [
        deadline - plan.delivery_offsets_s[box_id]
        for box_id in plan.box_ids
        for deadline in (_deadline(data, box_id, evaluation_mode),)
        if deadline is not None
    ]
    return min(limits) if limits else inf


def _hard_deadlines_hold(
    data: Q2Data,
    plan: TripPlan,
    start_time_s: float,
    evaluation_mode: str = PUBLISHED_MODE,
) -> bool:
    return all(
        deadline is None
        or start_time_s + plan.delivery_offsets_s[box_id] <= deadline + 1e-9
        for box_id in plan.box_ids
        for deadline in (_deadline(data, box_id, evaluation_mode),)
    )


def _make_solution(
    data: Q2Data,
    method: str,
    executions: Iterable[TripExecution],
    runtime_s: float,
    solver_status: str,
    diagnostics: Dict[str, object] | None = None,
    evaluation_mode: str = PUBLISHED_MODE,
) -> Q2Solution:
    ordered = tuple(sorted(executions, key=lambda trip: (trip.start_time_s, trip.trip_id)))
    deliveries = tuple(
        DeliveryRecord(
            box_id=box_id,
            trip_id=trip.trip_id,
            service_id=data.boxes[box_id].service_id,
            delivery_time_s=trip.start_time_s + trip.plan.delivery_offsets_s[box_id],
        )
        for trip in ordered
        for box_id in trip.plan.box_ids
    )
    provisional = Q2Solution(
        method=method,
        trips=ordered,
        deliveries=deliveries,
        objective=(0.0, 0.0, 0.0, 0),
        runtime_s=runtime_s,
        solver_status=solver_status,
        diagnostics=diagnostics or {},
    )
    objective = (
        urgent_objective_vector(data, provisional)
        if evaluation_mode == URGENT_PRIORITY_MODE
        else objective_vector(data, provisional)
    )
    return replace(provisional, objective=objective)


def schedule_trips_greedy(
    data: Q2Data,
    plans: Sequence[TripPlan],
    method: str = "greedy_schedule",
    *,
    enforce_hard_deadlines: bool = True,
    evaluation_mode: str = PUBLISHED_MODE,
) -> Q2Solution:
    """Schedule urgent trips first at the earliest compatible resource time."""
    started = perf_counter()
    aircraft_available = {unit.aircraft_id: 0.0 for unit in data.aircraft_units}
    battery_available = {battery.battery_id: 0.0 for battery in data.batteries}
    batteries = {battery.battery_id: battery for battery in data.batteries}
    ordered_indices = sorted(
        range(len(plans)),
        key=lambda index: (
            min(
                (
                    _deadline(data, box_id, evaluation_mode)
                    for box_id in plans[index].box_ids
                    if _deadline(data, box_id, evaluation_mode) is not None
                ),
                default=inf,
            ),
            min(
                data.boxes[box_id].expected_time_s
                for box_id in plans[index].box_ids
            ),
            index,
        ),
    )
    ordered_plans = [plans[index] for index in ordered_indices]

    executions: List[TripExecution] = []
    for index, plan in enumerate(ordered_plans, start=1):
        compatible_aircraft = [
            unit for unit in data.aircraft_units if unit.model_id == plan.model_id
        ]
        compatible_batteries = [
            battery for battery in data.batteries if battery.model_id == plan.model_id
        ]
        choices = []
        for unit in compatible_aircraft:
            for battery in compatible_batteries:
                start = max(
                    aircraft_available[unit.aircraft_id],
                    battery_available[battery.battery_id],
                )
                if not enforce_hard_deadlines or _hard_deadlines_hold(
                    data, plan, start, evaluation_mode
                ):
                    choices.append((start, unit.aircraft_id, battery.battery_id))
        if not choices:
            raise InfeasibleQ2Error(
                f"no resource assignment can meet hard deadlines for {plan.signature}"
            )
        start, aircraft_id, battery_id = min(choices)
        return_time = start + plan.duration_s
        recharge = charge_time_s(
            plan.return_soc_percent / 100.0,
            batteries[battery_id].full_charge_time_s,
        )
        battery_ready = return_time + recharge
        aircraft_available[aircraft_id] = return_time
        battery_available[battery_id] = battery_ready
        executions.append(
            TripExecution(
                trip_id=f"T{index:03d}",
                plan=plan,
                aircraft_id=aircraft_id,
                battery_id=battery_id,
                start_time_s=start,
                return_time_s=return_time,
                battery_ready_time_s=battery_ready,
            )
        )

    return _make_solution(
        data,
        method,
        executions,
        perf_counter() - started,
        "FEASIBLE",
        evaluation_mode=evaluation_mode,
    )


def construct_resource_aware_boxwise_solution(
    data: Q2Data,
    candidates: Sequence[TripPlan],
    method: str,
    *,
    evaluation_mode: str = PUBLISHED_MODE,
) -> Q2Solution:
    """Build an independent feasible incumbent from single-box alternatives."""
    started = perf_counter()
    alternatives: Dict[str, List[TripPlan]] = {box_id: [] for box_id in data.boxes}
    for candidate in candidates:
        if len(candidate.box_ids) == 1:
            alternatives[candidate.box_ids[0]].append(candidate)
    if any(not plans for plans in alternatives.values()):
        raise InfeasibleQ2Error("boxwise construction lacks a candidate for some box")

    aircraft_available = {unit.aircraft_id: 0.0 for unit in data.aircraft_units}
    battery_available = {battery.battery_id: 0.0 for battery in data.batteries}
    battery_by_id: Dict[str, BatteryUnit] = {
        battery.battery_id: battery for battery in data.batteries
    }
    ordered_box_ids = sorted(
        data.boxes,
        key=lambda box_id: (
            _deadline(data, box_id, evaluation_mode)
            if _deadline(data, box_id, evaluation_mode) is not None
            else inf,
            data.boxes[box_id].expected_time_s,
            -data.boxes[box_id].priority_weight,
            box_id,
        ),
    )
    executions: List[TripExecution] = []
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
                    deadline = _deadline(data, box_id, evaluation_mode)
                    if deadline is not None and delivery_time > deadline + 1e-9:
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
            raise InfeasibleQ2Error(
                f"boxwise construction misses hard deadline for {box_id}"
            )
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
        method,
        executions,
        perf_counter() - started,
        "FEASIBLE",
        diagnostics={"initialization": "independent_resource_aware"},
        evaluation_mode=evaluation_mode,
    )


def schedule_trips_cp_sat(
    data: Q2Data,
    plans: Sequence[TripPlan],
    method: str = "cp_sat_schedule",
    time_limit_s: float = 60.0,
    *,
    evaluation_mode: str = PUBLISHED_MODE,
) -> Q2Solution:
    """Assign fixed trip plans to aircraft and batteries with exact no-overlap."""
    started = perf_counter()
    if not plans:
        return _make_solution(
            data, method, (), 0.0, "OPTIMAL", evaluation_mode=evaluation_mode
        )

    battery_by_id = {battery.battery_id: battery for battery in data.batteries}
    max_full_charge = max(battery.full_charge_time_s for battery in data.batteries)
    horizon = int(
        ceil(
            sum(plan.duration_s + max_full_charge for plan in plans)
            + max(
                (
                    _deadline(data, box_id, evaluation_mode) or 0.0
                    for box_id in data.boxes
                ),
                default=0.0,
            )
            + 1.0
        )
    )
    model = cp_model.CpModel()
    starts = []
    ends = []
    battery_ends = []
    aircraft_assignments: List[Dict[str, cp_model.IntVar]] = []
    battery_assignments: List[Dict[str, cp_model.IntVar]] = []
    aircraft_intervals: Dict[str, List[cp_model.IntervalVar]] = {
        unit.aircraft_id: [] for unit in data.aircraft_units
    }
    battery_intervals: Dict[str, List[cp_model.IntervalVar]] = {
        battery.battery_id: [] for battery in data.batteries
    }

    for index, plan in enumerate(plans):
        duration = int(ceil(plan.duration_s))
        compatible_batteries = [
            battery for battery in data.batteries if battery.model_id == plan.model_id
        ]
        if not compatible_batteries:
            raise InfeasibleQ2Error(f"no battery for model {plan.model_id}")
        recharge = int(
            ceil(
                charge_time_s(
                    plan.return_soc_percent / 100.0,
                    compatible_batteries[0].full_charge_time_s,
                )
            )
        )
        start = model.new_int_var(0, horizon, f"start_{index}")
        end = model.new_int_var(0, horizon, f"end_{index}")
        battery_end = model.new_int_var(0, horizon, f"battery_end_{index}")
        model.add(end == start + duration)
        model.add(battery_end == start + duration + recharge)
        starts.append(start)
        ends.append(end)
        battery_ends.append(battery_end)

        aircraft_bools = {}
        for unit in data.aircraft_units:
            if unit.model_id != plan.model_id:
                continue
            present = model.new_bool_var(f"trip_{index}_aircraft_{unit.aircraft_id}")
            interval = model.new_optional_interval_var(
                start, duration, end, present, f"aircraft_interval_{index}_{unit.aircraft_id}"
            )
            aircraft_bools[unit.aircraft_id] = present
            aircraft_intervals[unit.aircraft_id].append(interval)
        if not aircraft_bools:
            raise InfeasibleQ2Error(f"no aircraft for model {plan.model_id}")
        model.add_exactly_one(aircraft_bools.values())
        aircraft_assignments.append(aircraft_bools)

        battery_bools = {}
        for battery in compatible_batteries:
            present = model.new_bool_var(f"trip_{index}_battery_{battery.battery_id}")
            interval = model.new_optional_interval_var(
                start,
                duration + recharge,
                battery_end,
                present,
                f"battery_interval_{index}_{battery.battery_id}",
            )
            battery_bools[battery.battery_id] = present
            battery_intervals[battery.battery_id].append(interval)
        model.add_exactly_one(battery_bools.values())
        battery_assignments.append(battery_bools)

        for box_id in plan.box_ids:
            deadline = _deadline(data, box_id, evaluation_mode)
            if deadline is not None:
                latest = floor(deadline - plan.delivery_offsets_s[box_id] + 1e-9)
                if latest < 0:
                    raise InfeasibleQ2Error(f"trip cannot meet deadline for {box_id}")
                model.add(start <= latest)

    for intervals in aircraft_intervals.values():
        if intervals:
            model.add_no_overlap(intervals)
    for intervals in battery_intervals.values():
        if intervals:
            model.add_no_overlap(intervals)

    makespan = model.new_int_var(0, horizon, "makespan")
    model.add_max_equality(makespan, ends)
    priority_start = sum(
        int(
            sum(
                data.boxes[box_id].priority_weight
                * (
                    1_000_000.0 / data.boxes[box_id].expected_time_s
                    if evaluation_mode == URGENT_PRIORITY_MODE
                    and is_urgent_box(data.boxes[box_id])
                    else 1.0
                )
                for box_id in plan.box_ids
                if evaluation_mode != URGENT_PRIORITY_MODE
                or is_urgent_box(data.boxes[box_id])
            )
        )
        * starts[index]
        for index, plan in enumerate(plans)
    )
    if evaluation_mode == URGENT_PRIORITY_MODE:
        model.minimize(priority_start * (horizon + 1) + makespan)
    else:
        model.minimize(makespan * 100_000 + priority_start)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_s)
    solver.parameters.num_search_workers = 8
    status = solver.solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise InfeasibleQ2Error("CP-SAT could not find a feasible resource schedule")

    executions = []
    for index, plan in enumerate(plans, start=1):
        array_index = index - 1
        start = float(solver.value(starts[array_index]))
        aircraft_id = next(
            resource_id
            for resource_id, variable in aircraft_assignments[array_index].items()
            if solver.value(variable)
        )
        battery_id = next(
            resource_id
            for resource_id, variable in battery_assignments[array_index].items()
            if solver.value(variable)
        )
        return_time = start + plan.duration_s
        battery_ready = return_time + charge_time_s(
            plan.return_soc_percent / 100.0,
            battery_by_id[battery_id].full_charge_time_s,
        )
        executions.append(
            TripExecution(
                trip_id=f"T{index:03d}",
                plan=plan,
                aircraft_id=aircraft_id,
                battery_id=battery_id,
                start_time_s=start,
                return_time_s=return_time,
                battery_ready_time_s=battery_ready,
            )
        )

    status_name = "OPTIMAL" if status == cp_model.OPTIMAL else "FEASIBLE"
    return _make_solution(
        data,
        method,
        executions,
        perf_counter() - started,
        status_name,
        diagnostics={
            "cp_sat_objective": solver.objective_value,
            "cp_sat_best_bound": solver.best_objective_bound,
        },
        evaluation_mode=evaluation_mode,
    )
