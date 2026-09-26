"""Communication-aware joint search primitives for Problem D, question 3.

This program and code were completed with assistance from OpenAI Codex.
The authors reviewed and validated the resulting implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor, inf
from time import perf_counter
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csc_matrix, vstack
from ortools.sat.python import cp_model

from .problem_d_q2 import (
    InfeasibleQ2Error,
    Q2Data,
    Q2Solution,
    TripExecution,
    TripPlan,
)
from .problem_d_q2_candidates import generate_initial_candidates
from .problem_d_q2_geometry import ArcGeometry
from .problem_d_q2_physics import charge_time_s
from .problem_d_q2_schedule import (
    _make_solution,
    construct_resource_aware_boxwise_solution,
    schedule_trips_greedy,
)
from .problem_d_q3 import (
    CommunicationSegment,
    Q3Data,
    Q3Objective,
    Q3Solution,
    RelayState,
)
from .problem_d_q3_relay_candidates import CoverageAtlas


@dataclass(frozen=True)
class RelayBatchTemplate:
    batch_id: str
    state_id: str
    segment_ids: tuple[str, ...]


def relay_batches_from_solution(solution: Q3Solution) -> tuple[RelayBatchTemplate, ...]:
    """Freeze the state/segment grouping of a feasible relaxed relay schedule."""
    state_by_sortie = {
        sortie.relay_sortie_id: sortie.state_id for sortie in solution.relay_sorties
    }
    segment_ids_by_sortie: dict[str, list[str]] = {
        sortie_id: [] for sortie_id in state_by_sortie
    }
    for assignment in solution.communication_assignments:
        if assignment.relay_sortie_id is not None:
            segment_ids_by_sortie[assignment.relay_sortie_id].append(
                assignment.segment_id
            )
    return tuple(
        RelayBatchTemplate(
            batch_id=sortie_id,
            state_id=state_by_sortie[sortie_id],
            segment_ids=tuple(sorted(segment_ids_by_sortie[sortie_id])),
        )
        for sortie_id in sorted(state_by_sortie)
        if segment_ids_by_sortie[sortie_id]
    )


def reschedule_transport_with_relay_batches(
    data: Q3Data,
    transport: Q2Solution,
    communication_segments: Sequence[CommunicationSegment],
    relay_states: Sequence[RelayState],
    batches: Sequence[RelayBatchTemplate],
    *,
    time_limit_s: float = 60.0,
) -> Q2Solution:
    """Shift fixed transport routes while respecting relay-batch capacity.

    Segment-to-state and segment-to-batch assignments are held fixed, while
    all transport starts and compatible transport resource assignments remain
    decision variables.  A batch is one continuously hovering relay sortie.
    """
    started = perf_counter()
    trip_by_id = {trip.trip_id: trip for trip in transport.trips}
    segment_by_id = {segment.segment_id: segment for segment in communication_segments}
    state_by_id = {state.state_id: state for state in relay_states}
    if not batches:
        return transport
    unknown_segments = sorted(
        segment_id
        for batch in batches
        for segment_id in batch.segment_ids
        if segment_id not in segment_by_id
    )
    if unknown_segments:
        raise ValueError(f"relay batches reference unknown segments: {unknown_segments[:3]}")
    unknown_states = sorted(
        {batch.state_id for batch in batches if batch.state_id not in state_by_id}
    )
    if unknown_states:
        raise ValueError(f"relay batches reference unknown states: {unknown_states}")

    model = cp_model.CpModel()
    horizon = int(
        ceil(
            sum(trip.plan.duration_s for trip in transport.trips)
            + sum(battery.full_charge_time_s for battery in data.transport.batteries)
            + 40_000.0
        )
    )
    starts: dict[str, cp_model.IntVar] = {}
    ends: dict[str, cp_model.IntVar] = {}
    aircraft_intervals = {
        unit.aircraft_id: [] for unit in data.transport.aircraft_units
    }
    battery_intervals = {
        unit.battery_id: [] for unit in data.transport.batteries
    }
    aircraft_choice: dict[tuple[str, str], cp_model.IntVar] = {}
    battery_choice: dict[tuple[str, str], cp_model.IntVar] = {}
    battery_by_id = {
        battery.battery_id: battery for battery in data.transport.batteries
    }
    for trip in transport.trips:
        trip_id = trip.trip_id
        duration = int(ceil(trip.plan.duration_s))
        start = model.new_int_var(0, horizon, f"start_{trip_id}")
        end = model.new_int_var(0, horizon, f"end_{trip_id}")
        model.add(end == start + duration)
        starts[trip_id] = start
        ends[trip_id] = end

        compatible_aircraft = [
            unit
            for unit in data.transport.aircraft_units
            if unit.model_id == trip.plan.model_id
        ]
        air_bools = []
        for unit in compatible_aircraft:
            assigned = model.new_bool_var(f"air_{trip_id}_{unit.aircraft_id}")
            aircraft_choice[(trip_id, unit.aircraft_id)] = assigned
            aircraft_intervals[unit.aircraft_id].append(
                model.new_optional_interval_var(
                    start,
                    duration,
                    end,
                    assigned,
                    f"air_interval_{trip_id}_{unit.aircraft_id}",
                )
            )
            air_bools.append(assigned)
        model.add_exactly_one(air_bools)

        compatible_batteries = [
            battery
            for battery in data.transport.batteries
            if battery.model_id == trip.plan.model_id
        ]
        battery_bools = []
        for battery in compatible_batteries:
            recharge = int(
                ceil(
                    charge_time_s(
                        trip.plan.return_soc_percent / 100.0,
                        battery.full_charge_time_s,
                    )
                )
            )
            battery_end = model.new_int_var(
                0, horizon, f"battery_end_{trip_id}_{battery.battery_id}"
            )
            model.add(battery_end == start + duration + recharge)
            assigned = model.new_bool_var(
                f"battery_{trip_id}_{battery.battery_id}"
            )
            battery_choice[(trip_id, battery.battery_id)] = assigned
            battery_intervals[battery.battery_id].append(
                model.new_optional_interval_var(
                    start,
                    duration + recharge,
                    battery_end,
                    assigned,
                    f"battery_interval_{trip_id}_{battery.battery_id}",
                )
            )
            battery_bools.append(assigned)
        model.add_exactly_one(battery_bools)

        for box_id in trip.plan.box_ids:
            deadline = data.transport.boxes[box_id].hard_deadline_s
            if deadline is not None:
                model.add(
                    start
                    <= floor(deadline - trip.plan.delivery_offsets_s[box_id] + 1e-9)
                )

    for intervals in aircraft_intervals.values():
        model.add_no_overlap(intervals)
    for intervals in battery_intervals.values():
        model.add_no_overlap(intervals)

    relay_intervals = []
    energy_intervals = []
    for batch in batches:
        state = state_by_id[batch.state_id]
        members = [segment_by_id[segment_id] for segment_id in batch.segment_ids]
        service_starts = []
        service_ends = []
        for segment in members:
            original_trip = trip_by_id[segment.transport_trip_id]
            service_starts.append(
                starts[segment.transport_trip_id]
                + floor(segment.start_time_s - original_trip.start_time_s)
            )
            service_ends.append(
                starts[segment.transport_trip_id]
                + ceil(segment.end_time_s - original_trip.start_time_s)
            )
        service_start = model.new_int_var(0, horizon, f"service_start_{batch.batch_id}")
        service_end = model.new_int_var(0, horizon, f"service_end_{batch.batch_id}")
        model.add_min_equality(service_start, service_starts)
        model.add_max_equality(service_end, service_ends)
        outbound = state.round_trip_time_s / 2.0
        lead = int(
            ceil(
                data.relay_model.preparation_time_s
                + outbound
                + data.relay_model.link_setup_time_s
            )
        )
        tail = int(ceil(outbound + data.relay_model.turnaround_time_s))
        relay_start = model.new_int_var(0, horizon, f"relay_start_{batch.batch_id}")
        relay_end = model.new_int_var(0, horizon, f"relay_end_{batch.batch_id}")
        relay_duration = model.new_int_var(1, horizon, f"relay_duration_{batch.batch_id}")
        model.add(relay_start == service_start - lead)
        model.add(relay_end == service_end + tail)
        model.add(relay_duration == relay_end - relay_start)
        relay_intervals.append(
            model.new_interval_var(
                relay_start,
                relay_duration,
                relay_end,
                f"relay_interval_{batch.batch_id}",
            )
        )

        service_power_kw = (
            data.relay_model.hover_power_kw
            + data.relay_model.communication_power_kw
        )
        usable_service_energy = (
            (1.0 - data.relay_model.reserve_ratio)
            * data.relay_model.usable_energy_kwh
            - state.round_trip_energy_kwh
        )
        maximum_service_s = floor(usable_service_energy * 3600.0 / service_power_kw)
        model.add(service_end - service_start <= maximum_service_s)

        energy_end = model.new_int_var(0, horizon, f"energy_end_{batch.batch_id}")
        energy_duration = model.new_int_var(1, horizon, f"energy_duration_{batch.batch_id}")
        model.add(
            energy_end
            == service_end + int(ceil(outbound + data.relay_model.full_charge_time_s))
        )
        model.add(energy_duration == energy_end - relay_start)
        energy_intervals.append(
            model.new_interval_var(
                relay_start,
                energy_duration,
                energy_end,
                f"energy_interval_{batch.batch_id}",
            )
        )

    model.add_cumulative(
        relay_intervals,
        [1] * len(relay_intervals),
        len(data.relay_units),
    )
    model.add_cumulative(
        energy_intervals,
        [1] * len(energy_intervals),
        len(data.energy_units),
    )
    makespan = model.new_int_var(0, horizon, "makespan")
    model.add_max_equality(makespan, tuple(ends.values()))
    weighted_starts = []
    for trip in transport.trips:
        weight = max(
            1,
            int(
                round(
                    10_000
                    * sum(
                        data.transport.boxes[box_id].priority_weight
                        / data.transport.boxes[box_id].expected_time_s
                        for box_id in trip.plan.box_ids
                    )
                )
            ),
        )
        weighted_starts.append(weight * starts[trip.trip_id])
        model.add_hint(starts[trip.trip_id], int(round(trip.start_time_s)))
        for unit in data.transport.aircraft_units:
            variable = aircraft_choice.get((trip.trip_id, unit.aircraft_id))
            if variable is not None:
                model.add_hint(variable, int(unit.aircraft_id == trip.aircraft_id))
        for battery in data.transport.batteries:
            variable = battery_choice.get((trip.trip_id, battery.battery_id))
            if variable is not None:
                model.add_hint(variable, int(battery.battery_id == trip.battery_id))
    model.minimize(sum(weighted_starts) * 10_000 + makespan)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_s)
    solver.parameters.num_search_workers = 8
    solver.parameters.random_seed = 20260925
    status = solver.solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise InfeasibleQ2Error(
            f"relay-batch transport rescheduling failed: {solver.status_name(status)}"
        )

    executions = []
    for trip in transport.trips:
        trip_id = trip.trip_id
        start_time = float(solver.value(starts[trip_id]))
        aircraft_id = next(
            unit.aircraft_id
            for unit in data.transport.aircraft_units
            if (trip_id, unit.aircraft_id) in aircraft_choice
            and solver.value(aircraft_choice[(trip_id, unit.aircraft_id)])
        )
        battery_id = next(
            battery.battery_id
            for battery in data.transport.batteries
            if (trip_id, battery.battery_id) in battery_choice
            and solver.value(battery_choice[(trip_id, battery.battery_id)])
        )
        return_time = start_time + trip.plan.duration_s
        battery = battery_by_id[battery_id]
        battery_ready = return_time + charge_time_s(
            trip.plan.return_soc_percent / 100.0,
            battery.full_charge_time_s,
        )
        executions.append(
            TripExecution(
                trip_id,
                trip.plan,
                aircraft_id,
                battery_id,
                start_time,
                return_time,
                battery_ready,
            )
        )
    return _make_solution(
        data.transport,
        "q3_relay_batch_reschedule",
        executions,
        perf_counter() - started,
        "OPTIMAL" if status == cp_model.OPTIMAL else "FEASIBLE",
        diagnostics={
            "relay_batch_count": len(batches),
            "relay_capacity": len(data.relay_units),
            "energy_capacity": len(data.energy_units),
        },
    )


def reschedule_transport_with_state_grid(
    data: Q3Data,
    transport: Q2Solution,
    communication_segments: Sequence[CommunicationSegment],
    relay_states: Sequence[RelayState],
    atlas: CoverageAtlas,
    *,
    grid_step_s: float = 60.0,
    soft_grid_step_s: float = 300.0,
    occupancy_slot_s: float = 30.0,
    transition_buffer_s: float = 0.0,
    soft_horizon_s: float = 30_000.0,
    time_limit_s: float = 60.0,
    stop_after_first_solution: bool = True,
) -> Q2Solution:
    """Choose transport starts and whole-trip relay states on a time grid.

    The model limits the number of simultaneously occupied relay states.  A
    configurable buffer approximates travel and turnaround; the exact relay
    scheduler remains the final feasibility oracle.
    """
    if min(grid_step_s, soft_grid_step_s, occupancy_slot_s) <= 0.0:
        raise ValueError("time-grid steps must be positive")
    if transition_buffer_s < 0.0:
        raise ValueError("transition_buffer_s must be non-negative")
    started = perf_counter()
    scale = 10
    grid_step = max(1, int(round(grid_step_s * scale)))
    soft_grid_step = max(1, int(round(soft_grid_step_s * scale)))
    occupancy_slot = max(1, int(round(occupancy_slot_s * scale)))
    buffer = int(ceil(transition_buffer_s * scale))
    horizon = int(ceil((soft_horizon_s + 20_000.0) * scale))
    dark_segments = tuple(
        segment for segment in communication_segments if not segment.direct_available
    )
    dark_by_trip: dict[str, list[CommunicationSegment]] = {}
    for segment in dark_segments:
        dark_by_trip.setdefault(segment.transport_trip_id, []).append(segment)
    state_by_id = {state.state_id: state for state in relay_states}

    model = cp_model.CpModel()
    starts: dict[str, cp_model.IntVar] = {}
    ends: dict[str, cp_model.IntVar] = {}
    aircraft_intervals: dict[str, list[cp_model.IntervalVar]] = {
        unit.aircraft_id: [] for unit in data.transport.aircraft_units
    }
    battery_intervals: dict[str, list[cp_model.IntervalVar]] = {
        battery.battery_id: [] for battery in data.transport.batteries
    }
    battery_by_id = {
        battery.battery_id: battery for battery in data.transport.batteries
    }
    for trip in transport.trips:
        duration = int(ceil(trip.plan.duration_s * scale))
        start = model.new_int_var(0, horizon, f"grid_start_{trip.trip_id}")
        end = model.new_int_var(0, horizon, f"grid_end_{trip.trip_id}")
        model.add(end == start + duration)
        starts[trip.trip_id] = start
        ends[trip.trip_id] = end
        aircraft_intervals[trip.aircraft_id].append(
            model.new_interval_var(start, duration, end, f"grid_air_{trip.trip_id}")
        )
        battery = battery_by_id[trip.battery_id]
        battery_duration = int(
            ceil(
                (
                    trip.plan.duration_s
                    + charge_time_s(
                        trip.plan.return_soc_percent / 100.0,
                        battery.full_charge_time_s,
                    )
                )
                * scale
            )
        )
        battery_end = model.new_int_var(
            0, horizon, f"grid_battery_end_{trip.trip_id}"
        )
        model.add(battery_end == start + battery_duration)
        battery_intervals[trip.battery_id].append(
            model.new_interval_var(
                start,
                battery_duration,
                battery_end,
                f"grid_battery_{trip.trip_id}",
            )
        )
        for box_id in trip.plan.box_ids:
            deadline = data.transport.boxes[box_id].hard_deadline_s
            if deadline is not None:
                model.add(
                    start
                    <= floor(
                        (deadline - trip.plan.delivery_offsets_s[box_id])
                        * scale
                        + 1e-9
                    )
                )
    for intervals in aircraft_intervals.values():
        model.add_no_overlap(intervals)
    for intervals in battery_intervals.values():
        model.add_no_overlap(intervals)

    occupancy_choices: dict[tuple[str, int], list[cp_model.IntVar]] = {}
    trip_choices: dict[str, list[tuple[cp_model.IntVar, int, str]]] = {}
    for trip in transport.trips:
        members = dark_by_trip.get(trip.trip_id, [])
        if not members:
            continue
        valid_states = [
            state_id
            for state_id in state_by_id
            if all(
                state_id in atlas.segment_to_states.get(segment.segment_id, ())
                for segment in members
            )
        ]
        if not valid_states:
            raise InfeasibleQ2Error(
                f"no single relay state covers trip {trip.trip_id}"
            )
        latest_start = min(
            (
                data.transport.boxes[box_id].hard_deadline_s
                - trip.plan.delivery_offsets_s[box_id]
                for box_id in trip.plan.box_ids
                if data.transport.boxes[box_id].hard_deadline_s is not None
            ),
            default=None,
        )
        maximum_start = int(
            floor(
                (soft_horizon_s if latest_start is None else latest_start)
                * scale
                + 1e-9
            )
        )
        step = soft_grid_step if latest_start is None else grid_step
        candidate_starts = set(range(0, maximum_start + 1, step))
        original_start = int(ceil(trip.start_time_s * scale))
        candidate_starts.update(
            (original_start, original_start + 1, original_start + scale)
        )
        candidate_starts = {
            value for value in candidate_starts if 0 <= value <= maximum_start
        }
        first_offset = floor(
            min(segment.start_time_s - trip.start_time_s for segment in members)
            * scale
        )
        last_offset = ceil(
            max(segment.end_time_s - trip.start_time_s for segment in members)
            * scale
        )
        choices = []
        for candidate_start in sorted(candidate_starts):
            for state_id in valid_states:
                variable = model.new_bool_var(
                    f"grid_choice_{trip.trip_id}_{candidate_start}_{state_id}"
                )
                choices.append((variable, candidate_start, state_id))
                left = candidate_start + first_offset - buffer
                right = candidate_start + last_offset + buffer
                for slot in range(
                    floor(left / occupancy_slot),
                    ceil(right / occupancy_slot),
                ):
                    occupancy_choices.setdefault((state_id, slot), []).append(variable)
        model.add_exactly_one(variable for variable, _, _ in choices)
        model.add(
            starts[trip.trip_id]
            == sum(candidate_start * variable for variable, candidate_start, _ in choices)
        )
        trip_choices[trip.trip_id] = choices

    occupied_state_by_slot: dict[int, list[cp_model.IntVar]] = {}
    for (state_id, slot), choices in occupancy_choices.items():
        occupied = model.new_bool_var(f"occupied_{state_id}_{slot}")
        for choice in choices:
            model.add(occupied >= choice)
        model.add(occupied <= sum(choices))
        occupied_state_by_slot.setdefault(slot, []).append(occupied)
    for occupied_states in occupied_state_by_slot.values():
        model.add(sum(occupied_states) <= len(data.relay_units))

    makespan = model.new_int_var(0, horizon, "grid_makespan")
    model.add_max_equality(makespan, tuple(ends.values()))
    weighted_starts = []
    for trip in transport.trips:
        weight = max(
            1,
            int(
                round(
                    10_000
                    * sum(
                        data.transport.boxes[box_id].priority_weight
                        / data.transport.boxes[box_id].expected_time_s
                        for box_id in trip.plan.box_ids
                    )
                )
            ),
        )
        weighted_starts.append(weight * starts[trip.trip_id])
        model.add_hint(starts[trip.trip_id], int(ceil(trip.start_time_s * scale)))
    model.minimize(sum(weighted_starts) * 10_000 + makespan)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_s)
    solver.parameters.num_search_workers = 8
    solver.parameters.random_seed = 20260925
    solver.parameters.stop_after_first_solution = bool(stop_after_first_solution)
    status = solver.solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise InfeasibleQ2Error(
            f"state-grid transport rescheduling failed: {solver.status_name(status)}"
        )

    executions = []
    selected_states = {}
    for trip in transport.trips:
        start_time = solver.value(starts[trip.trip_id]) / scale
        return_time = start_time + trip.plan.duration_s
        battery = battery_by_id[trip.battery_id]
        battery_ready_time = return_time + charge_time_s(
            trip.plan.return_soc_percent / 100.0,
            battery.full_charge_time_s,
        )
        executions.append(
            TripExecution(
                trip.trip_id,
                trip.plan,
                trip.aircraft_id,
                trip.battery_id,
                start_time,
                return_time,
                battery_ready_time,
            )
        )
        if trip.trip_id in trip_choices:
            selected_states[trip.trip_id] = next(
                state_id
                for variable, _, state_id in trip_choices[trip.trip_id]
                if solver.value(variable)
            )
    return _make_solution(
        data.transport,
        "q3_state_grid_reschedule",
        executions,
        perf_counter() - started,
        "OPTIMAL" if status == cp_model.OPTIMAL else "FEASIBLE",
        diagnostics={
            "grid_step_s": grid_step_s,
            "occupancy_slot_s": occupancy_slot_s,
            "transition_buffer_s": transition_buffer_s,
            "selected_relay_states": selected_states,
        },
    )


def dominates(left: Q3Objective, right: Q3Objective) -> bool:
    left_values = left.as_tuple()
    right_values = right.as_tuple()
    return all(a <= b for a, b in zip(left_values, right_values)) and any(
        a < b for a, b in zip(left_values, right_values)
    )


def _grouping_cost(data: Q2Data, plan: TripPlan) -> float:
    timeliness = sum(
        data.boxes[box_id].priority_weight
        * plan.delivery_offsets_s[box_id]
        / data.boxes[box_id].expected_time_s
        for box_id in plan.box_ids
    )
    # One additional sortie dominates all secondary differences.  The
    # secondary score keeps the selected cover deterministic and timely.
    return 1_000_000_000.0 + 10_000.0 * timeliness + plan.energy_kwh


def _select_cover(
    data: Q2Data,
    candidates: Sequence[TripPlan],
    cuts: Sequence[Sequence[int]],
    time_limit_s: float,
) -> tuple[tuple[TripPlan, ...], tuple[int, ...]]:
    box_ids = tuple(sorted(data.boxes))
    box_index = {box_id: index for index, box_id in enumerate(box_ids)}
    cover = np.zeros((len(box_ids), len(candidates)), dtype=float)
    for column, candidate in enumerate(candidates):
        for box_id in candidate.box_ids:
            cover[box_index[box_id], column] = 1.0
    matrices = [csc_matrix(cover)]
    lower = [np.ones(len(box_ids))]
    upper = [np.ones(len(box_ids))]
    for cut in cuts:
        row = np.zeros((1, len(candidates)), dtype=float)
        row[0, list(cut)] = 1.0
        matrices.append(csc_matrix(row))
        lower.append(np.array([-inf]))
        upper.append(np.array([len(cut) - 1.0]))
    constraints = LinearConstraint(
        vstack(matrices, format="csc"),
        np.concatenate(lower),
        np.concatenate(upper),
    )
    result = milp(
        c=np.asarray([_grouping_cost(data, plan) for plan in candidates]),
        integrality=np.ones(len(candidates)),
        bounds=Bounds(np.zeros(len(candidates)), np.ones(len(candidates))),
        constraints=constraints,
        options={"time_limit": max(1.0, float(time_limit_s)), "mip_rel_gap": 0.0},
    )
    if result.x is None:
        raise InfeasibleQ2Error(f"grouped transport MILP failed: {result.message}")
    selected_indices = tuple(
        index for index, value in enumerate(result.x) if value > 0.5
    )
    return tuple(candidates[index] for index in selected_indices), selected_indices


def select_grouped_transport(
    data: Q2Data,
    arcs: Mapping[tuple[str, str], ArcGeometry],
    *,
    max_stops: int = 2,
    time_limit_s: float = 30.0,
    candidates: Sequence[TripPlan] | None = None,
    max_candidates: int = 1_200,
) -> Q2Solution:
    """Jointly select and schedule routes with transport-sortie count first."""
    started = perf_counter()
    pool = tuple(candidates) if candidates is not None else generate_initial_candidates(
        data, arcs, max_stops=max_stops
    )
    mandatory = [plan for plan in pool if len(plan.box_ids) == 1]
    optional = sorted(
        (plan for plan in pool if len(plan.box_ids) != 1),
        key=lambda plan: (_grouping_cost(data, plan), plan.signature),
    )
    plans = tuple(mandatory + optional[: max(0, max_candidates - len(mandatory))])
    bootstrap = construct_resource_aware_boxwise_solution(
        data, plans, "q3_grouped_bootstrap"
    )
    model = cp_model.CpModel()
    horizon = int(
        ceil(
            sum(
                max(
                    (plan.duration_s for plan in plans if box_id in plan.box_ids),
                    default=0.0,
                )
                for box_id in data.boxes
            )
            + sum(battery.full_charge_time_s for battery in data.batteries)
            + 20_000.0
        )
    )
    selected = []
    starts = []
    ends = []
    aircraft_assignments = []
    battery_assignments = []
    aircraft_intervals = {unit.aircraft_id: [] for unit in data.aircraft_units}
    battery_intervals = {battery.battery_id: [] for battery in data.batteries}
    recharge_by_index = []
    for index, plan in enumerate(plans):
        use = model.new_bool_var(f"use_{index}")
        start = model.new_int_var(0, horizon, f"start_{index}")
        end = model.new_int_var(0, horizon, f"end_{index}")
        duration = int(ceil(plan.duration_s))
        model.add(end == start + duration).only_enforce_if(use)
        model.add(start == 0).only_enforce_if(use.negated())
        model.add(end == 0).only_enforce_if(use.negated())
        selected.append(use)
        starts.append(start)
        ends.append(end)

        air_bools = {}
        for unit in data.aircraft_units:
            if unit.model_id != plan.model_id:
                continue
            assigned = model.new_bool_var(f"air_{index}_{unit.aircraft_id}")
            model.add(assigned <= use)
            aircraft_intervals[unit.aircraft_id].append(
                model.new_optional_interval_var(
                    start, duration, end, assigned, f"air_i_{index}_{unit.aircraft_id}"
                )
            )
            air_bools[unit.aircraft_id] = assigned
        model.add(sum(air_bools.values()) == use)
        aircraft_assignments.append(air_bools)

        compatible_batteries = [
            battery for battery in data.batteries if battery.model_id == plan.model_id
        ]
        recharge = int(
            ceil(
                charge_time_s(
                    plan.return_soc_percent / 100.0,
                    compatible_batteries[0].full_charge_time_s,
                )
            )
        )
        recharge_by_index.append(recharge)
        battery_end = model.new_int_var(0, horizon, f"battery_end_{index}")
        model.add(battery_end == start + duration + recharge).only_enforce_if(use)
        model.add(battery_end == 0).only_enforce_if(use.negated())
        battery_bools = {}
        for battery in compatible_batteries:
            assigned = model.new_bool_var(f"battery_{index}_{battery.battery_id}")
            model.add(assigned <= use)
            battery_intervals[battery.battery_id].append(
                model.new_optional_interval_var(
                    start,
                    duration + recharge,
                    battery_end,
                    assigned,
                    f"battery_i_{index}_{battery.battery_id}",
                )
            )
            battery_bools[battery.battery_id] = assigned
        model.add(sum(battery_bools.values()) == use)
        battery_assignments.append(battery_bools)

        for box_id in plan.box_ids:
            deadline = data.boxes[box_id].hard_deadline_s
            if deadline is not None:
                latest = floor(deadline - plan.delivery_offsets_s[box_id] + 1e-9)
                if latest < 0:
                    model.add(use == 0)
                else:
                    model.add(start <= latest).only_enforce_if(use)

    for box_id in sorted(data.boxes):
        model.add(
            sum(
                selected[index]
                for index, plan in enumerate(plans)
                if box_id in plan.box_ids
            )
            == 1
        )
    for intervals in aircraft_intervals.values():
        model.add_no_overlap(intervals)
    for intervals in battery_intervals.values():
        model.add_no_overlap(intervals)
    makespan = model.new_int_var(0, horizon, "makespan")
    model.add_max_equality(makespan, ends)
    timing_terms = []
    for index, plan in enumerate(plans):
        weight = int(
            round(
                10_000
                * sum(
                    data.boxes[box_id].priority_weight
                    / data.boxes[box_id].expected_time_s
                    for box_id in plan.box_ids
                )
            )
        )
        internal = int(
            round(
                10_000
                * sum(
                    data.boxes[box_id].priority_weight
                    * plan.delivery_offsets_s[box_id]
                    / data.boxes[box_id].expected_time_s
                    for box_id in plan.box_ids
                )
            )
        )
        timing_terms.append(weight * starts[index] + internal * selected[index])
    model.minimize(
        sum(selected) * 1_000_000_000_000
        + sum(timing_terms) * 10_000
        + makespan
    )
    bootstrap_by_signature = {
        execution.plan.signature: execution for execution in bootstrap.trips
    }
    for index, plan in enumerate(plans):
        execution = bootstrap_by_signature.get(plan.signature)
        model.add_hint(selected[index], 1 if execution is not None else 0)
        if execution is None:
            model.add_hint(starts[index], 0)
            continue
        model.add_hint(starts[index], int(round(execution.start_time_s)))
        for identifier, variable in aircraft_assignments[index].items():
            model.add_hint(variable, int(identifier == execution.aircraft_id))
        for identifier, variable in battery_assignments[index].items():
            model.add_hint(variable, int(identifier == execution.battery_id))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_s)
    solver.parameters.num_search_workers = 8
    solver.parameters.random_seed = 20260925
    status = solver.solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        current = bootstrap
        pair_candidates = sorted(
            (plan for plan in pool if len(plan.box_ids) == 2),
            key=lambda plan: (
                min(
                    data.boxes[box_id].hard_deadline_s
                    if data.boxes[box_id].hard_deadline_s is not None
                    else inf
                    for box_id in plan.box_ids
                ),
                plan.duration_s,
                plan.energy_kwh,
                plan.signature,
            ),
        )
        accepted_merges = 0
        for pair in pair_candidates:
            owner = {
                box_id: index
                for index, execution in enumerate(current.trips)
                for box_id in execution.plan.box_ids
            }
            left, right = pair.box_ids
            left_index = owner[left]
            right_index = owner[right]
            if left_index == right_index:
                continue
            if (
                len(current.trips[left_index].plan.box_ids) != 1
                or len(current.trips[right_index].plan.box_ids) != 1
            ):
                continue
            retained = [
                execution.plan
                for index, execution in enumerate(current.trips)
                if index not in (left_index, right_index)
            ]
            try:
                candidate_solution = schedule_trips_greedy(
                    data,
                    retained + [pair],
                    method="q3_grouped_transport",
                )
            except InfeasibleQ2Error:
                continue
            current = candidate_solution
            accepted_merges += 1
        diagnostics = dict(current.diagnostics)
        diagnostics.update(
            {
                "candidate_count": len(plans),
                "generated_candidate_count": len(pool),
                "objective_priority": "greedy_feasible_pair_merges",
                "integrated_status": solver.status_name(status),
                "accepted_pair_merges": accepted_merges,
            }
        )
        from dataclasses import replace

        return replace(current, diagnostics=diagnostics)

    battery_by_id = {battery.battery_id: battery for battery in data.batteries}
    executions = []
    for index, plan in enumerate(plans):
        if not solver.value(selected[index]):
            continue
        aircraft_id = next(
            identifier for identifier, variable in aircraft_assignments[index].items()
            if solver.value(variable)
        )
        battery_id = next(
            identifier for identifier, variable in battery_assignments[index].items()
            if solver.value(variable)
        )
        start_time = float(solver.value(starts[index]))
        return_time = start_time + plan.duration_s
        battery_ready = return_time + charge_time_s(
            plan.return_soc_percent / 100.0,
            battery_by_id[battery_id].full_charge_time_s,
        )
        executions.append(
            TripExecution(
                trip_id=f"T{len(executions) + 1:03d}",
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
        "q3_grouped_transport",
        executions,
        perf_counter() - started,
        "OPTIMAL" if status == cp_model.OPTIMAL else "FEASIBLE",
        diagnostics={
            "candidate_count": len(plans),
            "generated_candidate_count": len(pool),
            "objective_priority": "trip_count_then_timeliness",
        },
    )


def schedule_hard_then_serial_soft(
    data: Q2Data,
    plans: Sequence[TripPlan],
    *,
    soft_gap_s: float = 900.0,
) -> Q2Solution:
    """Protect hard deadlines first, then serialize deadline-free trips."""
    if soft_gap_s < 0.0:
        raise ValueError("soft_gap_s must be non-negative")
    hard_plans = [
        plan
        for plan in plans
        if any(data.boxes[box_id].hard_deadline_s is not None for box_id in plan.box_ids)
    ]
    soft_plans = [plan for plan in plans if plan not in hard_plans]
    hard_solution = schedule_trips_greedy(
        data, hard_plans, method="q3_hard_then_serial_soft"
    )
    executions: list[TripExecution] = []
    aircraft_available = {unit.aircraft_id: 0.0 for unit in data.aircraft_units}
    battery_available = {battery.battery_id: 0.0 for battery in data.batteries}
    battery_by_id = {battery.battery_id: battery for battery in data.batteries}
    for trip in hard_solution.trips:
        executions.append(
            TripExecution(
                trip_id=f"T{len(executions) + 1:03d}",
                plan=trip.plan,
                aircraft_id=trip.aircraft_id,
                battery_id=trip.battery_id,
                start_time_s=trip.start_time_s,
                return_time_s=trip.return_time_s,
                battery_ready_time_s=trip.battery_ready_time_s,
            )
        )
        aircraft_available[trip.aircraft_id] = max(
            aircraft_available[trip.aircraft_id], trip.return_time_s
        )
        battery_available[trip.battery_id] = max(
            battery_available[trip.battery_id], trip.battery_ready_time_s
        )
    serial_available = max(
        [trip.return_time_s for trip in executions],
        default=0.0,
    ) + soft_gap_s
    ordered_soft = sorted(
        soft_plans,
        key=lambda plan: (
            -sum(data.boxes[box_id].priority_weight for box_id in plan.box_ids),
            plan.duration_s,
            plan.signature,
        ),
    )
    for plan in ordered_soft:
        choices = []
        for aircraft in data.aircraft_units:
            if aircraft.model_id != plan.model_id:
                continue
            for battery in data.batteries:
                if battery.model_id != plan.model_id:
                    continue
                start_time = max(
                    serial_available,
                    aircraft_available[aircraft.aircraft_id],
                    battery_available[battery.battery_id],
                )
                choices.append((start_time, aircraft.aircraft_id, battery.battery_id))
        if not choices:
            raise InfeasibleQ2Error(f"no resources for soft plan {plan.signature}")
        start_time, aircraft_id, battery_id = min(choices)
        return_time = start_time + plan.duration_s
        battery_ready = return_time + charge_time_s(
            plan.return_soc_percent / 100.0,
            battery_by_id[battery_id].full_charge_time_s,
        )
        executions.append(
            TripExecution(
                trip_id=f"T{len(executions) + 1:03d}",
                plan=plan,
                aircraft_id=aircraft_id,
                battery_id=battery_id,
                start_time_s=start_time,
                return_time_s=return_time,
                battery_ready_time_s=battery_ready,
            )
        )
        aircraft_available[aircraft_id] = return_time
        battery_available[battery_id] = battery_ready
        serial_available = return_time + soft_gap_s
    return _make_solution(
        data,
        "q3_hard_then_serial_soft",
        executions,
        0.0,
        "FEASIBLE",
        diagnostics={
            "hard_trip_count": len(hard_plans),
            "soft_trip_count": len(soft_plans),
            "soft_gap_s": soft_gap_s,
        },
    )
