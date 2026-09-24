"""Restricted integrated route-selection and scheduling model for Q2."""

from __future__ import annotations

from dataclasses import replace
from math import ceil, floor
from time import perf_counter
from typing import Dict, Mapping, Tuple

from ortools.sat.python import cp_model

from .problem_d_q2 import InfeasibleQ2Error, Q2Data, Q2Solution, TripExecution
from .problem_d_q2_candidates import generate_initial_candidates, solve_candidate_method
from .problem_d_q2_geometry import ArcGeometry
from .problem_d_q2_physics import charge_time_s
from .problem_d_q2_schedule import _make_solution


def solve_integrated_milp(
    data: Q2Data,
    arcs: Mapping[Tuple[str, str], ArcGeometry],
    time_limit_s: float = 1800.0,
    max_stops: int = 3,
) -> Q2Solution:
    """Jointly select candidate routes and schedule both reusable resources."""
    started = perf_counter()
    candidates = generate_initial_candidates(data, arcs, max_stops=max_stops)
    model = cp_model.CpModel()
    horizon = int(
        ceil(
            sum(
                max(
                    (candidate.duration_s for candidate in candidates if box_id in candidate.box_ids),
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
    aircraft_intervals: Dict[str, list] = {
        unit.aircraft_id: [] for unit in data.aircraft_units
    }
    battery_intervals: Dict[str, list] = {
        battery.battery_id: [] for battery in data.batteries
    }

    for index, candidate in enumerate(candidates):
        use = model.new_bool_var(f"use_{index}")
        start = model.new_int_var(0, horizon, f"start_{index}")
        end = model.new_int_var(0, horizon, f"end_{index}")
        duration = int(ceil(candidate.duration_s))
        model.add(end == start + duration).only_enforce_if(use)
        model.add(start == 0).only_enforce_if(use.negated())
        model.add(end == 0).only_enforce_if(use.negated())
        selected.append(use)
        starts.append(start)
        ends.append(end)

        air_bools = {}
        for unit in data.aircraft_units:
            if unit.model_id != candidate.model_id:
                continue
            assigned = model.new_bool_var(f"air_{index}_{unit.aircraft_id}")
            model.add(assigned <= use)
            interval = model.new_optional_interval_var(
                start, duration, end, assigned, f"air_interval_{index}_{unit.aircraft_id}"
            )
            aircraft_intervals[unit.aircraft_id].append(interval)
            air_bools[unit.aircraft_id] = assigned
        model.add(sum(air_bools.values()) == use)
        aircraft_assignments.append(air_bools)

        compatible_batteries = [
            battery for battery in data.batteries if battery.model_id == candidate.model_id
        ]
        recharge = int(
            ceil(
                charge_time_s(
                    candidate.return_soc_percent / 100.0,
                    compatible_batteries[0].full_charge_time_s,
                )
            )
        )
        block_end = model.new_int_var(0, horizon, f"battery_end_{index}")
        model.add(block_end == start + duration + recharge).only_enforce_if(use)
        model.add(block_end == 0).only_enforce_if(use.negated())
        battery_bools = {}
        for battery in compatible_batteries:
            assigned = model.new_bool_var(f"battery_{index}_{battery.battery_id}")
            model.add(assigned <= use)
            interval = model.new_optional_interval_var(
                start,
                duration + recharge,
                block_end,
                assigned,
                f"battery_interval_{index}_{battery.battery_id}",
            )
            battery_intervals[battery.battery_id].append(interval)
            battery_bools[battery.battery_id] = assigned
        model.add(sum(battery_bools.values()) == use)
        battery_assignments.append(battery_bools)

        for box_id in candidate.box_ids:
            deadline = data.boxes[box_id].hard_deadline_s
            if deadline is not None:
                latest = floor(deadline - candidate.delivery_offsets_s[box_id] + 1e-9)
                if latest < 0:
                    model.add(use == 0)
                else:
                    model.add(start <= latest).only_enforce_if(use)

    for box_id in sorted(data.boxes):
        model.add(
            sum(
                selected[index]
                for index, candidate in enumerate(candidates)
                if box_id in candidate.box_ids
            )
            == 1
        )
    for intervals in aircraft_intervals.values():
        if intervals:
            model.add_no_overlap(intervals)
    for intervals in battery_intervals.values():
        if intervals:
            model.add_no_overlap(intervals)

    makespan = model.new_int_var(0, horizon, "makespan")
    model.add_max_equality(makespan, ends)
    timing_terms = []
    energy_terms = []
    for index, candidate in enumerate(candidates):
        start_weight = int(
            round(
                10_000
                * sum(
                    data.boxes[box_id].priority_weight
                    / data.boxes[box_id].expected_time_s
                    for box_id in candidate.box_ids
                )
            )
        )
        internal = int(
            round(
                10_000
                * sum(
                    data.boxes[box_id].priority_weight
                    * candidate.delivery_offsets_s[box_id]
                    / data.boxes[box_id].expected_time_s
                    for box_id in candidate.box_ids
                )
            )
        )
        timing_terms.append(start_weight * starts[index] + internal * selected[index])
        energy_terms.append(int(round(candidate.energy_kwh * 1000)) * selected[index])
    model.minimize(
        sum(timing_terms) * 1_000_000
        + makespan * 10_000
        + sum(energy_terms) * 10
        + sum(selected)
    )

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_s)
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 0
    status = solver.solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        fallback = solve_candidate_method(data, arcs, time_limit_s=max(5.0, time_limit_s))
        diagnostics = dict(fallback.diagnostics)
        diagnostics.update(
            {
                "backend": "OR-Tools CP-SAT restricted integrated model",
                "integrated_status": solver.status_name(status),
                "fallback": "candidate_method",
            }
        )
        return replace(
            fallback,
            method="integrated_milp",
            runtime_s=perf_counter() - started,
            solver_status="FEASIBLE_FALLBACK",
            diagnostics=diagnostics,
        )

    battery_by_id = {battery.battery_id: battery for battery in data.batteries}
    executions = []
    trip_number = 0
    for index, candidate in enumerate(candidates):
        if not solver.value(selected[index]):
            continue
        trip_number += 1
        start_time = float(solver.value(starts[index]))
        aircraft_id = next(
            key for key, variable in aircraft_assignments[index].items() if solver.value(variable)
        )
        battery_id = next(
            key for key, variable in battery_assignments[index].items() if solver.value(variable)
        )
        return_time = start_time + candidate.duration_s
        battery_ready = return_time + charge_time_s(
            candidate.return_soc_percent / 100.0,
            battery_by_id[battery_id].full_charge_time_s,
        )
        executions.append(
            TripExecution(
                trip_id=f"T{trip_number:03d}",
                plan=candidate,
                aircraft_id=aircraft_id,
                battery_id=battery_id,
                start_time_s=start_time,
                return_time_s=return_time,
                battery_ready_time_s=battery_ready,
            )
        )
    solution = _make_solution(
        data,
        "integrated_milp",
        executions,
        perf_counter() - started,
        "OPTIMAL" if status == cp_model.OPTIMAL else "FEASIBLE",
        diagnostics={
            "backend": "OR-Tools CP-SAT restricted integrated model",
            "candidate_count": len(candidates),
            "objective_bound": solver.best_objective_bound,
            "solver_objective": solver.objective_value,
        },
    )
    return solution
