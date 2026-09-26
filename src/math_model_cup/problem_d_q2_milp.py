"""Restricted integrated route-selection and scheduling model for Q2."""

from __future__ import annotations

from dataclasses import replace
from math import ceil, floor
from time import perf_counter
from typing import Dict, Mapping, Tuple

from ortools.sat.python import cp_model

from .problem_d_q2 import InfeasibleQ2Error, Q2Data, Q2Solution, TripExecution
from .problem_d_q2_candidates import generate_initial_candidates
from .problem_d_q2_geometry import ArcGeometry
from .problem_d_q2_incumbent import ensure_valid_initial_solution, finalize_seeded_solution
from .problem_d_q2_physics import charge_time_s
from .problem_d_q2_schedule import (
    _make_solution,
    construct_resource_aware_boxwise_solution,
)


def _integrated_candidate_key(data: Q2Data, candidate) -> tuple:
    timing = sum(
        data.boxes[box_id].priority_weight
        * candidate.delivery_offsets_s[box_id]
        / data.boxes[box_id].expected_time_s
        for box_id in candidate.box_ids
    )
    return timing, candidate.duration_s, candidate.energy_kwh, candidate.signature


def solve_integrated_milp(
    data: Q2Data,
    arcs: Mapping[Tuple[str, str], ArcGeometry],
    time_limit_s: float = 1800.0,
    max_stops: int = 3,
    max_candidates: int = 1_200,
    *,
    initial_solution: Q2Solution | None = None,
) -> Q2Solution:
    """Jointly select candidate routes and schedule both reusable resources."""
    started = perf_counter()
    ensure_valid_initial_solution(data, arcs, initial_solution)
    generated_candidates = generate_initial_candidates(data, arcs, max_stops=max_stops)
    seed_executions = {
        trip.plan.signature: trip for trip in (initial_solution.trips if initial_solution else ())
    }
    all_candidates = {candidate.signature: candidate for candidate in generated_candidates}
    for signature, execution in seed_executions.items():
        all_candidates[signature] = execution.plan
    mandatory = [candidate for candidate in all_candidates.values() if len(candidate.box_ids) == 1]
    seed_optional = [
        execution.plan
        for execution in seed_executions.values()
        if len(execution.plan.box_ids) != 1
    ]
    seed_signatures = set(seed_executions)
    optional = [
        candidate
        for candidate in all_candidates.values()
        if len(candidate.box_ids) != 1 and candidate.signature not in seed_signatures
    ]
    optional.sort(key=lambda candidate: _integrated_candidate_key(data, candidate))
    protected = mandatory + seed_optional
    candidates = tuple(
        protected + optional[: max(0, max_candidates - len(protected))]
    )
    bootstrap = construct_resource_aware_boxwise_solution(
        data, candidates, "integrated_milp"
    )
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

    for index, candidate in enumerate(candidates):
        execution = seed_executions.get(candidate.signature)
        model.add_hint(selected[index], int(execution is not None))
        if execution is None:
            continue
        model.add_hint(starts[index], max(0, int(round(execution.start_time_s))))
        for aircraft_id, variable in aircraft_assignments[index].items():
            model.add_hint(variable, int(aircraft_id == execution.aircraft_id))
        for battery_id, variable in battery_assignments[index].items():
            model.add_hint(variable, int(battery_id == execution.battery_id))

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
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 0
    phase_objectives = (
        ("timing", sum(timing_terms)),
        ("makespan", makespan),
        ("energy", sum(energy_terms)),
        ("trip_count", sum(selected)),
    )
    phase_statuses: dict[str, str] = {}
    phase_values: dict[str, int] = {}
    snapshot: list[tuple[int, float, str, str]] | None = None
    status = cp_model.UNKNOWN
    solve_started = perf_counter()
    battery_by_id = {battery.battery_id: battery for battery in data.batteries}

    for phase_index, (phase_name, expression) in enumerate(phase_objectives):
        remaining = max(0.01, float(time_limit_s) - (perf_counter() - solve_started))
        phases_left = len(phase_objectives) - phase_index
        solver.parameters.max_time_in_seconds = max(0.01, remaining / phases_left)
        model.minimize(expression)
        phase_status = solver.solve(model)
        phase_statuses[phase_name] = solver.status_name(phase_status)
        if phase_status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            break
        status = phase_status
        phase_value = int(solver.value(expression))
        phase_values[phase_name] = phase_value
        snapshot = []
        for index, candidate in enumerate(candidates):
            if not solver.value(selected[index]):
                continue
            aircraft_id = next(
                key
                for key, variable in aircraft_assignments[index].items()
                if solver.value(variable)
            )
            battery_id = next(
                key
                for key, variable in battery_assignments[index].items()
                if solver.value(variable)
            )
            snapshot.append(
                (index, float(solver.value(starts[index])), aircraft_id, battery_id)
            )
        if phase_index + 1 < len(phase_objectives):
            model.add(expression == phase_value)

    if snapshot is None:
        diagnostics = dict(bootstrap.diagnostics)
        diagnostics.update({
            "backend": "OR-Tools CP-SAT restricted integrated model",
            "integrated_status": phase_statuses.get("timing", "NOT_RUN"),
            "incumbent_source": "integrated_primal_heuristic",
            "generated_candidate_count": len(generated_candidates),
            "candidate_count": len(candidates),
        })
        return finalize_seeded_solution(
            "integrated_milp",
            bootstrap,
            initial_solution,
            started=started,
            native_source="integrated_primal_heuristic",
            diagnostics={
                **diagnostics,
                "seed_candidate_count": len(seed_executions),
                "seed_hint_count": len(seed_executions),
                "lexicographic_phases_completed": 0,
                "lexicographic_phase_statuses": phase_statuses,
            },
        )

    executions = []
    for trip_number, (index, start_time, aircraft_id, battery_id) in enumerate(
        snapshot, start=1
    ):
        candidate = candidates[index]
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
        (
            "OPTIMAL"
            if len(phase_values) == len(phase_objectives)
            and all(value == "OPTIMAL" for value in phase_statuses.values())
            else "FEASIBLE"
        ),
        diagnostics={
            "backend": "OR-Tools CP-SAT restricted integrated model",
            "candidate_count": len(candidates),
            "generated_candidate_count": len(generated_candidates),
            "lexicographic_phases_completed": len(phase_values),
            "lexicographic_phase_statuses": phase_statuses,
            "lexicographic_phase_values": phase_values,
        },
    )
    return finalize_seeded_solution(
        "integrated_milp",
        solution,
        initial_solution,
        started=started,
        native_source="integrated_cp_sat",
        diagnostics={
            **solution.diagnostics,
            "seed_candidate_count": len(seed_executions),
            "seed_hint_count": len(seed_executions),
        },
    )
