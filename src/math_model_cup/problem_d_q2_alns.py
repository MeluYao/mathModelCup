"""Reproducible minimum viable adaptive large-neighborhood search for Q2."""

from __future__ import annotations

from dataclasses import replace
from random import Random
from time import perf_counter
from typing import Mapping, Sequence, Tuple

from .problem_d_q2 import InfeasibleQ2Error, Q2Data, Q2Solution, TripDraft, TripPlan, TripStop
from .problem_d_q2_candidates import solve_candidate_method
from .problem_d_q2_geometry import ArcGeometry
from .problem_d_q2_physics import evaluate_trip
from .problem_d_q2_schedule import schedule_trips_greedy


def _merge_stops(left: TripPlan, right: TripPlan, reverse: bool) -> Tuple[TripStop, ...]:
    ordered = (right, left) if reverse else (left, right)
    by_service = {}
    service_order = []
    for plan in ordered:
        for stop in plan.stops:
            if stop.service_id not in by_service:
                service_order.append(stop.service_id)
                by_service[stop.service_id] = []
            by_service[stop.service_id].extend(stop.box_ids)
    return tuple(
        TripStop(service_id, tuple(by_service[service_id]))
        for service_id in service_order
    )


def solve_alns(
    data: Q2Data,
    arcs: Mapping[Tuple[str, str], ArcGeometry],
    seed: int = 0,
    iterations: int = 10_000,
    *,
    initial_solution: Q2Solution | None = None,
) -> Q2Solution:
    started = perf_counter()
    rng = Random(seed)
    initial = initial_solution or solve_candidate_method(data, arcs, time_limit_s=30.0)
    current = replace(initial, method="alns")
    plans = [trip.plan for trip in current.trips]
    accepted = 0
    attempted = 0

    for _ in range(max(0, iterations)):
        if len(plans) < 2:
            break
        left_index, right_index = sorted(rng.sample(range(len(plans)), 2))
        left = plans[left_index]
        right = plans[right_index]
        if left.model_id != right.model_id:
            continue
        attempted += 1
        merged_options = []
        for reverse in (False, True):
            try:
                merged_options.append(
                    evaluate_trip(
                        data,
                        arcs,
                        TripDraft(left.model_id, _merge_stops(left, right, reverse)),
                    )
                )
            except InfeasibleQ2Error:
                continue
        if not merged_options:
            continue
        merged = min(
            merged_options,
            key=lambda plan: (plan.duration_s, plan.energy_kwh, plan.signature),
        )
        candidate_plans = [
            plan
            for index, plan in enumerate(plans)
            if index not in (left_index, right_index)
        ] + [merged]
        try:
            candidate = schedule_trips_greedy(data, candidate_plans, method="alns")
        except InfeasibleQ2Error:
            continue
        if candidate.objective < current.objective:
            current = candidate
            plans = [trip.plan for trip in current.trips]
            accepted += 1

    diagnostics = dict(current.diagnostics)
    diagnostics.update(
        {
            "seed": seed,
            "iterations_requested": iterations,
            "merge_attempts": attempted,
            "accepted_moves": accepted,
            "operator_weights": {"pair_merge": 1.0},
        }
    )
    return replace(
        current,
        method="alns",
        runtime_s=perf_counter() - started,
        diagnostics=diagnostics,
    )
