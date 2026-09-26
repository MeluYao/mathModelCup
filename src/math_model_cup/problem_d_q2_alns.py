"""Reproducible adaptive large-neighborhood search for problem D question 2."""

from __future__ import annotations

from dataclasses import replace
from math import exp
from random import Random
from time import perf_counter
from typing import Dict, Mapping, Sequence, Tuple

from .problem_d_q2 import InfeasibleQ2Error, Q2Data, Q2Solution, TripDraft, TripPlan, TripStop
from .problem_d_q2_candidates import generate_initial_candidates
from .problem_d_q2_geometry import ArcGeometry
from .problem_d_q2_incumbent import ensure_valid_initial_solution, finalize_seeded_solution
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
)


DESTROY_OPERATORS = (
    "random_box",
    "worst_timeliness",
    "high_energy",
    "same_service",
    "whole_trip",
)
REPAIR_OPERATORS = ("minimum_increment", "regret_2")


def _plan_proxy_cost(data: Q2Data, plan: TripPlan) -> float:
    timing = sum(
        data.boxes[box_id].priority_weight
        * plan.delivery_offsets_s[box_id]
        / data.boxes[box_id].expected_time_s
        for box_id in plan.box_ids
    )
    return timing + plan.duration_s / 100_000.0 + plan.energy_kwh / 10_000.0


def _annealing_score(solution: Q2Solution) -> float:
    objective = solution.objective
    return (
        objective[0]
        + objective[1] / 10_000_000.0
        + objective[2] / 1_000_000.0
        + objective[3] / 10_000_000.0
    )


def _weighted_choice(rng: Random, weights: Mapping[str, float]) -> str:
    names = tuple(weights)
    return rng.choices(names, weights=[weights[name] for name in names], k=1)[0]


def _destroy_boxes(
    data: Q2Data,
    solution: Q2Solution,
    operator: str,
    remove_count: int,
    rng: Random,
    evaluation_mode: str = PUBLISHED_MODE,
) -> Tuple[str, ...]:
    box_ids = [record.box_id for record in solution.deliveries]
    if operator == "random_box":
        return tuple(rng.sample(box_ids, min(remove_count, len(box_ids))))
    if operator == "worst_timeliness":
        ranked = sorted(
            solution.deliveries,
            key=lambda record: (
                1
                if evaluation_mode == URGENT_PRIORITY_MODE
                and is_urgent_box(data.boxes[record.box_id])
                else 0,
                data.boxes[record.box_id].priority_weight
                * record.delivery_time_s
                / data.boxes[record.box_id].expected_time_s
            ),
            reverse=True,
        )
        return tuple(record.box_id for record in ranked[:remove_count])
    if operator == "high_energy":
        ranked_trips = sorted(
            solution.trips,
            key=lambda trip: trip.plan.energy_kwh / max(1, len(trip.plan.box_ids)),
            reverse=True,
        )
        selected = [box_id for trip in ranked_trips for box_id in trip.plan.box_ids]
        return tuple(selected[:remove_count])
    if operator == "same_service":
        anchor = rng.choice(box_ids)
        service_id = data.boxes[anchor].service_id
        related = [box_id for box_id in box_ids if data.boxes[box_id].service_id == service_id]
        rng.shuffle(related)
        if len(related) < remove_count:
            remaining = [box_id for box_id in box_ids if box_id not in related]
            rng.shuffle(remaining)
            related.extend(remaining)
        return tuple(related[:remove_count])
    trip = rng.choice(solution.trips)
    selected = list(trip.plan.box_ids)
    if len(selected) < remove_count:
        remaining = [box_id for box_id in box_ids if box_id not in selected]
        rng.shuffle(remaining)
        selected.extend(remaining[: remove_count - len(selected)])
    return tuple(selected)


def _remove_from_plans(
    data: Q2Data,
    arcs: Mapping[Tuple[str, str], ArcGeometry],
    plans: Sequence[TripPlan],
    removed_box_ids: Sequence[str],
) -> list[TripPlan]:
    removed = set(removed_box_ids)
    retained = []
    for plan in plans:
        stops = tuple(
            TripStop(stop.service_id, tuple(box for box in stop.box_ids if box not in removed))
            for stop in plan.stops
        )
        stops = tuple(stop for stop in stops if stop.box_ids)
        if stops:
            retained.append(evaluate_trip(data, arcs, TripDraft(plan.model_id, stops)))
    return retained


def _inserted_stops(plan: TripPlan, box_id: str, service_id: str, max_stops: int):
    existing_index = next(
        (index for index, stop in enumerate(plan.stops) if stop.service_id == service_id),
        None,
    )
    if existing_index is not None:
        stops = list(plan.stops)
        stop = stops[existing_index]
        stops[existing_index] = TripStop(stop.service_id, stop.box_ids + (box_id,))
        yield tuple(stops)
    elif len(plan.stops) < max_stops:
        new_stop = TripStop(service_id, (box_id,))
        for position in range(len(plan.stops) + 1):
            yield plan.stops[:position] + (new_stop,) + plan.stops[position:]


def _insertion_options(
    data: Q2Data,
    arcs: Mapping[Tuple[str, str], ArcGeometry],
    plans: Sequence[TripPlan],
    box_id: str,
    single_plans: Mapping[str, Sequence[TripPlan]],
    max_stops: int,
) -> list[tuple[float, list[TripPlan]]]:
    options: list[tuple[float, list[TripPlan]]] = []
    service_id = data.boxes[box_id].service_id
    for index, plan in enumerate(plans):
        old_cost = _plan_proxy_cost(data, plan)
        for stops in _inserted_stops(plan, box_id, service_id, max_stops):
            try:
                inserted = evaluate_trip(data, arcs, TripDraft(plan.model_id, stops))
            except InfeasibleQ2Error:
                continue
            updated = list(plans)
            updated[index] = inserted
            options.append((_plan_proxy_cost(data, inserted) - old_cost, updated))
    for single in single_plans[box_id]:
        options.append((_plan_proxy_cost(data, single) + 0.001, list(plans) + [single]))
    options.sort(key=lambda item: (item[0], tuple(plan.signature for plan in item[1])))
    return options


def _repair_plans(
    data: Q2Data,
    arcs: Mapping[Tuple[str, str], ArcGeometry],
    plans: Sequence[TripPlan],
    removed_box_ids: Sequence[str],
    single_plans: Mapping[str, Sequence[TripPlan]],
    operator: str,
    max_stops: int,
    evaluation_mode: str = PUBLISHED_MODE,
) -> list[TripPlan]:
    repaired = list(plans)
    remaining = list(removed_box_ids)
    while remaining:
        if operator == "minimum_increment":
            box_id = min(
                remaining,
                key=lambda current: (
                    required_deadline_s(data.boxes[current])
                    if evaluation_mode == URGENT_PRIORITY_MODE
                    else data.boxes[current].hard_deadline_s
                    if data.boxes[current].hard_deadline_s is not None
                    else float("inf"),
                    0
                    if evaluation_mode == URGENT_PRIORITY_MODE
                    and is_urgent_box(data.boxes[current])
                    else 1,
                    -data.boxes[current].priority_weight,
                    current,
                ),
            )
            options = _insertion_options(
                data, arcs, repaired, box_id, single_plans, max_stops
            )
        else:
            regret_options = []
            for current in remaining:
                current_options = _insertion_options(
                    data, arcs, repaired, current, single_plans, max_stops
                )
                if not current_options:
                    continue
                second = current_options[1][0] if len(current_options) > 1 else current_options[0][0] + 1.0
                regret_options.append((second - current_options[0][0], current, current_options))
            if not regret_options:
                raise InfeasibleQ2Error("ALNS repair has no insertion option")
            _, box_id, options = max(regret_options, key=lambda item: (item[0], item[1]))
        if not options:
            raise InfeasibleQ2Error(f"ALNS cannot repair box {box_id}")
        repaired = options[0][1]
        remaining.remove(box_id)
    return repaired


def _update_weight(
    weights: Dict[str, float],
    scores: Dict[str, float],
    uses: Dict[str, int],
    reaction: float = 0.20,
) -> None:
    for name in weights:
        if uses[name]:
            observed = max(0.1, scores[name] / uses[name])
            weights[name] = (1.0 - reaction) * weights[name] + reaction * observed
        scores[name] = 0.0
        uses[name] = 0


def _accept_candidate(
    current: Q2Solution,
    candidate: Q2Solution,
    temperature: float,
    random_draw,
) -> bool:
    """Use lexicographic dominance first and anneal only the first worse level."""
    if candidate.objective <= current.objective:
        return True
    for current_value, candidate_value in zip(current.objective, candidate.objective):
        if candidate_value == current_value:
            continue
        scale = max(1.0, abs(float(current_value)))
        delta = (float(candidate_value) - float(current_value)) / scale
        return random_draw() < exp(-delta / max(temperature, 1e-12))
    return True


def solve_alns(
    data: Q2Data,
    arcs: Mapping[Tuple[str, str], ArcGeometry],
    seed: int = 0,
    iterations: int = 10_000,
    *,
    initial_solution: Q2Solution | None = None,
    max_stops: int = 3,
    candidate_pool: Sequence[TripPlan] | None = None,
    evaluation_mode: str = PUBLISHED_MODE,
) -> Q2Solution:
    """Run destroy/repair ALNS with deterministic resource decoding."""
    started = perf_counter()
    ensure_valid_initial_solution(
        data, arcs, initial_solution, evaluation_mode=evaluation_mode
    )
    rng = Random(seed)
    initial_candidates = (
        tuple(candidate_pool)
        if candidate_pool is not None
        else generate_initial_candidates(data, arcs, max_stops=max_stops)
    )
    single_plans: Dict[str, list[TripPlan]] = {box_id: [] for box_id in data.boxes}
    for plan in initial_candidates:
        if len(plan.box_ids) == 1:
            single_plans[plan.box_ids[0]].append(plan)
    if initial_solution is None:
        independent_base = construct_resource_aware_boxwise_solution(
            data,
            initial_candidates,
            "alns",
            evaluation_mode=evaluation_mode,
        )
        initial = independent_base
        initialization = "independent_resource_aware"
    else:
        initial = replace(initial_solution, method="alns")
        initialization = "provided_solution"

    current = initial
    best = initial
    destroy_weights = {name: 1.0 for name in DESTROY_OPERATORS}
    repair_weights = {name: 1.0 for name in REPAIR_OPERATORS}
    destroy_scores = {name: 0.0 for name in DESTROY_OPERATORS}
    repair_scores = {name: 0.0 for name in REPAIR_OPERATORS}
    destroy_uses = {name: 0 for name in DESTROY_OPERATORS}
    repair_uses = {name: 0 for name in REPAIR_OPERATORS}
    accepted = 0
    improving = 0
    feasible_moves = 0
    repair_feasibility_fallbacks = 0
    rejected_infeasible_moves = 0
    restarts = 0
    no_improvement = 0
    history = [(0, best.objective)]
    initial_temperature = max(0.01, 0.05 * _annealing_score(initial))

    for iteration in range(1, max(0, iterations) + 1):
        destroy = _weighted_choice(rng, destroy_weights)
        repair = _weighted_choice(rng, repair_weights)
        destroy_uses[destroy] += 1
        repair_uses[repair] += 1
        remove_count = min(
            len(data.boxes),
            max(1, int(round(len(data.boxes) * rng.uniform(0.02, 0.08)))),
        )
        removed = _destroy_boxes(
            data, current, destroy, remove_count, rng, evaluation_mode
        )
        partial = _remove_from_plans(
            data, arcs, [trip.plan for trip in current.trips], removed
        )
        try:
            repaired = _repair_plans(
                data,
                arcs,
                partial,
                removed,
                single_plans,
                repair,
                max_stops,
                evaluation_mode,
            )
            candidate = schedule_trips_greedy(
                data,
                repaired,
                method="alns",
                evaluation_mode=evaluation_mode,
            )
        except InfeasibleQ2Error:
            safe_repair = list(partial) + [
                min(single_plans[box_id], key=lambda plan: _plan_proxy_cost(data, plan))
                for box_id in removed
            ]
            repair_feasibility_fallbacks += 1
            try:
                candidate = schedule_trips_greedy(
                    data,
                    safe_repair,
                    method="alns",
                    evaluation_mode=evaluation_mode,
                )
            except InfeasibleQ2Error:
                rejected_infeasible_moves += 1
                no_improvement += 1
                continue
        feasible_moves += 1
        fraction = iteration / max(1, iterations)
        temperature = initial_temperature * (0.001 ** fraction)
        delta = _annealing_score(candidate) - _annealing_score(current)
        accepted_move = _accept_candidate(current, candidate, temperature, rng.random)
        reward = 0.0
        if accepted_move:
            current = candidate
            accepted += 1
            reward = 0.5 if delta > 0.0 else 2.0
            if current.objective < best.objective:
                best = current
                improving += 1
                no_improvement = 0
                reward = 5.0
            else:
                no_improvement += 1
        else:
            no_improvement += 1
        destroy_scores[destroy] += reward
        repair_scores[repair] += reward

        if iteration % 50 == 0:
            _update_weight(destroy_weights, destroy_scores, destroy_uses)
            _update_weight(repair_weights, repair_scores, repair_uses)
            history.append((iteration, best.objective))
        if no_improvement >= 250:
            current = best
            no_improvement = 0
            restarts += 1

    if iterations > 0 and history[-1][0] != iterations:
        history.append((iterations, best.objective))

    diagnostics = dict(best.diagnostics)
    diagnostics.pop("fallback", None)
    diagnostics.update(
        {
            "seed": seed,
            "iterations_requested": iterations,
            "feasible_moves": feasible_moves,
            "repair_feasibility_fallbacks": repair_feasibility_fallbacks,
            "rejected_infeasible_moves": rejected_infeasible_moves,
            "accepted_moves": accepted,
            "improving_moves": improving,
            "restarts": restarts,
            "destroy_weights": destroy_weights,
            "repair_weights": repair_weights,
            "convergence_history": history,
            "initialization": initialization,
            "evaluation_mode": evaluation_mode,
        }
    )
    return finalize_seeded_solution(
        "alns",
        best,
        initial_solution,
        started=started,
        native_source="alns_search",
        diagnostics=diagnostics,
        data=data,
        evaluation_mode=evaluation_mode,
    )
