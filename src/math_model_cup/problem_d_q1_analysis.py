"""Extended multi-objective and sensitivity analysis for Problem D, question 1."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from math import inf
from time import perf_counter
from typing import Iterable, Literal, Sequence, Tuple

from .problem_d_q1 import (
    NUMERIC_TOLERANCE,
    CandidateTrip,
    InfeasibleProblemError,
    MethodSolution,
    ProblemData,
    TripResult,
    _solution_from_candidates,
    _target_counts,
    _type_properties,
    generate_candidates,
    solve_dynamic_programming,
    trip_energy,
    trip_time,
    validate_solution,
)


@dataclass(frozen=True)
class ParetoPoint:
    trip_count: int
    total_time_s: float
    total_energy_kwh: float
    candidates: Tuple[CandidateTrip, ...]
    reserve_ratio: float = 0.20
    range_energy_fraction: float = 1.0

    @property
    def objective(self) -> Tuple[int, float, float]:
        return self.trip_count, self.total_time_s, self.total_energy_kwh


@dataclass(frozen=True)
class CriticalMarginRecord:
    service_id: str
    model_id: str
    counts: Tuple[int, ...]
    mass_kg: float
    volume_m3: float
    energy_kwh: float
    rho_crit: float


@dataclass(frozen=True)
class ReserveInterval:
    lower_ratio: float
    upper_ratio: float
    lower_inclusive: bool
    upper_inclusive: bool
    sample_ratio: float
    feasible: bool
    trip_count: int | None
    total_time_s: float | None
    total_energy_kwh: float | None
    trips: Tuple[TripResult, ...]

    @property
    def solution_signature(self) -> Tuple[object, ...]:
        if not self.feasible:
            return ("INFEASIBLE",)
        return tuple(
            sorted(
                (
                    trip.service_id,
                    trip.model_id,
                    trip.counts,
                )
                for trip in self.trips
            )
        )


def _path_key(point: ParetoPoint) -> Tuple[Tuple[object, ...], ...]:
    return tuple(
        (
            candidate.service_id,
            candidate.model_id,
            candidate.counts,
            round(candidate.time_s, 9),
            round(candidate.energy_kwh, 12),
        )
        for candidate in point.candidates
    )


def _dominates(left: ParetoPoint, right: ParetoPoint) -> bool:
    no_worse = (
        left.trip_count <= right.trip_count
        and left.total_time_s <= right.total_time_s + 1e-8
        and left.total_energy_kwh <= right.total_energy_kwh + 1e-10
    )
    strictly_better = (
        left.trip_count < right.trip_count
        or left.total_time_s < right.total_time_s - 1e-8
        or left.total_energy_kwh < right.total_energy_kwh - 1e-10
    )
    return no_worse and strictly_better


def _prune_pareto(points: Iterable[ParetoPoint]) -> Tuple[ParetoPoint, ...]:
    ordered = sorted(
        points,
        key=lambda point: (
            point.trip_count,
            point.total_time_s,
            point.total_energy_kwh,
            _path_key(point),
        ),
    )
    unique: list[ParetoPoint] = []
    objective_keys: set[Tuple[int, int, int]] = set()
    for point in ordered:
        key = (
            point.trip_count,
            round(point.total_time_s * 1e8),
            round(point.total_energy_kwh * 1e10),
        )
        if key not in objective_keys:
            unique.append(point)
            objective_keys.add(key)
    return tuple(
        point
        for point in unique
        if not any(
            other is not point and _dominates(other, point) for other in unique
        )
    )


def _service_pareto(
    data: ProblemData,
    service_id: str,
    reserve_ratio: float,
    range_energy_fraction: float,
) -> Tuple[ParetoPoint, ...]:
    target = _target_counts(data, service_id)
    candidates = generate_candidates(
        data,
        service_id,
        reserve_ratio,
        range_energy_fraction=range_energy_fraction,
    )
    if not candidates:
        raise InfeasibleProblemError(f"no feasible trip for {service_id}")

    @lru_cache(maxsize=None)
    def frontier(state: Tuple[int, ...]) -> Tuple[ParetoPoint, ...]:
        if state == target:
            return (
                ParetoPoint(
                    trip_count=0,
                    total_time_s=0.0,
                    total_energy_kwh=0.0,
                    candidates=tuple(),
                    reserve_ratio=reserve_ratio,
                    range_energy_fraction=range_energy_fraction,
                ),
            )
        options: list[ParetoPoint] = []
        for candidate in candidates:
            next_state = tuple(
                state[index] + candidate.counts[index]
                for index in range(len(target))
            )
            if next_state == state or any(
                next_state[index] > target[index] for index in range(len(target))
            ):
                continue
            for sub_point in frontier(next_state):
                options.append(
                    ParetoPoint(
                        trip_count=1 + sub_point.trip_count,
                        total_time_s=candidate.time_s + sub_point.total_time_s,
                        total_energy_kwh=(
                            candidate.energy_kwh + sub_point.total_energy_kwh
                        ),
                        candidates=(candidate,) + sub_point.candidates,
                        reserve_ratio=reserve_ratio,
                        range_energy_fraction=range_energy_fraction,
                    )
                )
        return _prune_pareto(options)

    result = frontier(tuple(0 for _ in target))
    if not result:
        raise InfeasibleProblemError(f"cannot cover all boxes for {service_id}")
    return result


def pareto_dynamic_programming(
    data: ProblemData,
    reserve_ratio: float = 0.20,
    *,
    range_energy_fraction: float = 1.0,
) -> Tuple[ParetoPoint, ...]:
    """Return the global non-dominated trip/time/energy frontier."""
    global_frontier: Tuple[ParetoPoint, ...] = (
        ParetoPoint(
            trip_count=0,
            total_time_s=0.0,
            total_energy_kwh=0.0,
            candidates=tuple(),
            reserve_ratio=reserve_ratio,
            range_energy_fraction=range_energy_fraction,
        ),
    )
    for service_id in sorted(data.geometries):
        local_frontier = _service_pareto(
            data, service_id, reserve_ratio, range_energy_fraction
        )
        combined = (
            ParetoPoint(
                trip_count=global_point.trip_count + local_point.trip_count,
                total_time_s=(
                    global_point.total_time_s + local_point.total_time_s
                ),
                total_energy_kwh=(
                    global_point.total_energy_kwh + local_point.total_energy_kwh
                ),
                candidates=global_point.candidates + local_point.candidates,
                reserve_ratio=reserve_ratio,
                range_energy_fraction=range_energy_fraction,
            )
            for global_point in global_frontier
            for local_point in local_frontier
        )
        global_frontier = _prune_pareto(combined)
    return global_frontier


def select_anchor(
    points: Sequence[ParetoPoint], objective: Literal["trips", "time", "energy"]
) -> ParetoPoint:
    if not points:
        raise InfeasibleProblemError("Pareto frontier is empty")
    keys = {
        "trips": lambda point: (
            point.trip_count,
            point.total_time_s,
            point.total_energy_kwh,
        ),
        "time": lambda point: (
            point.total_time_s,
            point.trip_count,
            point.total_energy_kwh,
        ),
        "energy": lambda point: (
            point.total_energy_kwh,
            point.trip_count,
            point.total_time_s,
        ),
    }
    if objective not in keys:
        raise ValueError(f"unknown objective: {objective}")
    return min(points, key=keys[objective])


def pareto_for_trip_count(
    points: Sequence[ParetoPoint], trip_count: int
) -> Tuple[ParetoPoint, ...]:
    return tuple(
        sorted(
            (point for point in points if point.trip_count == trip_count),
            key=lambda point: (point.total_time_s, point.total_energy_kwh),
        )
    )


def best_under_trip_budget(
    points: Sequence[ParetoPoint],
    maximum_trips: int,
    objective: Literal["time", "energy"],
) -> ParetoPoint:
    feasible = [point for point in points if point.trip_count <= maximum_trips]
    if not feasible:
        raise InfeasibleProblemError(
            f"no Pareto solution uses at most {maximum_trips} trips"
        )
    return select_anchor(feasible, objective)


def pareto_solution(
    data: ProblemData,
    point: ParetoPoint,
    *,
    method: str,
    runtime_s: float = 0.0,
) -> MethodSolution:
    solution = _solution_from_candidates(data, method, point.candidates, runtime_s)
    validate_solution(
        data,
        solution.trips,
        point.reserve_ratio,
        range_energy_fraction=point.range_energy_fraction,
    )
    return solution


def candidate_critical_margins(
    data: ProblemData,
    *,
    range_energy_fraction: float = 1.0,
) -> Tuple[CriticalMarginRecord, ...]:
    """Enumerate every mass/volume-feasible service-model-pattern threshold."""
    properties = _type_properties(data)
    records: list[CriticalMarginRecord] = []
    for service_id in sorted(data.geometries):
        target = _target_counts(data, service_id)
        geometry = data.geometries[service_id]
        for counts in product(*(range(count + 1) for count in target)):
            if not any(counts):
                continue
            mass_kg = sum(
                counts[index] * properties[material_type][0]
                for index, material_type in enumerate(data.material_types)
            )
            volume_m3 = sum(
                counts[index] * properties[material_type][1]
                for index, material_type in enumerate(data.material_types)
            )
            for model_id in sorted(data.aircraft):
                aircraft = data.aircraft[model_id]
                if mass_kg > aircraft.max_payload_kg + NUMERIC_TOLERANCE:
                    continue
                if volume_m3 > aircraft.volume_capacity_m3 + NUMERIC_TOLERANCE:
                    continue
                energy_kwh = trip_energy(
                    aircraft,
                    geometry,
                    mass_kg,
                    range_energy_fraction=range_energy_fraction,
                )
                records.append(
                    CriticalMarginRecord(
                        service_id=service_id,
                        model_id=model_id,
                        counts=tuple(int(value) for value in counts),
                        mass_kg=mass_kg,
                        volume_m3=volume_m3,
                        energy_kwh=energy_kwh,
                        rho_crit=1.0 - energy_kwh / aircraft.usable_energy_kwh,
                    )
                )
    return tuple(
        sorted(
            records,
            key=lambda record: (
                record.rho_crit,
                record.service_id,
                record.model_id,
                record.counts,
            ),
        )
    )


def _critical_boundaries(records: Sequence[CriticalMarginRecord]) -> list[float]:
    values = sorted(
        record.rho_crit for record in records if 0.0 < record.rho_crit < 1.0
    )
    boundaries = [0.0]
    for value in values:
        if value - boundaries[-1] > 1e-12:
            boundaries.append(value)
        else:
            boundaries[-1] = max(boundaries[-1], value)
    if 1.0 - boundaries[-1] > 1e-12:
        boundaries.append(1.0)
    elif boundaries[-1] != 1.0:
        boundaries[-1] = 1.0
    return boundaries


def reserve_stability_intervals(
    data: ProblemData,
    *,
    records: Sequence[CriticalMarginRecord] | None = None,
    range_energy_fraction: float = 1.0,
) -> Tuple[ReserveInterval, ...]:
    """Solve between every critical margin and merge unchanged optima."""
    if records is None:
        records = candidate_critical_margins(
            data, range_energy_fraction=range_energy_fraction
        )
    boundaries = _critical_boundaries(records)
    raw: list[ReserveInterval] = []
    for index, (lower, upper) in enumerate(zip(boundaries, boundaries[1:])):
        sample = (lower + upper) / 2.0
        try:
            solution = solve_dynamic_programming(
                data,
                sample,
                range_energy_fraction=range_energy_fraction,
            )
        except InfeasibleProblemError:
            raw.append(
                ReserveInterval(
                    lower_ratio=lower,
                    upper_ratio=1.0,
                    lower_inclusive=index == 0,
                    upper_inclusive=False,
                    sample_ratio=sample,
                    feasible=False,
                    trip_count=None,
                    total_time_s=None,
                    total_energy_kwh=None,
                    trips=tuple(),
                )
            )
            break
        raw.append(
            ReserveInterval(
                lower_ratio=lower,
                upper_ratio=upper,
                lower_inclusive=index == 0,
                upper_inclusive=upper < 1.0,
                sample_ratio=sample,
                feasible=True,
                trip_count=solution.trip_count,
                total_time_s=solution.total_time_s,
                total_energy_kwh=solution.total_energy_kwh,
                trips=solution.trips,
            )
        )

    compressed: list[ReserveInterval] = []
    for interval in raw:
        if (
            compressed
            and compressed[-1].solution_signature == interval.solution_signature
        ):
            previous = compressed[-1]
            compressed[-1] = ReserveInterval(
                lower_ratio=previous.lower_ratio,
                upper_ratio=interval.upper_ratio,
                lower_inclusive=previous.lower_inclusive,
                upper_inclusive=interval.upper_inclusive,
                sample_ratio=previous.sample_ratio,
                feasible=previous.feasible,
                trip_count=previous.trip_count,
                total_time_s=previous.total_time_s,
                total_energy_kwh=previous.total_energy_kwh,
                trips=previous.trips,
            )
        else:
            compressed.append(interval)
    return tuple(compressed)
