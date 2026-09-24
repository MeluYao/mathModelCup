from __future__ import annotations

import math

from math_model_cup.problem_d_q1 import trip_energy
from math_model_cup.problem_d_q1_analysis import (
    candidate_critical_margins,
    reserve_stability_intervals,
)

from test_problem_d_q1 import make_problem


def test_candidate_critical_margin_matches_energy_definition() -> None:
    data = make_problem()

    records = candidate_critical_margins(data)

    assert records
    assert all(record.service_id == "S001" for record in records)
    assert all(record.model_id == "T" for record in records)
    assert [record.rho_crit for record in records] == sorted(
        (record.rho_crit for record in records)
    )
    for record in records:
        expected_energy = trip_energy(
            data.aircraft[record.model_id],
            data.geometries[record.service_id],
            record.mass_kg,
        )
        assert math.isclose(record.energy_kwh, expected_energy, rel_tol=1e-12)
        assert math.isclose(
            record.rho_crit,
            1.0 - expected_energy / data.aircraft[record.model_id].usable_energy_kwh,
            rel_tol=1e-12,
        )


def test_reserve_intervals_cover_domain_and_trip_count_is_monotone() -> None:
    data = make_problem()
    records = candidate_critical_margins(data)

    intervals = reserve_stability_intervals(data, records=records)

    assert intervals[0].lower_ratio == 0.0
    assert intervals[0].lower_inclusive
    assert intervals[-1].upper_ratio == 1.0
    assert not intervals[-1].upper_inclusive
    for left, right in zip(intervals, intervals[1:]):
        assert math.isclose(left.upper_ratio, right.lower_ratio, abs_tol=1e-15)
        assert left.upper_inclusive
        assert not right.lower_inclusive
    feasible_counts = [
        interval.trip_count for interval in intervals if interval.feasible
    ]
    assert feasible_counts == sorted(feasible_counts)


def test_compressed_intervals_change_only_when_solution_changes() -> None:
    data = make_problem()

    intervals = reserve_stability_intervals(data)

    signatures = [interval.solution_signature for interval in intervals]
    assert all(left != right for left, right in zip(signatures, signatures[1:]))
