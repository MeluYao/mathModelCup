"""Persistence and comparison reports for aligned question-3 scenarios."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import pickle
from typing import Sequence

import pandas as pd

from .problem_d_q3_aligned import AlignedScenarioResult


def _maximum_concurrency(intervals) -> int:
    events = []
    for start, end in intervals:
        events.append((float(start), 1))
        events.append((float(end), -1))
    active = 0
    maximum = 0
    for _, delta in sorted(events, key=lambda item: (item[0], item[1])):
        active += delta
        maximum = max(maximum, active)
    return maximum


def _summary(result: AlignedScenarioResult) -> dict[str, object]:
    if result.solution is None:
        return {
            "scenario_id": result.scenario_id,
            "q2_method": result.q2_method,
            "q2_objective": result.q2_objective,
            "q2_provenance": dict(result.provenance),
            "solver_status": result.status,
            "is_valid": False,
            "reason": result.reason,
            "runtime_s": result.runtime_s,
        }
    solution = result.solution
    prepared = result.prepared
    validation_issues = (
        []
        if result.validation is None
        else [asdict(issue) for issue in result.validation.issues]
    )
    relay_peak = _maximum_concurrency(
        (
            sortie.preparation_start_time_s,
            sortie.return_time_s,
        )
        for sortie in solution.relay_sorties
    )
    energy_peak = _maximum_concurrency(
        (
            sortie.preparation_start_time_s,
            sortie.energy_ready_time_s,
        )
        for sortie in solution.relay_sorties
    )
    return {
        "scenario_id": result.scenario_id,
        "q2_method": result.q2_method,
        "q2_objective": result.q2_objective,
        "q2_provenance": dict(result.provenance),
        "method": solution.method,
        "solver_status": solution.solver_status,
        "is_valid": result.is_valid,
        "validation_issues": validation_issues,
        "reason": result.reason,
        "runtime_s": result.runtime_s,
        "objective": solution.objective.as_tuple(),
        "transport_objective": solution.transport.objective,
        "transport_trip_count": len(solution.transport.trips),
        "relay_trip_count": len(solution.relay_sorties),
        "communication_segment_count": len(solution.communication_assignments),
        "direct_segment_count": sum(
            assignment.mode == "direct"
            for assignment in solution.communication_assignments
        ),
        "relay_segment_count": sum(
            assignment.mode == "relay"
            for assignment in solution.communication_assignments
        ),
        "dark_segment_count": 0 if prepared is None else len(prepared.dark_segments),
        "relay_state_count": 0 if prepared is None else len(prepared.relay_states),
        "maximum_relay_concurrency": relay_peak,
        "maximum_energy_unit_concurrency": energy_peak,
    }


def write_scenario_result(
    result: AlignedScenarioResult,
    output_dir: Path,
) -> Path:
    """Write one scenario's complete reusable result bundle."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = _summary(result)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if result.solution is None:
        return output_dir

    solution = result.solution
    with (output_dir / "q3_solution.pkl").open("wb") as stream:
        pickle.dump(solution, stream)
    pd.DataFrame(
        [
            {
                "trip_id": trip.trip_id,
                "model_id": trip.plan.model_id,
                "aircraft_id": trip.aircraft_id,
                "battery_id": trip.battery_id,
                "stop_sequence": ">".join(
                    stop.service_id for stop in trip.plan.stops
                ),
                "box_ids": "|".join(trip.plan.box_ids),
                "start_time_s": trip.start_time_s,
                "return_time_s": trip.return_time_s,
                "battery_ready_time_s": trip.battery_ready_time_s,
                "energy_kwh": trip.plan.energy_kwh,
                "return_soc_percent": trip.plan.return_soc_percent,
            }
            for trip in solution.transport.trips
        ]
    ).to_csv(output_dir / "transport_schedule.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [asdict(delivery) for delivery in solution.transport.deliveries]
    ).to_csv(output_dir / "deliveries.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [asdict(sortie) for sortie in solution.relay_sorties]
    ).to_csv(output_dir / "relay_schedule.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [asdict(assignment) for assignment in solution.communication_assignments]
    ).to_csv(
        output_dir / "communication_assignments.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return output_dir


def _comparison_row(result: AlignedScenarioResult) -> dict[str, object]:
    summary = _summary(result)
    objective = summary.get("objective", (None,) * 5)
    return {
        "scenario_id": result.scenario_id,
        "q2_method": result.q2_method,
        "q2_weighted_delivery_time": result.q2_objective[0],
        "q2_makespan_s": result.q2_objective[1],
        "q2_energy_kwh": result.q2_objective[2],
        "q2_trip_count": result.q2_objective[3],
        "status": result.status,
        "is_valid": result.is_valid,
        "q3_weighted_delivery_time": objective[0],
        "q3_joint_makespan_s": objective[1],
        "q3_total_energy_kwh": objective[2],
        "q3_transport_trip_count": objective[3],
        "q3_relay_trip_count": objective[4],
        "communication_segment_count": summary.get("communication_segment_count", 0),
        "direct_segment_count": summary.get("direct_segment_count", 0),
        "relay_segment_count": summary.get("relay_segment_count", 0),
        "maximum_relay_concurrency": summary.get("maximum_relay_concurrency", 0),
        "maximum_energy_unit_concurrency": summary.get(
            "maximum_energy_unit_concurrency", 0
        ),
        "runtime_s": result.runtime_s,
        "reason": result.reason,
    }


def write_aligned_comparison(
    results: Sequence[AlignedScenarioResult],
    selected: AlignedScenarioResult | None,
    output_dir: Path,
) -> Path:
    """Write the cross-scenario comparison and stable selected manifest."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([_comparison_row(result) for result in results]).to_csv(
        output_dir / "comparison.csv", index=False, encoding="utf-8-sig"
    )
    if selected is not None:
        selected_dir = output_dir / "selected"
        selected_dir.mkdir(exist_ok=True)
        (selected_dir / "selection.json").write_text(
            json.dumps(
                {
                    "selected_scenario_id": selected.scenario_id,
                    "selected_result_directory": f"../{selected.scenario_id}",
                    "selection_rule": (
                        "lexicographic_q3_objective_among_"
                        "independently_validated_results"
                    ),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    return output_dir
