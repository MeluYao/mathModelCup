"""Validated adapters from persisted question-2 results to question 3."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from .problem_d_q2 import (
    Q2Data,
    Q2Solution,
    TripDraft,
    TripExecution,
    TripStop,
)
from .problem_d_q2_geometry import ArcGeometry
from .problem_d_q2_physics import charge_time_s, evaluate_trip
from .problem_d_q2_schedule import _make_solution
from .problem_d_q2_validation import validate_q2_solution


@dataclass(frozen=True)
class TransportScenario:
    scenario_id: str
    q2_method: str
    source_dir: Path
    source_objective: tuple[float, float, float, int]
    solution: Q2Solution
    provenance: Mapping[str, object]


class Q2ResultIntegrityError(ValueError):
    """Raised when persisted Q2 files disagree with recomputed physics."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_objective(summary: Mapping[str, object]) -> tuple[float, float, float, int]:
    raw = summary.get("objective")
    if not isinstance(raw, dict):
        raise Q2ResultIntegrityError("summary objective must be an object")
    try:
        return (
            float(raw["normalized_weighted_delivery_time"]),
            float(raw["makespan_s"]),
            float(raw["energy_kwh"]),
            int(raw["trip_count"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise Q2ResultIntegrityError("summary objective is incomplete") from error


def load_transport_scenario(
    data: Q2Data,
    arcs: Mapping[tuple[str, str], ArcGeometry],
    result_dir: Path,
    scenario_id: str,
) -> TransportScenario:
    """Rebuild and independently validate one persisted Q2 schedule."""
    result_dir = Path(result_dir).resolve()
    paths = {
        name: result_dir / name
        for name in ("summary.json", "trips.csv", "deliveries.csv")
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise Q2ResultIntegrityError(f"missing Q2 result files: {missing}")

    summary = json.loads(paths["summary.json"].read_text(encoding="utf-8"))
    trips = pd.read_csv(paths["trips.csv"])
    deliveries = pd.read_csv(paths["deliveries.csv"])
    source_objective = _source_objective(summary)
    battery_by_id = {battery.battery_id: battery for battery in data.batteries}
    executions = []

    for record in trips.to_dict("records"):
        trip_id = str(record["trip_id"])
        listed_box_ids = tuple(str(record["box_ids"]).split("|"))
        delivery_rows = deliveries[deliveries["trip_id"].astype(str) == trip_id]
        delivered_box_ids = set(delivery_rows["box_id"].astype(str))
        if delivered_box_ids != set(listed_box_ids):
            raise Q2ResultIntegrityError(
                f"trip {trip_id} box IDs disagree between trips and deliveries"
            )

        service_ids = tuple(str(record["stop_sequence"]).split(">"))
        stops = tuple(
            TripStop(
                service_id,
                tuple(
                    box_id
                    for box_id in listed_box_ids
                    if data.boxes[box_id].service_id == service_id
                ),
            )
            for service_id in service_ids
        )
        plan = evaluate_trip(
            data,
            arcs,
            TripDraft(str(record["aircraft_model"]), stops),
        )
        start_time = float(record["start_time_s"])
        return_time = start_time + plan.duration_s
        battery_id = str(record["battery_id"])
        if battery_id not in battery_by_id:
            raise Q2ResultIntegrityError(
                f"trip {trip_id} references unknown battery {battery_id}"
            )
        battery = battery_by_id[battery_id]
        battery_ready_time = return_time + charge_time_s(
            plan.return_soc_percent / 100.0,
            battery.full_charge_time_s,
        )
        persisted_times = (
            float(record["return_time_s"]),
            float(record["battery_ready_time_s"]),
        )
        if not np.allclose(
            persisted_times,
            (return_time, battery_ready_time),
            rtol=0.0,
            atol=1e-7,
        ):
            raise Q2ResultIntegrityError(
                f"trip {trip_id} persisted times disagree with recomputed physics"
            )
        executions.append(
            TripExecution(
                trip_id=trip_id,
                plan=plan,
                aircraft_id=str(record["aircraft_id"]),
                battery_id=battery_id,
                start_time_s=start_time,
                return_time_s=return_time,
                battery_ready_time_s=battery_ready_time,
            )
        )

    diagnostics = dict(summary.get("diagnostics") or {})
    reconstructed = _make_solution(
        data,
        method=str(summary.get("method", scenario_id)),
        executions=executions,
        runtime_s=float(summary.get("runtime_s", 0.0)),
        solver_status=str(summary.get("solver_status", "FEASIBLE")),
        diagnostics=diagnostics,
    )
    report = validate_q2_solution(data, arcs, reconstructed)
    if not report.is_valid:
        details = "; ".join(
            f"{issue.code}: {issue.message}" for issue in report.issues
        )
        raise Q2ResultIntegrityError(details)
    if not np.allclose(
        reconstructed.objective[:3],
        source_objective[:3],
        rtol=0.0,
        atol=1e-7,
    ):
        raise Q2ResultIntegrityError(
            "reconstructed objective does not match summary objective"
        )
    if reconstructed.objective[3] != source_objective[3]:
        raise Q2ResultIntegrityError(
            "reconstructed trip count does not match summary objective"
        )

    provenance = {
        "incumbent_source": diagnostics.get("incumbent_source", ""),
        "native_improved_seed": bool(
            diagnostics.get("native_improved_seed", False)
        ),
        "seed_retained": bool(diagnostics.get("seed_retained", False)),
        "summary_sha256": _sha256(paths["summary.json"]),
        "trips_sha256": _sha256(paths["trips.csv"]),
        "deliveries_sha256": _sha256(paths["deliveries.csv"]),
    }
    return TransportScenario(
        scenario_id=str(scenario_id),
        q2_method=str(summary.get("method", scenario_id)),
        source_dir=result_dir,
        source_objective=source_objective,
        solution=reconstructed,
        provenance=provenance,
    )
