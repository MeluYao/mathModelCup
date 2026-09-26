"""Experiments and paper-ready outputs for the urgent-priority Q2 model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from .problem_d_q2 import Q2Data, Q2Solution
from .problem_d_q2_geometry import ArcGeometry
from .problem_d_q2_grouped import solve_grouped_local_search
from .problem_d_q2_hybrid import solve_hybrid
from .problem_d_q2_milp import solve_integrated_milp
from .problem_d_q2_urgent import (
    URGENT_PRIORITY_MODE,
    all_box_lateness,
    is_urgent_box,
    required_deadline_s,
    urgent_lateness,
    urgent_objective_vector,
)
from .problem_d_q2_validation import ValidationReport, validate_q2_solution


METHOD_ORDER = ("integrated_milp", "hybrid")


def _write_csv(rows: list[dict[str, object]], path: Path) -> None:
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def _late_class_counts(data: Q2Data, solution: Q2Solution) -> tuple[int, int]:
    urgent = 0
    normal = 0
    for record in solution.deliveries:
        box = data.boxes[record.box_id]
        if record.delivery_time_s <= required_deadline_s(box) + 1e-7:
            continue
        if is_urgent_box(box):
            urgent += 1
        else:
            normal += 1
    return urgent, normal


def _comparison_row(
    method: str,
    data: Q2Data,
    solution: Q2Solution,
    report: ValidationReport,
) -> dict[str, object]:
    objective = urgent_objective_vector(data, solution)
    late_count, total_lateness_s = all_box_lateness(data, solution)
    urgent_late, normal_late = _late_class_counts(data, solution)
    return {
        "method": method,
        "validation": "PASS" if report.is_valid else "FAIL",
        "solver_status": solution.solver_status,
        "urgent_weighted_delivery": objective[0],
        "energy_kwh": objective[1],
        "trip_count": int(objective[2]),
        "makespan_s": objective[3],
        "runtime_s": solution.runtime_s,
        "late_box_count": late_count,
        "total_lateness_s": total_lateness_s,
        "urgent_late_count": urgent_late,
        "normal_late_count": normal_late,
        "minimum_return_soc_percent": min(
            (trip.plan.return_soc_percent for trip in solution.trips),
            default=100.0,
        ),
        "incumbent_source": solution.diagnostics.get("incumbent_source", ""),
        "seed_retained": bool(solution.diagnostics.get("seed_retained", False)),
    }


def _trip_rows(solution: Q2Solution) -> list[dict[str, object]]:
    return [
        {
            "trip_id": trip.trip_id,
            "aircraft_model": trip.plan.model_id,
            "aircraft_id": trip.aircraft_id,
            "battery_id": trip.battery_id,
            "start_time_s": trip.start_time_s,
            "return_time_s": trip.return_time_s,
            "battery_ready_time_s": trip.battery_ready_time_s,
            "duration_s": trip.plan.duration_s,
            "energy_kwh": trip.plan.energy_kwh,
            "return_soc_percent": trip.plan.return_soc_percent,
            "total_mass_kg": trip.plan.total_mass_kg,
            "total_volume_m3": trip.plan.total_volume_m3,
            "stop_sequence": ">".join(stop.service_id for stop in trip.plan.stops),
            "box_ids": "|".join(trip.plan.box_ids),
        }
        for trip in sorted(solution.trips, key=lambda item: (item.start_time_s, item.trip_id))
    ]


def _delivery_rows(data: Q2Data, solution: Q2Solution) -> list[dict[str, object]]:
    rows = []
    for record in sorted(solution.deliveries, key=lambda item: item.box_id):
        box = data.boxes[record.box_id]
        deadline = required_deadline_s(box)
        rows.append(
            {
                "box_id": record.box_id,
                "trip_id": record.trip_id,
                "service_id": record.service_id,
                "material_type": box.material_type,
                "is_urgent": is_urgent_box(box),
                "first_batch": box.first_batch,
                "priority_weight": box.priority_weight,
                "delivery_time_s": record.delivery_time_s,
                "required_deadline_s": deadline,
                "lateness_s": max(0.0, record.delivery_time_s - deadline),
                "on_time": record.delivery_time_s <= deadline + 1e-7,
            }
        )
    return rows


def _write_method_outputs(
    output_dir: Path,
    method: str,
    data: Q2Data,
    solution: Q2Solution,
    report: ValidationReport,
) -> None:
    method_dir = output_dir / method
    method_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(_trip_rows(solution), method_dir / "trips.csv")
    _write_csv(_delivery_rows(data, solution), method_dir / "deliveries.csv")
    _write_csv(
        [
            {
                "aircraft_id": trip.aircraft_id,
                "trip_id": trip.trip_id,
                "start_time_s": trip.start_time_s,
                "end_time_s": trip.return_time_s,
            }
            for trip in solution.trips
        ],
        method_dir / "aircraft_timeline.csv",
    )
    _write_csv(
        [
            {
                "battery_id": trip.battery_id,
                "trip_id": trip.trip_id,
                "start_time_s": trip.start_time_s,
                "flight_end_time_s": trip.return_time_s,
                "charge_end_time_s": trip.battery_ready_time_s,
            }
            for trip in solution.trips
        ],
        method_dir / "battery_timeline.csv",
    )
    validation_rows = (
        [{"status": "PASS", "code": "", "reference": "", "message": ""}]
        if report.is_valid
        else [
            {
                "status": "FAIL",
                "code": issue.code,
                "reference": issue.reference,
                "message": issue.message,
            }
            for issue in report.issues
        ]
    )
    _write_csv(validation_rows, method_dir / "validation.csv")
    summary = _comparison_row(method, data, solution, report)
    summary["objective_order"] = [
        "urgent_weighted_delivery",
        "energy_kwh",
        "trip_count",
        "makespan_s",
    ]
    summary["diagnostics"] = dict(solution.diagnostics)
    (method_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def write_urgent_outputs(
    output_dir: Path,
    data: Q2Data,
    arcs: Mapping[Tuple[str, str], ArcGeometry],
    solutions: Mapping[str, Q2Solution],
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for method in METHOD_ORDER:
        solution = solutions[method]
        report = validate_q2_solution(
            data,
            arcs,
            solution,
            evaluation_mode=URGENT_PRIORITY_MODE,
        )
        _write_method_outputs(output_dir, method, data, solution, report)
        rows.append(_comparison_row(method, data, solution, report))
    comparison = pd.DataFrame(rows)
    comparison.to_csv(
        output_dir / "method_comparison.csv", index=False, encoding="utf-8-sig"
    )
    return comparison


def run_urgent_methods(
    data: Q2Data,
    arcs: Mapping[Tuple[str, str], ArcGeometry],
    output_dir: Path,
    *,
    seed: int,
    iterations: int,
    time_limit_s: float,
) -> dict[str, Q2Solution]:
    common_seed = solve_grouped_local_search(
        data,
        arcs,
        seed=seed,
        iterations=iterations,
        evaluation_mode=URGENT_PRIORITY_MODE,
    )
    integrated = solve_integrated_milp(
        data,
        arcs,
        time_limit_s=time_limit_s,
        initial_solution=common_seed,
        evaluation_mode=URGENT_PRIORITY_MODE,
    )
    hybrid = solve_hybrid(
        data,
        arcs,
        seed=seed,
        iterations=iterations,
        rounds=1,
        time_limit_s=time_limit_s,
        initial_solution=common_seed,
        evaluation_mode=URGENT_PRIORITY_MODE,
    )
    solutions = {"integrated_milp": integrated, "hybrid": hybrid}
    write_urgent_outputs(output_dir, data, arcs, solutions)
    return solutions


def run_hybrid_repetitions(
    data: Q2Data,
    arcs: Mapping[Tuple[str, str], ArcGeometry],
    initial_solution: Q2Solution,
    *,
    seeds: Sequence[int],
    iterations: int,
    time_limit_s: float,
    output_dir: Path | None = None,
    solutions_out: dict[int, Q2Solution] | None = None,
) -> pd.DataFrame:
    rows = []
    convergence_rows = []
    for seed in seeds:
        solution = solve_hybrid(
            data,
            arcs,
            seed=seed,
            iterations=iterations,
            rounds=1,
            time_limit_s=time_limit_s,
            initial_solution=initial_solution,
            evaluation_mode=URGENT_PRIORITY_MODE,
        )
        report = validate_q2_solution(
            data, arcs, solution, evaluation_mode=URGENT_PRIORITY_MODE
        )
        if solutions_out is not None:
            solutions_out[seed] = solution
        row = _comparison_row("hybrid", data, solution, report)
        row["seed"] = seed
        rows.append(row)
        for iteration, objective in solution.diagnostics.get("convergence_history", []):
            convergence_rows.append(
                {
                    "seed": seed,
                    "iteration": iteration,
                    "urgent_weighted_delivery": objective[0],
                    "energy_kwh": objective[1],
                    "trip_count": objective[2],
                    "makespan_s": objective[3],
                }
            )
    frame = pd.DataFrame(rows)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        frame.to_csv(
            output_dir / "hybrid_repetitions.csv",
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame(convergence_rows).to_csv(
            output_dir / "hybrid_convergence.csv",
            index=False,
            encoding="utf-8-sig",
        )
    return frame


def create_urgent_figures(
    output_dir: Path,
    data: Q2Data,
    solutions: Mapping[str, Q2Solution],
    repetitions: pd.DataFrame,
) -> list[Path]:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    comparison = pd.read_csv(output_dir / "method_comparison.csv")
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    columns = (
        ("urgent_weighted_delivery", "Urgent timing"),
        ("energy_kwh", "Energy (kWh)"),
        ("trip_count", "Trips"),
        ("makespan_s", "Makespan (s)"),
    )
    for axis, (column, title) in zip(axes.flat, columns):
        axis.bar(comparison["method"], comparison[column], color=("#285F85", "#D9822B"))
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    path = figures_dir / "objective_comparison.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths.append(path)

    best = min(solutions.values(), key=lambda item: item.objective)
    aircraft_ids = sorted({trip.aircraft_id for trip in best.trips})
    y_index = {resource: index for index, resource in enumerate(aircraft_ids)}
    fig, axis = plt.subplots(figsize=(11, max(4, 0.55 * len(aircraft_ids))))
    for trip in best.trips:
        axis.barh(
            y_index[trip.aircraft_id],
            trip.return_time_s - trip.start_time_s,
            left=trip.start_time_s,
            height=0.6,
            color="#3274A1",
        )
    axis.set_yticks(range(len(aircraft_ids)), aircraft_ids)
    axis.set_xlabel("Time (s)")
    axis.set_title(f"Aircraft schedule: {best.method}")
    axis.grid(axis="x", alpha=0.25)
    path = figures_dir / "aircraft_gantt.png"
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths.append(path)

    centers = [node for node in data.nodes.values() if node.is_center]
    center = centers[0]
    fig, axis = plt.subplots(figsize=(8, 7))
    for trip in best.trips:
        sequence = [center.node_id] + [stop.service_id for stop in trip.plan.stops] + [center.node_id]
        axis.plot(
            [data.nodes[node_id].longitude for node_id in sequence],
            [data.nodes[node_id].latitude for node_id in sequence],
            alpha=0.35,
            linewidth=1.0,
        )
    axis.scatter(
        [node.longitude for node in data.nodes.values()],
        [node.latitude for node in data.nodes.values()],
        s=24,
        color="#1F1F1F",
        zorder=3,
    )
    axis.set_xlabel("Longitude")
    axis.set_ylabel("Latitude")
    axis.set_title(f"Routes: {best.method}")
    axis.grid(alpha=0.2)
    path = figures_dir / "route_map.png"
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths.append(path)

    if not repetitions.empty:
        fig, axis = plt.subplots(figsize=(9, 4.5))
        axis.plot(
            repetitions["seed"].astype(str),
            repetitions["urgent_weighted_delivery"],
            marker="o",
            color="#D9822B",
        )
        axis.set_xlabel("Hybrid seed")
        axis.set_ylabel("Urgent weighted delivery")
        axis.set_title("Hybrid repetition stability")
        axis.grid(alpha=0.25)
        path = figures_dir / "hybrid_repetition_stability.png"
        fig.tight_layout()
        fig.savefig(path, dpi=220)
        plt.close(fig)
        paths.append(path)
    return paths
