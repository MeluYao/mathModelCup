"""Repeated experiments, sensitivity analysis, and publication figures for D2."""

from __future__ import annotations

from dataclasses import replace
from math import ceil
from pathlib import Path
from typing import Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .problem_d_q2 import Q2Data, Q2Solution, TripPlan
from .problem_d_q2_alns import solve_alns
from .problem_d_q2_geometry import ArcGeometry
from .problem_d_q2_schedule import construct_resource_aware_boxwise_solution
from .problem_d_q2_validation import validate_q2_solution


METHOD_LABELS = {
    "integrated_milp": "Integrated",
    "candidate": "Candidate",
    "alns": "ALNS",
    "hybrid": "Hybrid",
}
MODEL_COLORS = {"A": "#4477AA", "B": "#EE6677", "C": "#228833"}


def _configure_paper_style() -> None:
    available = {font.name for font in matplotlib.font_manager.fontManager.ttflist}
    preferred = [
        "Microsoft YaHei",
        "Noto Sans CJK SC",
        "SimHei",
        "Arial",
        "DejaVu Sans",
    ]
    matplotlib.rcParams.update(
        {
            "font.family": next((font for font in preferred if font in available), "DejaVu Sans"),
            "axes.unicode_minus": False,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.6,
        }
    )


def _solution_row(seed: int, solution: Q2Solution, valid: bool) -> dict[str, object]:
    return {
        "seed": seed,
        "validation": "PASS" if valid else "FAIL",
        "objective_primary": solution.objective[0],
        "makespan_s": solution.objective[1],
        "energy_kwh": solution.objective[2],
        "trip_count": solution.objective[3],
        "runtime_s": solution.runtime_s,
        "feasible_moves": solution.diagnostics.get("feasible_moves", 0),
        "accepted_moves": solution.diagnostics.get("accepted_moves", 0),
        "improving_moves": solution.diagnostics.get("improving_moves", 0),
    }


def run_alns_repetitions(
    data: Q2Data,
    arcs: Mapping[Tuple[str, str], ArcGeometry],
    seeds: Sequence[int],
    iterations: int,
    candidate_pool: Sequence[TripPlan] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run ALNS for fixed seeds and return summary and convergence observations."""
    rows = []
    convergence_rows = []
    for seed in seeds:
        solution = solve_alns(
            data,
            arcs,
            seed=int(seed),
            iterations=iterations,
            candidate_pool=candidate_pool,
        )
        report = validate_q2_solution(data, arcs, solution)
        rows.append(_solution_row(int(seed), solution, report.is_valid))
        for iteration, objective in solution.diagnostics["convergence_history"]:
            convergence_rows.append(
                {
                    "seed": int(seed),
                    "iteration": int(iteration),
                    "objective_primary": float(objective[0]),
                    "makespan_s": float(objective[1]),
                    "energy_kwh": float(objective[2]),
                    "trip_count": int(objective[3]),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(convergence_rows)


def _resource_subset(data: Q2Data, fraction: float) -> Q2Data:
    aircraft = []
    batteries = []
    for model_id in sorted(data.aircraft_models):
        model_aircraft = sorted(
            (unit for unit in data.aircraft_units if unit.model_id == model_id),
            key=lambda unit: unit.aircraft_id,
        )
        model_batteries = sorted(
            (battery for battery in data.batteries if battery.model_id == model_id),
            key=lambda battery: battery.battery_id,
        )
        aircraft.extend(model_aircraft[: max(1, ceil(len(model_aircraft) * fraction))])
        batteries.extend(model_batteries[: max(1, ceil(len(model_batteries) * fraction))])
    return replace(data, aircraft_units=tuple(aircraft), batteries=tuple(batteries))


def _sensitivity_result(
    factor: str,
    level: float,
    data: Q2Data,
    arcs: Mapping[Tuple[str, str], ArcGeometry],
    pool: Sequence[TripPlan],
) -> dict[str, object]:
    try:
        solution = construct_resource_aware_boxwise_solution(data, pool, "sensitivity")
        report = validate_q2_solution(data, arcs, solution)
        return {
            "factor": factor,
            "level": level,
            "feasible": report.is_valid,
            "objective_primary": solution.objective[0],
            "makespan_s": solution.objective[1],
            "energy_kwh": solution.objective[2],
            "trip_count": solution.objective[3],
            "minimum_return_soc_percent": min(
                trip.plan.return_soc_percent for trip in solution.trips
            ),
            "candidate_count": len(pool),
            "issue_count": len(report.issues),
            "error": "",
        }
    except Exception as error:
        return {
            "factor": factor,
            "level": level,
            "feasible": False,
            "objective_primary": np.nan,
            "makespan_s": np.nan,
            "energy_kwh": np.nan,
            "trip_count": np.nan,
            "minimum_return_soc_percent": np.nan,
            "candidate_count": len(pool),
            "issue_count": 1,
            "error": str(error),
        }


def run_sensitivity_analysis(
    data: Q2Data,
    arcs: Mapping[Tuple[str, str], ArcGeometry],
    candidate_pool: Sequence[TripPlan],
) -> pd.DataFrame:
    """Evaluate four operational sensitivity families with a common constructor."""
    rows = []
    for reserve_ratio in (0.15, 0.20, 0.25, 0.30, 0.35):
        pool = tuple(
            plan
            for plan in candidate_pool
            if plan.return_soc_percent + 1e-9 >= 100.0 * reserve_ratio
        )
        rows.append(
            _sensitivity_result("reserve_ratio", reserve_ratio, data, arcs, pool)
        )

    for multiplier in (0.75, 1.00, 1.25):
        modified = replace(
            data,
            batteries=tuple(
                replace(
                    battery,
                    full_charge_time_s=battery.full_charge_time_s * multiplier,
                )
                for battery in data.batteries
            ),
        )
        rows.append(
            _sensitivity_result(
                "charge_time_multiplier", multiplier, modified, arcs, candidate_pool
            )
        )

    for fraction in (0.50, 0.75, 1.00):
        modified = _resource_subset(data, fraction)
        rows.append(
            _sensitivity_result(
                "resource_fraction", fraction, modified, arcs, candidate_pool
            )
        )

    for multiplier in (0.80, 1.00, 1.20):
        modified = replace(
            data,
            boxes={
                box_id: replace(
                    box,
                    expected_time_s=box.expected_time_s * multiplier,
                    first_deadline_s=(
                        None
                        if box.first_deadline_s is None
                        else box.first_deadline_s * multiplier
                    ),
                )
                for box_id, box in data.boxes.items()
            },
        )
        rows.append(
            _sensitivity_result(
                "deadline_multiplier", multiplier, modified, arcs, candidate_pool
            )
        )
    return pd.DataFrame(rows)


def _save_figure(
    fig: plt.Figure,
    output_dir: Path,
    stem: str,
    *,
    tight_rect: tuple[float, float, float, float] | None = None,
) -> list[Path]:
    paths = [output_dir / f"{stem}.png", output_dir / f"{stem}.pdf"]
    fig.tight_layout(rect=tight_rect)
    fig.savefig(paths[0], dpi=300)
    fig.savefig(paths[1])
    plt.close(fig)
    return paths


def _method_frame(solutions: Mapping[str, Q2Solution]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "method": method,
            "primary": solution.objective[0],
            "makespan": solution.objective[1],
            "energy": solution.objective[2],
            "trips": solution.objective[3],
            "runtime": solution.runtime_s,
        }
        for method, solution in solutions.items()
    )


def create_paper_figures(
    output_dir: Path,
    data: Q2Data,
    solutions: Mapping[str, Q2Solution],
    repetitions: pd.DataFrame,
    convergence: pd.DataFrame,
    sensitivity: pd.DataFrame,
) -> tuple[Path, ...]:
    """Create seven publication figures, each as 300-dpi PNG and vector PDF."""
    _configure_paper_style()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    methods = _method_frame(solutions)
    labels = [METHOD_LABELS.get(method, method) for method in methods["method"]]
    color = "#4477AA"

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2))
    metrics = [
        ("primary", "Weighted delivery", "ratio"),
        ("makespan", "Makespan", "s"),
        ("energy", "Energy", "kWh"),
        ("trips", "Trips", "count"),
    ]
    for axis, (column, title, unit) in zip(axes.flat, metrics):
        values = methods[column].astype(float)
        baseline = max(values.min(), 1e-12)
        normalized = values / baseline
        bars = axis.bar(labels, normalized, color=color, width=0.62)
        axis.set_title(title)
        axis.set_ylabel(f"Normalized ({unit})")
        axis.set_ylim(0, max(1.08, normalized.max() * 1.12))
        axis.tick_params(axis="x", rotation=20)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.3g}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
    fig.suptitle("Method comparison under a common objective", y=1.01)
    created.extend(_save_figure(fig, output_dir, "fig01_method_objectives"))

    fig, axis = plt.subplots(figsize=(7.2, 3.6))
    bars = axis.barh(labels, methods["runtime"], color=color, height=0.56)
    axis.set_xlabel("Wall-clock time (s)")
    axis.set_title("Computational time by method")
    axis.invert_yaxis()
    for bar, value in zip(bars, methods["runtime"]):
        axis.text(value, bar.get_y() + bar.get_height() / 2, f" {value:.2f}", va="center")
    created.extend(_save_figure(fig, output_dir, "fig02_method_runtime"))

    recommended = min(solutions.values(), key=lambda solution: (solution.objective, solution.runtime_s))
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.0), sharex=True)
    for axis, resource_name in zip(axes, ("aircraft", "battery")):
        if resource_name == "aircraft":
            resources = sorted({trip.aircraft_id for trip in recommended.trips})
        else:
            resources = sorted({trip.battery_id for trip in recommended.trips})
        positions = {resource: index for index, resource in enumerate(resources)}
        for trip in recommended.trips:
            resource = trip.aircraft_id if resource_name == "aircraft" else trip.battery_id
            end = trip.return_time_s if resource_name == "aircraft" else trip.battery_ready_time_s
            axis.barh(
                positions[resource],
                (end - trip.start_time_s) / 3600.0,
                left=trip.start_time_s / 3600.0,
                height=0.62,
                color=MODEL_COLORS.get(trip.plan.model_id, color),
                alpha=0.82,
            )
        axis.set_yticks(range(len(resources)), resources)
        axis.set_ylabel(resource_name.title())
        axis.set_title(f"{resource_name.title()} occupancy")
    axes[-1].set_xlabel("Time from dispatch start (h)")
    fig.suptitle(f"Resource Gantt — {METHOD_LABELS.get(recommended.method, recommended.method)}", y=1.01)
    created.extend(_save_figure(fig, output_dir, "fig03_resource_gantt"))

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))
    axes[0].scatter(repetitions["seed"], repetitions["objective_primary"], color=color, s=32)
    axes[0].set_xlabel("Random seed")
    axes[0].set_ylabel("Weighted delivery")
    axes[0].set_title("ALNS solution variation")
    axes[1].scatter(repetitions["seed"], repetitions["runtime_s"], color="#EE6677", s=32)
    axes[1].set_xlabel("Random seed")
    axes[1].set_ylabel("Runtime (s)")
    axes[1].set_title("ALNS runtime variation")
    created.extend(_save_figure(fig, output_dir, "fig04_alns_multiseed"))

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4))
    baseline_levels = {
        "reserve_ratio": 0.20,
        "charge_time_multiplier": 1.00,
        "resource_fraction": 1.00,
        "deadline_multiplier": 1.00,
    }
    legend_items = {}
    for axis, (factor, group) in zip(axes.flat, sensitivity.groupby("factor", sort=True)):
        group = group.sort_values("level")
        feasible = group[group["feasible"]]
        if not feasible.empty:
            baseline_index = (
                feasible["level"] - baseline_levels[factor]
            ).abs().idxmin()
            primary_base = max(float(feasible.loc[baseline_index, "objective_primary"]), 1e-12)
            makespan_base = max(float(feasible.loc[baseline_index, "makespan_s"]), 1e-12)
            axis.plot(
                feasible["level"],
                feasible["objective_primary"] / primary_base,
                marker="o",
                label="Weighted delivery",
                color="#4477AA",
            )
            axis.plot(
                feasible["level"],
                feasible["makespan_s"] / makespan_base,
                marker="s",
                label="Makespan",
                color="#EE6677",
            )
        infeasible = group[~group["feasible"]]
        if not infeasible.empty:
            axis.scatter(infeasible["level"], np.full(len(infeasible), 0.9), marker="x", color="#AA3377", label="Infeasible")
        axis.axhline(1.0, color="0.4", linewidth=0.7, linestyle="--")
        axis.set_title(factor.replace("_", " ").title())
        axis.set_xlabel("Factor level")
        axis.set_ylabel("Normalized response")
        handles, legend_labels = axis.get_legend_handles_labels()
        legend_items.update(zip(legend_labels, handles))
    if legend_items:
        fig.legend(
            legend_items.values(),
            legend_items.keys(),
            loc="upper center",
            bbox_to_anchor=(0.5, 0.955),
            ncol=3,
            frameon=False,
        )
    fig.suptitle("Operational sensitivity analysis", y=1.015)
    created.extend(
        _save_figure(
            fig,
            output_dir,
            "fig05_sensitivity",
            tight_rect=(0.0, 0.0, 1.0, 0.89),
        )
    )

    fig, axis = plt.subplots(figsize=(7.2, 3.6))
    for seed, group in convergence.groupby("seed"):
        group = group.sort_values("iteration")
        axis.plot(
            group["iteration"],
            group["objective_primary"],
            marker="o",
            markersize=3,
            linewidth=1.2,
            label=f"seed {seed}",
        )
    axis.set_xlabel("ALNS iteration")
    axis.set_ylabel("Best weighted delivery")
    axis.set_title("ALNS convergence")
    axis.legend(frameon=False, ncol=min(4, max(1, convergence["seed"].nunique())))
    created.extend(_save_figure(fig, output_dir, "fig06_alns_convergence"))

    fig, axis = plt.subplots(figsize=(6.8, 5.2))
    center = data.nodes["O01"]
    for trip in recommended.trips:
        route = [center] + [data.nodes[stop.service_id] for stop in trip.plan.stops] + [center]
        axis.plot(
            [node.longitude for node in route],
            [node.latitude for node in route],
            color=MODEL_COLORS.get(trip.plan.model_id, color),
            alpha=0.16,
            linewidth=0.8,
        )
    services = [node for node in data.nodes.values() if not node.is_center]
    axis.scatter(
        [node.longitude for node in services],
        [node.latitude for node in services],
        color="#4477AA",
        s=22,
        zorder=3,
        label="Service area",
    )
    axis.scatter([center.longitude], [center.latitude], marker="*", s=100, color="#CC3311", zorder=4, label="Dispatch center")
    for node in services:
        axis.annotate(node.node_id, (node.longitude, node.latitude), xytext=(3, 3), textcoords="offset points", fontsize=7)
    axis.set_xlabel("Longitude (°E)")
    axis.set_ylabel("Latitude (°N)")
    axis.set_title("Recommended dispatch routes")
    axis.legend(frameon=False)
    axis.set_aspect("equal", adjustable="datalim")
    created.extend(_save_figure(fig, output_dir, "fig07_route_map"))
    return tuple(created)
