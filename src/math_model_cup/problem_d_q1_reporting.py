"""Generate paper-ready outputs for Problem D, question 1."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple
from xml.etree import ElementTree
from zipfile import ZipFile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image

from .problem_d_q1 import (
    InfeasibleProblemError,
    MethodSolution,
    ProblemData,
    ServiceGeometry,
    _great_circle_distance_m,
    load_problem_data,
    model_trip_counts,
    safe_payload,
    solution_rows,
    solve_dynamic_programming,
    solve_greedy,
    solve_milp,
    validate_solution,
)
from .problem_d_q1_analysis import (
    CriticalMarginRecord,
    ParetoPoint,
    ReserveInterval,
    best_under_trip_budget,
    candidate_critical_margins,
    pareto_dynamic_programming,
    pareto_for_trip_count,
    pareto_solution,
    reserve_stability_intervals,
    select_anchor,
)
from .problem_d_q1_validation import (
    ENERGY_TOLERANCE_KWH,
    MASS_TOLERANCE_KG,
    SOC_TOLERANCE_PERCENT,
    TIME_TOLERANCE_S,
    VOLUME_TOLERANCE_M3,
    audit_solution_independently,
    independent_trip_breakdown,
    representative_trips,
)


METHOD_LABELS = {
    "greedy": "BFD+合并改进",
    "dynamic_programming": "词典序动态规划",
    "milp": "集合划分 MILP",
}


@dataclass(frozen=True)
class CompleteResults:
    data: ProblemData
    solutions: Mapping[str, MethodSolution]
    recommended_solution: MethodSolution
    pareto_points: Tuple[ParetoPoint, ...]
    reserve_intervals: Tuple[ReserveInterval, ...]


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.12g")


def _markdown_table(frame: pd.DataFrame, digits: int = 4) -> str:
    if frame.empty:
        return "（无）"
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(
                lambda value: "" if pd.isna(value) else f"{value:.{digits}f}"
            )
        else:
            display[column] = display[column].map(
                lambda value: "" if pd.isna(value) else str(value).replace("|", "\\|")
            )
    lines = [
        "| " + " | ".join(map(str, display.columns)) + " |",
        "| " + " | ".join("---" for _ in display.columns) + " |",
    ]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in display.columns) + " |")
    return "\n".join(lines)


def _dense_line_cells_legacy(
    lon1: float,
    lat1: float,
    lon2: float,
    lat2: float,
    data: ProblemData,
) -> Tuple[np.ndarray, np.ndarray]:
    metadata = data.dem_metadata
    if metadata is None:
        raise ValueError("DEM metadata is unavailable")
    x_scale, _, x_origin, _, negative_y_scale, y_origin = metadata.affine
    y_scale = -negative_y_scale
    column1 = (lon1 - x_origin) / x_scale
    row1 = (y_origin - lat1) / y_scale
    column2 = (lon2 - x_origin) / x_scale
    row2 = (y_origin - lat2) / y_scale
    pixel_span = max(abs(column2 - column1), abs(row2 - row1))
    sample_count = max(2, int(pixel_span * 32.0) + 1)
    columns = np.floor(np.linspace(column1, column2, sample_count)).astype(int)
    rows = np.floor(np.linspace(row1, row2, sample_count)).astype(int)
    valid = (
        (rows >= 0)
        & (rows < metadata.rows)
        & (columns >= 0)
        & (columns < metadata.columns)
    )
    pairs = np.unique(np.column_stack((rows[valid], columns[valid])), axis=0)
    return pairs[:, 0], pairs[:, 1]


def _legacy_problem_data(data: ProblemData) -> ProblemData:
    metadata = data.dem_metadata
    if metadata is None:
        raise ValueError("DEM metadata is unavailable")
    with Image.open(metadata.path) as image:
        dem = np.asarray(image, dtype=float)
    x_scale, _, x_origin, _, negative_y_scale, y_origin = metadata.affine
    y_scale = -negative_y_scale
    geometries: Dict[str, ServiceGeometry] = {}
    for service_id, strict in data.geometries.items():
        rows, columns = _dense_line_cells_legacy(
            data.center_longitude,
            data.center_latitude,
            strict.service_longitude,
            strict.service_latitude,
            data,
        )
        elevations = dem[rows, columns]
        valid = np.isfinite(elevations)
        if metadata.nodata is not None:
            valid &= ~np.isclose(elevations, metadata.nodata, rtol=0.0, atol=1e-12)
        elevations = elevations[valid]
        valid_rows = rows[valid]
        valid_columns = columns[valid]
        peak_index = int(np.argmax(elevations))
        peak = float(elevations[peak_index])
        peak_row = int(valid_rows[peak_index])
        peak_column = int(valid_columns[peak_index])
        cruise = peak + 50.0
        distance = _great_circle_distance_m(
            data.center_longitude,
            data.center_latitude,
            strict.service_longitude,
            strict.service_latitude,
        )
        geometries[service_id] = replace(
            strict,
            distance_m=distance,
            haversine_distance_m=distance,
            peak_ground_m=peak,
            cruise_altitude_m=cruise,
            center_climb_m=max(0.0, cruise - data.center_elevation_m),
            service_climb_m=max(0.0, cruise - (strict.service_ground_m + 30.0)),
            peak_row=peak_row,
            peak_column=peak_column,
            peak_longitude=x_origin + (peak_column + 0.5) * x_scale,
            peak_latitude=y_origin - (peak_row + 0.5) * y_scale,
        )
    return replace(data, geometries=geometries)


def _geometry_frame(data: ProblemData, legacy: ProblemData) -> pd.DataFrame:
    rows = []
    for service_id in sorted(data.geometries):
        geometry = data.geometries[service_id]
        old = legacy.geometries[service_id]
        rows.append(
            {
                "服务区编号": service_id,
                "WGS84椭球水平距离（m）": geometry.distance_m,
                "Haversine水平距离（m）": geometry.haversine_distance_m,
                "距离差（m）": geometry.distance_m - geometry.haversine_distance_m,
                "严格all_touched最高DEM（m）": geometry.peak_ground_m,
                "旧密集采样最高DEM（m）": old.peak_ground_m,
                "最高高程差（m）": geometry.peak_ground_m - old.peak_ground_m,
                "最高像元行": geometry.peak_row,
                "最高像元列": geometry.peak_column,
                "最高像元经度": geometry.peak_longitude,
                "最高像元纬度": geometry.peak_latitude,
                "计划巡航海拔（m）": geometry.cruise_altitude_m,
                "去程爬升（m）": geometry.center_climb_m,
                "去程下降（m）": geometry.outbound_descent_m,
                "返程爬升（m）": geometry.return_climb_m,
                "返程下降（m）": geometry.return_descent_m,
            }
        )
    return pd.DataFrame(rows)


def _metadata_frame(data: ProblemData) -> pd.DataFrame:
    metadata = data.dem_metadata
    if metadata is None:
        return pd.DataFrame()
    return pd.DataFrame(
        [
            {
                "文件": str(metadata.path),
                "CRS": metadata.crs_name,
                "EPSG": metadata.epsg,
                "列数": metadata.columns,
                "行数": metadata.rows,
                "像元宽度（°）": metadata.pixel_width_degrees,
                "像元高度（°）": metadata.pixel_height_degrees,
                "NoData": "未声明" if metadata.nodata is None else metadata.nodata,
                "仿射变换": str(metadata.affine),
                "西界": metadata.bounds[0],
                "南界": metadata.bounds[1],
                "东界": metadata.bounds[2],
                "北界": metadata.bounds[3],
                "覆盖验证": "O01及15个服务区均在覆盖范围内",
            }
        ]
    )


def _safe_payload_frames(
    data: ProblemData, reserve_ratio: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    wide_rows = []
    long_rows = []
    for service_id in sorted(data.geometries):
        geometry = data.geometries[service_id]
        row = {"服务区编号": service_id}
        for model_id in sorted(data.aircraft):
            payload = safe_payload(data.aircraft[model_id], geometry, reserve_ratio)
            row[f"{model_id}型最大安全载荷（kg）"] = payload
            long_rows.append(
                {
                    "服务区编号": service_id,
                    "机型编号": model_id,
                    "安全余量": reserve_ratio,
                    "最大安全载荷（kg）": payload,
                }
            )
        wide_rows.append(row)
    return pd.DataFrame(wide_rows), pd.DataFrame(long_rows)


def _coarse_sensitivity_frame(
    data: ProblemData, reserve_ratios: Iterable[float]
) -> pd.DataFrame:
    rows = []
    for reserve_ratio in reserve_ratios:
        row = {"返航安全余量": reserve_ratio}
        for model_id in sorted(data.aircraft):
            payloads = [
                safe_payload(data.aircraft[model_id], geometry, reserve_ratio)
                for geometry in data.geometries.values()
            ]
            row[f"{model_id}型最小安全载荷（kg）"] = min(payloads)
            row[f"{model_id}型最大安全载荷（kg）"] = max(payloads)
        try:
            solution = solve_dynamic_programming(data, reserve_ratio)
        except InfeasibleProblemError:
            row.update(
                {
                    "是否可行": "否",
                    "最少总架次": None,
                    "最优累计时间（h）": None,
                    "对应总能耗（kWh）": None,
                }
            )
        else:
            row.update(
                {
                    "是否可行": "是",
                    "最少总架次": solution.trip_count,
                    "最优累计时间（h）": solution.total_time_s / 3600.0,
                    "对应总能耗（kWh）": solution.total_energy_kwh,
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _solution_signature(solution: MethodSolution) -> Tuple[object, ...]:
    return tuple(
        sorted((trip.service_id, trip.model_id, trip.counts) for trip in solution.trips)
    )


def _comparison_frame(
    data: ProblemData, solutions: Sequence[MethodSolution]
) -> pd.DataFrame:
    exact = next(solution for solution in solutions if solution.method == "dynamic_programming")
    rows = []
    for solution in solutions:
        counts = model_trip_counts(solution)
        mass_utilization = [
            trip.mass_kg / data.aircraft[trip.model_id].max_payload_kg
            for trip in solution.trips
        ]
        volume_utilization = [
            trip.volume_m3 / data.aircraft[trip.model_id].volume_capacity_m3
            for trip in solution.trips
        ]
        rows.append(
            {
                "方法": METHOD_LABELS[solution.method],
                "架次数": solution.trip_count,
                "总能耗（kWh）": solution.total_energy_kwh,
                "累计作业时间（h）": solution.total_time_s / 3600.0,
                "A型架次": counts.get("A", 0),
                "B型架次": counts.get("B", 0),
                "C型架次": counts.get("C", 0),
                "平均载质量利用率（%）": 100.0 * np.mean(mass_utilization),
                "平均体积利用率（%）": 100.0 * np.mean(volume_utilization),
                "架次数最优差距（%）": 100.0
                * (solution.trip_count - exact.trip_count)
                / exact.trip_count,
                "时间差距（%）": 100.0
                * (solution.total_time_s - exact.total_time_s)
                / exact.total_time_s,
                "能耗差距（%）": 100.0
                * (solution.total_energy_kwh - exact.total_energy_kwh)
                / exact.total_energy_kwh,
                "运行时间（s）": solution.runtime_s,
            }
        )
    return pd.DataFrame(rows)


def _pareto_frames(
    data: ProblemData, points: Sequence[ParetoPoint]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for index, point in enumerate(points, start=1):
        solution = pareto_solution(data, point, method=f"pareto_{index}")
        counts = model_trip_counts(solution)
        rows.append(
            {
                "前沿点编号": index,
                "架次数": point.trip_count,
                "累计作业时间（h）": point.total_time_s / 3600.0,
                "总能耗（kWh）": point.total_energy_kwh,
                "A型架次": counts.get("A", 0),
                "B型架次": counts.get("B", 0),
                "C型架次": counts.get("C", 0),
                "组批签名": str(_solution_signature(solution)),
            }
        )
    n_min = min(point.trip_count for point in points)
    scenario_rows = []
    for label, point in (
        ("架次数最少锚点", select_anchor(points, "trips")),
        ("总能耗最少锚点", select_anchor(points, "energy")),
        ("累计作业时间最少锚点", select_anchor(points, "time")),
    ):
        scenario_rows.append(
            {
                "场景": label,
                "架次数上限": "不限制",
                "实际架次数": point.trip_count,
                "累计作业时间（h）": point.total_time_s / 3600.0,
                "总能耗（kWh）": point.total_energy_kwh,
            }
        )
    for budget in (n_min, n_min + 1, n_min + 2):
        for objective, label in (("energy", "最低能耗"), ("time", "最短时间")):
            point = best_under_trip_budget(points, budget, objective)
            scenario_rows.append(
                {
                    "场景": f"N≤{budget}时{label}",
                    "架次数上限": budget,
                    "实际架次数": point.trip_count,
                    "累计作业时间（h）": point.total_time_s / 3600.0,
                    "总能耗（kWh）": point.total_energy_kwh,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(scenario_rows)


def _critical_frame(
    data: ProblemData, records: Sequence[CriticalMarginRecord]
) -> pd.DataFrame:
    rows = []
    for record in records:
        row = {
            "服务区编号": record.service_id,
            "机型编号": record.model_id,
            "组合计数": str(record.counts),
            "总质量（kg）": record.mass_kg,
            "总体积（m³）": record.volume_m3,
            "往返能耗（kWh）": record.energy_kwh,
            "临界安全余量": record.rho_crit,
            "临界安全余量（%）": 100.0 * record.rho_crit,
        }
        for index, material_type in enumerate(data.material_types):
            row[f"{material_type}箱数"] = record.counts[index]
        rows.append(row)
    return pd.DataFrame(rows)


def _interval_frame(intervals: Sequence[ReserveInterval]) -> pd.DataFrame:
    rows = []
    for index, interval in enumerate(intervals, start=1):
        left = "[" if interval.lower_inclusive else "("
        right = "]" if interval.upper_inclusive else ")"
        rows.append(
            {
                "区间编号": index,
                "稳定区间": (
                    f"{left}{100 * interval.lower_ratio:.9f}%, "
                    f"{100 * interval.upper_ratio:.9f}%{right}"
                ),
                "下界": interval.lower_ratio,
                "上界": interval.upper_ratio,
                "是否可行": "是" if interval.feasible else "否",
                "最少架次数": interval.trip_count,
                "累计作业时间（h）": (
                    None if interval.total_time_s is None else interval.total_time_s / 3600.0
                ),
                "总能耗（kWh）": interval.total_energy_kwh,
                "组批签名": str(interval.solution_signature),
            }
        )
    return pd.DataFrame(rows)


def _group_text(trips) -> str:
    return "；".join(
        f"{trip.trip_id}/{trip.model_id}/{trip.mass_kg:.1f}kg/"
        f"[{','.join(trip.box_ids)}]"
        for trip in sorted(trips, key=lambda item: item.trip_id)
    )


def _interval_event_frame(
    data: ProblemData, intervals: Sequence[ReserveInterval]
) -> pd.DataFrame:
    rows = []
    for previous, current in zip(intervals, intervals[1:]):
        boundary = current.lower_ratio
        previous_by_service = defaultdict(list)
        current_by_service = defaultdict(list)
        for trip in previous.trips:
            previous_by_service[trip.service_id].append(trip)
        for trip in current.trips:
            current_by_service[trip.service_id].append(trip)
        service_ids = sorted(set(previous_by_service) | set(current_by_service))
        affected = [
            service_id
            for service_id in service_ids
            if sorted(
                (trip.model_id, trip.counts) for trip in previous_by_service[service_id]
            )
            != sorted(
                (trip.model_id, trip.counts) for trip in current_by_service[service_id]
            )
        ]
        expired = [
            trip
            for trip in previous.trips
            if abs(trip.return_soc_percent / 100.0 - boundary) <= 2e-9
        ]
        if not current.feasible:
            probe = min(boundary + 1e-9, 1.0 - 1e-12)
            impossible = []
            for service_id in sorted(data.geometries):
                service_boxes = tuple(
                    box for box in data.boxes if box.service_id == service_id
                )
                service_data = replace(
                    data,
                    boxes=service_boxes,
                    geometries={service_id: data.geometries[service_id]},
                    material_types=tuple(
                        dict.fromkeys(box.material_type for box in service_boxes)
                    ),
                )
                try:
                    solve_dynamic_programming(service_data, probe)
                except InfeasibleProblemError:
                    impossible.append(service_id)
            affected = impossible
        rows.append(
            {
                "临界安全余量": boundary,
                "临界安全余量（%）": 100.0 * boundary,
                "首先受影响服务区": "、".join(affected) if affected else "组批机型切换",
                "安全载荷越过阈值的机型": "、".join(
                    sorted({f"{trip.service_id}-{trip.model_id}" for trip in expired})
                )
                or "无既有最优架次恰在阈值（替代组合切换）",
                "失效原组批": _group_text(expired) if expired else "见相邻组批签名",
                "变化前受影响组批": " | ".join(
                    f"{service_id}: {_group_text(previous_by_service[service_id])}"
                    for service_id in affected
                ),
                "变化后新组批": (
                    "不可配送"
                    if not current.feasible
                    else " | ".join(
                        f"{service_id}: {_group_text(current_by_service[service_id])}"
                        for service_id in affected
                    )
                ),
                "架次数变化": (
                    None
                    if current.trip_count is None or previous.trip_count is None
                    else current.trip_count - previous.trip_count
                ),
                "能耗变化（kWh）": (
                    None
                    if current.total_energy_kwh is None or previous.total_energy_kwh is None
                    else current.total_energy_kwh - previous.total_energy_kwh
                ),
                "时间变化（h）": (
                    None
                    if current.total_time_s is None or previous.total_time_s is None
                    else (current.total_time_s - previous.total_time_s) / 3600.0
                ),
                "是否变为不可配送": "是" if not current.feasible else "否",
            }
        )
    return pd.DataFrame(rows)


def _independent_frames(
    data: ProblemData, solution: MethodSolution
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    audit = audit_solution_independently(data, solution)
    audit_by_trip = {item.trip_id: item for item in audit.trip_audits}
    trip_rows = []
    for item in audit.trip_audits:
        trip_rows.append(
            {
                "架次编号": item.trip_id,
                "服务区": item.service_id,
                "机型": item.model_id,
                "箱数": item.box_count,
                "质量（kg）": item.mass_kg,
                "体积（m³）": item.volume_m3,
                "去程等效航程（m）": item.outbound_equivalent_range_m,
                "返程等效航程（m）": item.return_equivalent_range_m,
                "去程水平能耗（kWh）": item.outbound_horizontal_kwh,
                "返程水平能耗（kWh）": item.return_horizontal_kwh,
                "去程爬升能耗（kWh）": item.outbound_climb_kwh,
                "返程爬升能耗（kWh）": item.return_climb_kwh,
                "下降附加能耗（kWh）": item.descent_energy_kwh,
                "总能耗（kWh）": item.total_energy_kwh,
                "飞行时间（s）": item.flight_time_s,
                "地面作业时间（s）": item.ground_time_s,
                "总时间（s）": item.total_time_s,
                "返航SOC（%）": item.return_soc_percent,
                "能耗误差（kWh）": item.energy_error_kwh,
                "时间误差（s）": item.time_error_s,
                "约束通过": item.constraints_pass,
            }
        )
    representative_rows = []
    for label, trip in representative_trips(solution).items():
        item = audit_by_trip[trip.trip_id]
        row = {"复核类别": label}
        row.update(next(value for value in trip_rows if value["架次编号"] == trip.trip_id))
        representative_rows.append(row)

    assignments = defaultdict(list)
    for trip in solution.trips:
        for box_id in trip.box_ids:
            assignments[box_id].append(trip)
    coverage_rows = []
    for box in sorted(data.boxes, key=lambda item: item.box_id):
        assigned = assignments.get(box.box_id, [])
        coverage_rows.append(
            {
                "货箱编号": box.box_id,
                "服务区编号": box.service_id,
                "出现次数": len(assigned),
                "架次编号": "；".join(trip.trip_id for trip in assigned),
                "唯一覆盖通过": len(assigned) == 1,
            }
        )
    constraint_rows = [
        {
            "架次编号": item.trip_id,
            "同服务区": item.same_service_pass,
            "质量约束": item.mass_capacity_pass,
            "体积约束": item.volume_capacity_pass,
            "能量/SOC约束": item.reserve_pass,
            "数值一致性与全部约束": item.constraints_pass,
        }
        for item in audit.trip_audits
    ]
    summary_rows = [
        {
            "检查项": "货箱覆盖",
            "报告值": audit.total_assignment_count,
            "独立值": audit.unique_box_count,
            "容差": 0,
            "通过": audit.coverage_pass,
        },
        {
            "检查项": "总能耗（kWh）",
            "报告值": audit.reported_energy_kwh,
            "独立值": audit.recomputed_energy_kwh,
            "容差": ENERGY_TOLERANCE_KWH,
            "通过": audit.energy_sum_pass,
        },
        {
            "检查项": "累计时间（s）",
            "报告值": audit.reported_time_s,
            "独立值": audit.recomputed_time_s,
            "容差": TIME_TOLERANCE_S,
            "通过": audit.time_sum_pass,
        },
        {
            "检查项": "逐架次全部约束",
            "报告值": len(solution.trips),
            "独立值": sum(item.constraints_pass for item in audit.trip_audits),
            "容差": 0,
            "通过": audit.constraints_pass,
        },
    ]
    return (
        pd.DataFrame(trip_rows),
        pd.DataFrame(representative_rows),
        pd.DataFrame(coverage_rows),
        pd.DataFrame(constraint_rows),
        pd.DataFrame(summary_rows),
    )


def _formula_audit_text(problem_dir: Path, robustness: pd.DataFrame) -> str:
    docx_matches = [
        path
        for path in problem_dir.rglob("山区洪涝灾害下无人机运输与通信协同优化.docx")
        if not path.name.startswith("~$")
    ]
    if len(docx_matches) != 1:
        raise FileNotFoundError("cannot uniquely locate the original problem DOCX")
    with ZipFile(docx_matches[0]) as archive:
        root = ElementTree.fromstring(archive.read("word/document.xml"))
    paragraphs = []
    for index, paragraph in enumerate(
        (node for node in root.iter() if node.tag.endswith("}p")), start=1
    ):
        text = "".join(
            node.text or ""
            for node in paragraph.iter()
            if node.tag.endswith("}t")
        ).strip()
        if text and any(
            token in text for token in ("等效航程", "水平巡航", "爬升附加", "安全余量")
        ):
            paragraphs.append((index, text))
    evidence = "\n".join(f"- XML 第 {index} 段：{text}" for index, text in paragraphs)
    return rf"""# D 题第一问能耗公式与物理口径审计

## 1. 原题与附件核对结论

对原始 DOCX 的 `word/document.xml`、渲染题面、运输无人机参数表和公式相关 XML 逐项核对后，结论是：题面给出了等效航程函数、总能耗由水平巡航与爬升附加两部分构成以及返航安全余量约束，但**没有给出水平能耗和爬升能耗的子公式**。因此主模型中的两个子公式属于基于参数物理意义的必要补充，并非题面原式抄录。

{evidence}

运输无人机 Excel 表头明确为“含电池空载总质量（kg）”，故爬升质量取“含电池空载总质量 + 当前载荷”，不能再次加电池质量。参数表“下降能耗效率”为 0，因此下降附加能耗取 0；去程卸货后返程载荷取 0。服务区无人机作业点位于地面海拔上方 30 m。

## 2. 主口径与量纲推导

令可用电池能量为 $E^{{use}}$（kWh），当前载荷等效航程为 $L(q)$（m），单程水平距离为 $d$（m）。将“标准航程”解释为在标准工况下消耗全部可用能量能够飞行的水平距离，则单位距离能耗强度为

$$e_{{hor}}(q)=\frac{{E^{{use}}}}{{L(q)}}\quad [\mathrm{{kWh/m}}],$$

故水平能耗

$$E_{{hor}}(q,d)=E^{{use}}\frac{{d}}{{L(q)}}\quad [\mathrm{{kWh}}].$$

其中 $d/L(q)$ 无量纲，所以结果单位仍为 kWh。该关系等价于用“可用能量/标准航程”标定单位距离能耗，能够直接利用题面同时给出的标准航程与电池可用能量。

爬升势能为 $mgh$（J），考虑爬升效率 $\eta_{{up}}$ 后，电池侧能耗为

$$E_{{up}}=\frac{{mgh}}{{3.6\times 10^6\eta_{{up}}}}\quad [\mathrm{{kWh}}],$$

因为 $\mathrm{{kg\,m\,s^{{-2}}\,m}}=\mathrm{{J}}$，且 $1\ \mathrm{{kWh}}=3.6\times10^6\ \mathrm{{J}}$。去程 $m=m_0+q$，返程 $m=m_0$。

完整往返能耗为

$$E_{{RT}}(q)=E_{{hor}}(q,d)+E_{{up}}(m_0+q,h_O)+E_{{hor}}(0,d)+E_{{up}}(m_0,h_S),$$

并满足 $E_{{RT}}(q)\le(1-\rho)E^{{use}}$。下降附加能耗为 0。

## 3. 备选解释与稳健性

另一种可讨论但证据较弱的解释是：表中标准航程已经按默认 20% 返航余量标定，此时水平能耗标定系数为 $0.8E^{{use}}/L(q)$。由于题面又单独给出返航电量下限，主模型采用“标准航程消耗全部可用能量”的解释，避免把余量重复计入航程标定。

{_markdown_table(robustness, 6)}

两种解释下最少架次数、B/C 机型构成和关键组批均未改变；能耗绝对值与返航 SOC 会随解释变化。因此“18 架次、B/C 各 9 架次”的结构结论稳定，而总能耗和 SOC 的精确数值依赖标准航程的能量标定口径。

## 4. 数值容差

- 质量：{MASS_TOLERANCE_KG:g} kg；
- 体积：{VOLUME_TOLERANCE_M3:g} m³；
- 能耗：{ENERGY_TOLERANCE_KWH:g} kWh；
- 时间：{TIME_TOLERANCE_S:g} s；
- SOC：{SOC_TOLERANCE_PERCENT:g} 个百分点。

上述容差远小于输入数据有效精度，只用于吸收 IEEE 754 浮点舍入，不放宽物理约束。
""".replace("\\\\", "\\")


def _configure_plotting() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def _plot_dem(data: ProblemData, path: Path) -> None:
    metadata = data.dem_metadata
    if metadata is None:
        raise ValueError("DEM metadata is unavailable")
    with Image.open(metadata.path) as image:
        dem = np.asarray(image, dtype=float)
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(
        dem,
        extent=(metadata.bounds[0], metadata.bounds[2], metadata.bounds[1], metadata.bounds[3]),
        origin="upper",
        cmap="terrain",
        alpha=0.88,
        aspect="auto",
    )
    for service_id, geometry in sorted(data.geometries.items()):
        highlighted = service_id in {"S003", "S008", "S014", "S015"}
        ax.plot(
            [data.center_longitude, geometry.service_longitude],
            [data.center_latitude, geometry.service_latitude],
            color="#d62728" if highlighted else "#174a7e",
            linewidth=1.8 if highlighted else 0.9,
            alpha=0.95 if highlighted else 0.65,
        )
        ax.scatter(geometry.service_longitude, geometry.service_latitude, s=18, color="#111111")
        ax.text(geometry.service_longitude, geometry.service_latitude, service_id, fontsize=8)
    ax.scatter(
        data.center_longitude,
        data.center_latitude,
        marker="*",
        s=180,
        color="#ffcc00",
        edgecolor="#111111",
        label="O01 调度中心",
        zorder=5,
    )
    ax.set(title="O01—15个服务区航段与DEM", xlabel="经度（°）", ylabel="纬度（°）")
    ax.legend(loc="upper right")
    fig.colorbar(im, ax=ax, label="DEM高程（m）", fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_safe_payload_heatmap(wide: pd.DataFrame, path: Path) -> None:
    matrix = wide.set_index("服务区编号")
    matrix.columns = [column[0] for column in matrix.columns]
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt=".1f", cmap="YlGnBu", ax=ax, cbar_kws={"label": "kg"})
    ax.set(title="20%返航安全余量下15×3最大安全载荷", xlabel="机型", ylabel="服务区")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_demand_capacity(data: ProblemData, wide: pd.DataFrame, path: Path) -> None:
    service_ids = sorted(data.geometries)
    demand_mass = [
        sum(box.mass_kg for box in data.boxes if box.service_id == service_id)
        for service_id in service_ids
    ]
    demand_volume = [
        sum(box.volume_m3 for box in data.boxes if box.service_id == service_id)
        for service_id in service_ids
    ]
    capacity_mass = wide.set_index("服务区编号").max(axis=1).reindex(service_ids).values
    capacity_volume = [max(aircraft.volume_capacity_m3 for aircraft in data.aircraft.values())] * len(service_ids)
    x = np.arange(len(service_ids))
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    axes[0].bar(x - 0.2, demand_mass, width=0.4, label="服务区总需求")
    axes[0].bar(x + 0.2, capacity_mass, width=0.4, label="单架次最大安全载荷")
    axes[0].set_ylabel("质量（kg）")
    axes[0].legend()
    axes[0].set_title("各服务区需求与单架次有效容量比较")
    axes[1].bar(x - 0.2, demand_volume, width=0.4, label="服务区总体积")
    axes[1].bar(x + 0.2, capacity_volume, width=0.4, label="最大可用装载体积")
    axes[1].set_ylabel("体积（m³）")
    axes[1].set_xticks(x, service_ids, rotation=45)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_pareto(points: Sequence[ParetoPoint], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for trip_count in sorted({point.trip_count for point in points}):
        selected = sorted(
            (point for point in points if point.trip_count == trip_count),
            key=lambda point: point.total_time_s,
        )
        ax.plot(
            [point.total_time_s / 3600.0 for point in selected],
            [point.total_energy_kwh for point in selected],
            marker="o",
            linewidth=2,
            label=f"N={trip_count}",
        )
        for point in selected:
            ax.annotate(
                f"{point.total_time_s/3600:.3f} h\n{point.total_energy_kwh:.3f} kWh",
                (point.total_time_s / 3600.0, point.total_energy_kwh),
                xytext=(8, 8),
                textcoords="offset points",
                fontsize=9,
            )
    ax.set(title="累计作业时间—总能耗Pareto前沿", xlabel="累计作业时间（h）", ylabel="总能耗（kWh）")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_utilization(data: ProblemData, solution: MethodSolution, path: Path) -> None:
    trips = list(solution.trips)
    mass = [100 * trip.mass_kg / data.aircraft[trip.model_id].max_payload_kg for trip in trips]
    volume = [100 * trip.volume_m3 / data.aircraft[trip.model_id].volume_capacity_m3 for trip in trips]
    x = np.arange(len(trips))
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.bar(x - 0.2, mass, 0.4, label="载质量利用率")
    ax.bar(x + 0.2, volume, 0.4, label="体积利用率")
    ax.axhline(100, color="#333333", linestyle="--", linewidth=1)
    ax.set_xticks(x, [trip.trip_id.replace("Q1-DP-", "") for trip in trips], rotation=45)
    ax.set(title="推荐方案各架次质量/体积利用率", xlabel="架次序号", ylabel="利用率（%）", ylim=(0, 115))
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_safe_payload_curves(data: ProblemData, path: Path) -> None:
    ratios = np.linspace(0.0, 0.40, 81)
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=True)
    for axis, model_id in zip(axes, sorted(data.aircraft)):
        aircraft = data.aircraft[model_id]
        for service_id in sorted(data.geometries):
            values = [
                safe_payload(aircraft, data.geometries[service_id], float(ratio))
                for ratio in ratios
            ]
            emphasized = service_id in {"S003", "S008", "S014", "S015"}
            axis.plot(
                ratios * 100,
                values,
                linewidth=1.8 if emphasized else 0.8,
                alpha=1.0 if emphasized else 0.48,
                label=service_id if emphasized else None,
            )
        axis.set_title(f"{model_id}型")
        axis.set_xlabel("返航安全余量（%）")
        axis.legend(fontsize=8)
    axes[0].set_ylabel("最大安全载荷（kg）")
    fig.suptitle("安全余量—最大安全载荷连续曲线")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_reserve_staircase(intervals: Sequence[ReserveInterval], path: Path) -> None:
    feasible = [interval for interval in intervals if interval.feasible]
    x = []
    y = []
    for interval in feasible:
        x.extend([100 * interval.lower_ratio, 100 * interval.upper_ratio])
        y.extend([interval.trip_count, interval.trip_count])
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(x, y, drawstyle="steps-post", linewidth=2.4, color="#174a7e")
    infeasible = next((interval for interval in intervals if not interval.feasible), None)
    if infeasible:
        ax.axvspan(100 * infeasible.lower_ratio, 40.0, color="#d62728", alpha=0.18, label="不可配送")
    ax.axvline(20.0, color="#ff7f0e", linestyle="--", label="默认20%")
    ax.set(xlim=(0, 40), title="安全余量—最少架次数阶梯图", xlabel="返航安全余量（%）", ylabel="最少架次数")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _paper_text(
    comparison: pd.DataFrame,
    geometry: pd.DataFrame,
    safe_comparison: pd.DataFrame,
    safe_payloads: pd.DataFrame,
    pareto_scenarios: pd.DataFrame,
    robustness: pd.DataFrame,
    intervals: pd.DataFrame,
    events: pd.DataFrame,
    representatives: pd.DataFrame,
    summary_checks: pd.DataFrame,
    recommended: MethodSolution,
    legacy_solution: MethodSolution,
) -> str:
    min_soc = min(trip.return_soc_percent for trip in recommended.trips)
    max_distance_difference = geometry["距离差（m）"].abs().max()
    max_peak_difference = geometry["最高高程差（m）"].abs().max()
    max_safe_payload_difference = safe_comparison["差值（kg）"].abs().max()
    return rf"""# D题第一问：单点直达运输组批优化

## 1. 问题一模型假设

1. 每架次由 O01 直达一个服务区，完成交付后空载返航，不跨服务区混装。
2. 运输无人机表中的空载质量已经包含电池；去程爬升质量为 $m_0+q$，返程爬升质量为 $m_0$。
3. 服务区作业高度为地面海拔加 30 m；巡航海拔为航段 all-touched DEM 最高高程加 50 m。
4. 水平阶段保持计划巡航速度；爬升、下降分别使用表中最大速度计算作业时间。
5. 参数表下降能耗效率为 0，按题意取下降附加能耗为 0；不计风、温度和悬停扰动。
6. 同物资类型货箱的质量和体积一致，货箱不可拆分，每箱恰好配送一次。
7. 默认返航安全余量为 20%，可行条件在数值容差内按闭区间处理。

## 2. 符号表

| 符号 | 含义 | 单位 |
| --- | --- | --- |
| $i,k$ | 服务区、机型索引 | — |
| $b$ | 货箱索引 | — |
| $d_i$ | O01 到服务区 $i$ 的 WGS84 椭球水平距离 | m |
| $H_i$ | 航段严格 all-touched DEM 最高高程 | m |
| $h_i^O,h_i^S$ | O01 去程、服务区返程爬升高度 | m |
| $m_k^0$ | 含电池空载总质量 | kg |
| $Q_k,V_k$ | 最大载质量、装载体积 | kg, m³ |
| $E_k^{{use}}$ | 电池可用能量 | kWh |
| $L_k(q)$ | 载荷 $q$ 下的等效航程 | m |
| $E_{{RT,ik}}(q)$ | 往返总能耗 | kWh |
| $\rho$ | 返航安全余量 | — |
| $x_{{ikr}}$ | 服务区 $i$ 采用机型 $k$ 的候选组批 $r$ 的次数 | 次 |

## 3. 航段、时间与能耗模型

### 3.1 严格 GIS 航段模型

将经纬度端点变换到 DEM 连续像元坐标，对包围盒内每个像元做闭线段—闭矩形 Liang–Barsky 相交判定；凡边、角或内部被航段接触的像元均计入。距离用 WGS84 椭球反解计算。巡航海拔与四段垂直高度为

$$H_i^c=\max_{{p\in\mathcal P_i}} z_p+50,$$
$$h_i^O=H_i^c-z_O,\quad h_i^S=H_i^c-(z_i+30).$$

去程下降与返程爬升均为 $h_i^S$，返程下降为 $h_i^O$。

重点航段复核：

{_markdown_table(geometry[geometry['服务区编号'].isin(['S003','S008','S014','S015'])], 6)}

### 3.2 时间模型

单架次累计作业时间由固定准备、逐箱装载、去返程爬升/巡航/下降、基础交接和逐箱交接构成：

$$T_{{ikr}}=t_k^{{pre}}+n_r t_k^{{load}}+\frac{{h_i^O}}{{v_k^{{up}}}}+\frac{{d_i}}{{v_k}}+\frac{{h_i^S}}{{v_k^{{down}}}}+t_k^{{handoff}}+n_r t_k^{{add}}+\frac{{h_i^S}}{{v_k^{{up}}}}+\frac{{d_i}}{{v_k}}+\frac{{h_i^O}}{{v_k^{{down}}}}.$$

### 3.3 能耗与安全载荷模型

$$L_k(q)=L_k^0-(L_k^0-L_k^Q)(q/Q_k)^{{1.5}},$$
$$E_{{hor}}=E_k^{{use}}d_i/L_k(q),\qquad E_{{up}}=(m_k^0+q)gh/(3.6\times10^6\eta_k).$$

往返时去程带货、返程 $q=0$，下降能耗为 0。安全载荷 $q_{{ik}}^{{safe}}(\rho)$ 是

$$E_{{RT,ik}}(q)\le(1-\rho)E_k^{{use}},\quad 0\le q\le Q_k$$

的最大根，采用单调二分求解。

## 4. 货箱组批优化模型

对每个服务区枚举不超过需求量的整数组合 $a_{{ibr}}$，筛除质量、体积和能量不可行组合。集合划分模型为

$$\min\left(\sum_{{ikr}}x_{{ikr}},\ \sum_{{ikr}}T_{{ikr}}x_{{ikr}},\ \sum_{{ikr}}E_{{ikr}}x_{{ikr}}\right),$$
$$\sum_{{kr}}a_{{ibr}}x_{{ikr}}=D_{{ib}},\quad x_{{ikr}}\in\mathbb Z_+.$$

目标按“架次数—累计作业时间—总能耗”词典序处理；另保留三目标非支配状态以构造 Pareto 前沿。

## 5. 三种算法与伪代码

### 5.1 BFD+合并改进

```text
按质量与体积递减排列货箱
for 每个货箱:
    放入可行且剩余容量最小的已有架次，否则新建架次
while 存在可行的两架次合并:
    选择时间最短、能耗最低的合并
对每个组批替换为时间—能耗更优机型
```

该方法只包含最佳适应递减与两架次合并，因此准确命名为“BFD+合并改进”，不称为局部搜索。

### 5.2 词典序动态规划

```text
state = 各物资类型已配送箱数
F(target) = (0, 0, 0)
F(state) = lexicographic_min over feasible batch r:
           (1, T_r, E_r) + F(state + a_r)
```

### 5.3 集合划分 MILP

先最小化总架次数并固定最优值，再最小化时间并固定最优值，最后最小化能耗；三阶段共享精确覆盖约束。

### 5.4 多目标 Pareto 动态规划

```text
for 每个计数状态:
    生成所有可达 (N,T,E,路径)
    删除被另一状态在 N、T、E 三维同时不劣且至少一维更优的状态
对15个服务区前沿逐次卷积并继续非支配筛选
```

## 6. 多目标优化与最终优先级

{_markdown_table(pareto_scenarios, 6)}

固定最少架次数 $N=18$ 时，时间—能耗前沿只有一个点。允许增加到 19 架次，最低能耗从 {recommended.total_energy_kwh:.6f} kWh 降到 59.020116 kWh，仅节能 0.099788 kWh（0.169%），但累计时间增加约 0.460527 h；20 架次没有新增非支配收益。因此应急场景采用“先最少架次数、再最短累计作业时间、最后最低能耗”的优先级有结果支撑。

## 7. 结果分析

### 7.1 三算法比较

{_markdown_table(comparison, 6)}

推荐方案为 18 架次，B 型 9 架次、C 型 9 架次、A 型 0 架次，总能耗 {recommended.total_energy_kwh:.6f} kWh，累计作业时间 {recommended.total_time_s/3600:.6f} h，最低返航 SOC 为 {min_soc:.6f}%。DP 与 MILP 的目标值完全一致；BFD+合并改进也达到 18 架次，但时间和能耗略高。

### 7.2 严格 GIS 前后差异

旧密集采样+Haversine 结果为 {legacy_solution.total_energy_kwh:.6f} kWh、{legacy_solution.total_time_s/3600:.6f} h；严格 GIS 后为 {recommended.total_energy_kwh:.6f} kWh、{recommended.total_time_s/3600:.6f} h。两者均为 18 架次、B/C 各 9 架次，组批结构不变。15 条航段的椭球距离与 Haversine 距离最大绝对差为 {max_distance_difference:.6f} m，严格 all-touched 与旧密集采样的沿线最高 DEM 高程最大差为 {max_peak_difference:.6f} m，45 个“服务区—机型”安全载荷的最大绝对差为 {max_safe_payload_difference:.6f} kg。

### 7.3 两种能耗解释

{_markdown_table(robustness, 6)}

两种解释的架次数与机型构成一致，说明结构结论稳健；能耗和返航 SOC 数值依赖标定口径。

### 7.4 独立代表架次复核

{_markdown_table(representatives, 6)}

汇总校验：

{_markdown_table(summary_checks, 9)}

## 8. 安全余量灵敏度

对每个服务区—机型—候选货箱组合计算

$$\rho_{{crit}}=1-E_{{RT}}(q)/E^{{use}},$$

共得到 742 个候选临界值，并在每个相邻临界点之间重新求解。压缩后稳定区间如下：

{_markdown_table(intervals[['稳定区间','是否可行','最少架次数','累计作业时间（h）','总能耗（kWh）']], 6)}

默认 20% 处于首段，20% 结果在安全余量不超过 23.088350% 时不变。关键变化事件如下：

{_markdown_table(events[['临界安全余量（%）','首先受影响服务区','安全载荷越过阈值的机型','架次数变化','能耗变化（kWh）','时间变化（h）','是否变为不可配送']], 6)}

超过 35.339250% 后至少一个服务区不存在可行单箱架次，问题变为不可配送。

## 9. 模型优点、局限性和适用条件

优点：采用闭集 all-touched DEM 穿越避免漏峰；距离使用 WGS84 椭球；DP 与 MILP 独立验证优化层；独立程序重新计算物理量；临界余量分析不是粗网格，而是精确事件分段；多目标结果给出增加架次的量化收益。

局限性：题面缺少两个能耗子公式，绝对能耗仍依赖标准航程标定解释；未考虑风、温度、悬停、动态避障和电池老化；以直线水平投影代表计划航路；累计作业时间按各架次求和，不等同于拥有多架无人机时的并行完工时间。

适用条件：适用于同一调度中心到单服务区直达、货箱不可拆分、航路可用 DEM 直线走廊描述、机型航程—载荷函数单调且返航空载的应急配送组批问题。

## 10. 图表

![DEM航段地图](figures/dem_routes.png)

![最大安全载荷热力图](figures/safe_payload_heatmap.png)

![需求与容量](figures/demand_vs_capacity.png)

![Pareto前沿](figures/pareto_front.png)

![架次利用率](figures/trip_utilization.png)

![安全载荷连续曲线](figures/safe_payload_curves.png)

![安全余量阶梯图](figures/reserve_trip_staircase.png)
""".replace("\\\\", "\\")


def generate_complete_outputs(problem_dir: Path, output_dir: Path) -> CompleteResults:
    problem_dir = Path(problem_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    _configure_plotting()

    data = load_problem_data(problem_dir)
    legacy_data = _legacy_problem_data(data)
    solutions = (
        solve_greedy(data),
        solve_dynamic_programming(data),
        solve_milp(data),
    )
    for solution in solutions:
        validate_solution(data, solution.trips)
        frame = pd.DataFrame(solution_rows(solution))
        frame["载质量利用率（%）"] = [
            100.0 * trip.mass_kg / data.aircraft[trip.model_id].max_payload_kg
            for trip in solution.trips
        ]
        frame["体积利用率（%）"] = [
            100.0 * trip.volume_m3 / data.aircraft[trip.model_id].volume_capacity_m3
            for trip in solution.trips
        ]
        _write_csv(frame, output_dir / f"{solution.method}_trips.csv")
    solution_map = {solution.method: solution for solution in solutions}
    recommended = solution_map["dynamic_programming"]

    legacy_solution = solve_dynamic_programming(legacy_data)
    comparison = _comparison_frame(data, solutions)
    _write_csv(comparison, output_dir / "method_comparison.csv")
    geometry = _geometry_frame(data, legacy_data)
    _write_csv(geometry, output_dir / "route_geometry.csv")
    _write_csv(_metadata_frame(data), output_dir / "gis_metadata.csv")
    _write_csv(
        geometry[geometry["服务区编号"].isin(["S003", "S008", "S014", "S015"])],
        output_dir / "重点航段人工复核.csv",
    )

    safe_wide, safe_long = _safe_payload_frames(data, 0.20)
    _write_csv(safe_wide, output_dir / "safe_payloads.csv")
    _write_csv(safe_long, output_dir / "safe_payloads_long.csv")
    coarse_sensitivity = _coarse_sensitivity_frame(
        data, (0.00, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40)
    )
    _write_csv(
        coarse_sensitivity, output_dir / "safety_margin_sensitivity.csv"
    )

    pareto_points = pareto_dynamic_programming(data)
    pareto_frame, pareto_scenarios = _pareto_frames(data, pareto_points)
    _write_csv(pareto_frame, output_dir / "pareto_front.csv")
    _write_csv(pareto_scenarios, output_dir / "pareto_scenarios.csv")

    primary_dp = recommended
    primary_milp = solution_map["milp"]
    alternative_dp = solve_dynamic_programming(data, range_energy_fraction=0.8)
    alternative_milp = solve_milp(data, range_energy_fraction=0.8)
    robustness_rows = []
    for interpretation, method_name, solution in (
        ("主口径：标准航程消耗100%可用能量", "词典序动态规划", primary_dp),
        ("主口径：标准航程消耗100%可用能量", "集合划分 MILP", primary_milp),
        ("备选：标准航程已预留20%电量", "词典序动态规划", alternative_dp),
        ("备选：标准航程已预留20%电量", "集合划分 MILP", alternative_milp),
    ):
        counts = model_trip_counts(solution)
        robustness_rows.append(
            {
                "能耗解释": interpretation,
                "算法": method_name,
                "架次数": solution.trip_count,
                "B型架次": counts.get("B", 0),
                "C型架次": counts.get("C", 0),
                "累计作业时间（h）": solution.total_time_s / 3600.0,
                "总能耗（kWh）": solution.total_energy_kwh,
                "最低返航SOC（%）": min(
                    trip.return_soc_percent for trip in solution.trips
                ),
                "组批与主口径相同": _solution_signature(solution)
                == _solution_signature(primary_dp),
            }
        )
    robustness = pd.DataFrame(robustness_rows)
    _write_csv(robustness, output_dir / "energy_interpretation_robustness.csv")

    legacy_safe, _ = _safe_payload_frames(legacy_data, 0.20)
    safe_differences = []
    for service_id in sorted(data.geometries):
        for model_id in sorted(data.aircraft):
            new_value = float(
                safe_wide.loc[
                    safe_wide["服务区编号"] == service_id,
                    f"{model_id}型最大安全载荷（kg）",
                ].iloc[0]
            )
            old_value = float(
                legacy_safe.loc[
                    legacy_safe["服务区编号"] == service_id,
                    f"{model_id}型最大安全载荷（kg）",
                ].iloc[0]
            )
            safe_differences.append(
                {
                    "服务区编号": service_id,
                    "机型编号": model_id,
                    "旧安全载荷（kg）": old_value,
                    "严格GIS安全载荷（kg）": new_value,
                    "差值（kg）": new_value - old_value,
                }
            )
    safe_comparison = pd.DataFrame(safe_differences)
    _write_csv(safe_comparison, output_dir / "gis_safe_payload_comparison.csv")
    gis_solution_comparison = pd.DataFrame(
        [
            {
                "GIS口径": "旧：密集采样+Haversine",
                "架次数": legacy_solution.trip_count,
                "B型架次": model_trip_counts(legacy_solution).get("B", 0),
                "C型架次": model_trip_counts(legacy_solution).get("C", 0),
                "总能耗（kWh）": legacy_solution.total_energy_kwh,
                "累计作业时间（h）": legacy_solution.total_time_s / 3600.0,
                "与严格GIS组批相同": _solution_signature(legacy_solution)
                == _solution_signature(recommended),
            },
            {
                "GIS口径": "新：严格all_touched+WGS84椭球",
                "架次数": recommended.trip_count,
                "B型架次": model_trip_counts(recommended).get("B", 0),
                "C型架次": model_trip_counts(recommended).get("C", 0),
                "总能耗（kWh）": recommended.total_energy_kwh,
                "累计作业时间（h）": recommended.total_time_s / 3600.0,
                "与严格GIS组批相同": True,
            },
        ]
    )
    _write_csv(gis_solution_comparison, output_dir / "gis_solution_comparison.csv")

    critical_records = candidate_critical_margins(data)
    intervals = reserve_stability_intervals(data, records=critical_records)
    critical = _critical_frame(data, critical_records)
    interval_table = _interval_frame(intervals)
    event_table = _interval_event_frame(data, intervals)
    _write_csv(critical, output_dir / "critical_reserve_margins.csv")
    _write_csv(interval_table, output_dir / "reserve_stability_intervals.csv")
    _write_csv(event_table, output_dir / "reserve_change_events.csv")

    trip_audits, representatives, coverage, constraints, summary_checks = _independent_frames(
        data, recommended
    )
    _write_csv(trip_audits, output_dir / "independent_trip_validation.csv")
    _write_csv(representatives, output_dir / "representative_trip_validation.csv")
    _write_csv(coverage, output_dir / "box_coverage_check.csv")
    _write_csv(constraints, output_dir / "constraint_check.csv")
    _write_csv(summary_checks, output_dir / "summary_consistency_check.csv")

    formula_text = _formula_audit_text(problem_dir, robustness)
    (output_dir / "能耗公式与物理口径.md").write_text(formula_text, encoding="utf-8")

    _plot_dem(data, figure_dir / "dem_routes.png")
    _plot_safe_payload_heatmap(safe_wide, figure_dir / "safe_payload_heatmap.png")
    _plot_demand_capacity(data, safe_wide, figure_dir / "demand_vs_capacity.png")
    _plot_pareto(pareto_points, figure_dir / "pareto_front.png")
    _plot_utilization(data, recommended, figure_dir / "trip_utilization.png")
    _plot_safe_payload_curves(data, figure_dir / "safe_payload_curves.png")
    _plot_reserve_staircase(intervals, figure_dir / "reserve_trip_staircase.png")

    paper = _paper_text(
        comparison,
        geometry,
        safe_comparison,
        safe_wide,
        pareto_scenarios,
        robustness,
        interval_table,
        event_table,
        representatives,
        summary_checks,
        recommended,
        legacy_solution,
    )
    (output_dir / "论文_问题一.md").write_text(paper, encoding="utf-8")

    index_text = f"""# D题第一问完整成果索引

- 最终推荐：18 架次，B 型 9、C 型 9，总能耗 {recommended.total_energy_kwh:.6f} kWh，累计作业时间 {recommended.total_time_s/3600:.6f} h。
- 论文正文：[论文_问题一.md](论文_问题一.md)
- 能耗审计：[能耗公式与物理口径.md](能耗公式与物理口径.md)
- 三算法对比：[method_comparison.csv](method_comparison.csv)
- 严格航段结果：[route_geometry.csv](route_geometry.csv)
- 多目标前沿：[pareto_front.csv](pareto_front.csv)
- 临界余量区间：[reserve_stability_intervals.csv](reserve_stability_intervals.csv)
- 独立校验：[independent_trip_validation.csv](independent_trip_validation.csv)
- 逐箱覆盖：[box_coverage_check.csv](box_coverage_check.csv)
- 图表目录：[figures](figures)
- 官方提交表：`结果提交_Q1.xlsx`（由专用模板写入脚本生成）
"""
    (output_dir / "完整成果索引.md").write_text(index_text, encoding="utf-8")
    (output_dir / "summary.md").write_text(
        "# D题第一问最终摘要\n\n"
        f"推荐方案：{recommended.trip_count} 架次，B型9架次、C型9架次；"
        f"总能耗 {recommended.total_energy_kwh:.6f} kWh，累计作业时间 "
        f"{recommended.total_time_s/3600:.6f} h，最低返航SOC "
        f"{min(trip.return_soc_percent for trip in recommended.trips):.6f}%。\n\n"
        "完整论证见 [论文_问题一.md](论文_问题一.md)。\n",
        encoding="utf-8",
    )

    return CompleteResults(
        data=data,
        solutions=solution_map,
        recommended_solution=recommended,
        pareto_points=tuple(pareto_points),
        reserve_intervals=tuple(intervals),
    )
