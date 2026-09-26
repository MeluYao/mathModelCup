"""Domain data and workbook loading for Problem D, question 2."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Tuple

import pandas as pd


@dataclass(frozen=True)
class Node:
    node_id: str
    name: str
    longitude: float
    latitude: float
    elevation_m: float
    is_center: bool = False

    @property
    def operation_altitude_m(self) -> float:
        return self.elevation_m if self.is_center else self.elevation_m + 30.0


@dataclass(frozen=True)
class Q2Box:
    box_id: str
    service_id: str
    material_type: str
    mass_kg: float
    volume_m3: float
    first_batch: bool
    first_deadline_s: float | None
    expected_time_s: float
    priority_weight: float

    @property
    def hard_deadline_s(self) -> float | None:
        deadlines = []
        if self.first_batch and self.first_deadline_s is not None:
            deadlines.append(self.first_deadline_s)
        if self.material_type == "医疗物资":
            deadlines.append(self.expected_time_s)
        return min(deadlines) if deadlines else None


@dataclass(frozen=True)
class AircraftModel:
    model_id: str
    name: str
    empty_mass_kg: float
    max_payload_kg: float
    volume_capacity_m3: float
    cruise_speed_mps: float
    empty_range_m: float
    full_range_m: float
    usable_energy_kwh: float
    reserve_ratio: float
    preparation_time_s: float
    load_time_per_box_s: float
    handoff_base_time_s: float
    handoff_time_per_box_s: float
    climb_speed_mps: float
    descent_speed_mps: float
    climb_efficiency: float


@dataclass(frozen=True)
class AircraftUnit:
    aircraft_id: str
    model_id: str
    initial_node_id: str


@dataclass(frozen=True)
class BatteryUnit:
    battery_id: str
    model_id: str
    full_charge_time_s: float


@dataclass(frozen=True)
class TripStop:
    service_id: str
    box_ids: Tuple[str, ...]


@dataclass(frozen=True)
class TripDraft:
    model_id: str
    stops: Tuple[TripStop, ...]


@dataclass(frozen=True)
class TripPlan:
    model_id: str
    stops: Tuple[TripStop, ...]
    box_ids: Tuple[str, ...]
    total_mass_kg: float
    total_volume_m3: float
    duration_s: float
    energy_kwh: float
    return_soc_percent: float
    leg_payloads_kg: Tuple[float, ...]
    delivery_offsets_s: Mapping[str, float]

    @property
    def signature(self) -> Tuple[object, ...]:
        return (
            self.model_id,
            tuple((stop.service_id, tuple(sorted(stop.box_ids))) for stop in self.stops),
        )


@dataclass(frozen=True)
class TripExecution:
    trip_id: str
    plan: TripPlan
    aircraft_id: str
    battery_id: str
    start_time_s: float
    return_time_s: float
    battery_ready_time_s: float


@dataclass(frozen=True)
class DeliveryRecord:
    box_id: str
    trip_id: str
    service_id: str
    delivery_time_s: float


@dataclass(frozen=True)
class Q2Solution:
    method: str
    trips: Tuple[TripExecution, ...]
    deliveries: Tuple[DeliveryRecord, ...]
    objective: Tuple[float, ...]
    runtime_s: float
    solver_status: str
    diagnostics: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class Q2Data:
    problem_dir: Path
    dem_path: Path
    nodes: Mapping[str, Node]
    boxes: Mapping[str, Q2Box]
    aircraft_models: Mapping[str, AircraftModel]
    aircraft_units: Tuple[AircraftUnit, ...]
    batteries: Tuple[BatteryUnit, ...]


class InfeasibleQ2Error(RuntimeError):
    """Raised when a Q2 route or schedule has no feasible realization."""


def _required_file(root: Path, name: str) -> Path:
    matches = list(Path(root).rglob(name))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected exactly one {name!r} below {root}, found {len(matches)}"
        )
    return matches[0]


def _optional_float(value: object) -> float | None:
    return None if pd.isna(value) or value == "" else float(value)


def load_q2_data(problem_dir: Path) -> Q2Data:
    """Load all transport data needed by question 2 without modifying sources."""
    problem_dir = Path(problem_dir)
    node_file = _required_file(problem_dir, "调度中心与服务区.xlsx")
    aircraft_file = _required_file(problem_dir, "运输无人机数据.xlsx")
    demand_file = _required_file(problem_dir, "物资需求与配送时限.xlsx")
    dem_file = _required_file(problem_dir, "镇龙乡及周边30米DEM.tif")

    node_raw = pd.read_excel(node_file, sheet_name="数据", header=None)
    center_row = node_raw.iloc[2]
    nodes = {
        str(center_row.iloc[0]): Node(
            node_id=str(center_row.iloc[0]),
            name=str(center_row.iloc[1]),
            longitude=float(center_row.iloc[2]),
            latitude=float(center_row.iloc[3]),
            elevation_m=float(center_row.iloc[4]),
            is_center=True,
        )
    }
    for _, row in node_raw.iloc[6:21].iterrows():
        node_id = str(row.iloc[0])
        nodes[node_id] = Node(
            node_id=node_id,
            name=str(row.iloc[1]),
            longitude=float(row.iloc[2]),
            latitude=float(row.iloc[3]),
            elevation_m=float(row.iloc[4]),
        )

    demand_frame = pd.read_excel(demand_file, sheet_name="逐箱货箱清单")
    boxes = {}
    for _, row in demand_frame.iterrows():
        box_id = str(row["货箱编号"])
        boxes[box_id] = Q2Box(
            box_id=box_id,
            service_id=str(row["服务区编号"]),
            material_type=str(row["物资类型"]),
            mass_kg=float(row["单箱质量（kg）"]),
            volume_m3=float(row["单箱体积（m³）"]),
            first_batch=str(row["是否首批保障"]) == "是",
            first_deadline_s=_optional_float(row["首批截止时间（s）"]),
            expected_time_s=float(row["期望送达时间（s）"]),
            priority_weight=float(row["应急优先系数"]),
        )

    aircraft_raw = pd.read_excel(aircraft_file, sheet_name="数据", header=None)
    models = {}
    for _, row in aircraft_raw.iloc[2:5].iterrows():
        model_id = str(row.iloc[0])
        models[model_id] = AircraftModel(
            model_id=model_id,
            name=str(row.iloc[1]),
            empty_mass_kg=float(row.iloc[2]),
            max_payload_kg=float(row.iloc[3]),
            volume_capacity_m3=float(row.iloc[4]),
            cruise_speed_mps=float(row.iloc[5]),
            empty_range_m=float(row.iloc[6]),
            full_range_m=float(row.iloc[7]),
            usable_energy_kwh=float(row.iloc[8]),
            reserve_ratio=float(row.iloc[9]) / 100.0,
            preparation_time_s=float(row.iloc[10]),
            load_time_per_box_s=float(row.iloc[11]),
            handoff_base_time_s=float(row.iloc[12]),
            handoff_time_per_box_s=float(row.iloc[13]),
            climb_speed_mps=float(row.iloc[14]),
            descent_speed_mps=float(row.iloc[15]),
            climb_efficiency=float(row.iloc[16]),
        )

    aircraft_units = tuple(
        AircraftUnit(
            aircraft_id=str(row.iloc[0]),
            model_id=str(row.iloc[1]),
            initial_node_id=str(row.iloc[2]),
        )
        for _, row in aircraft_raw.iloc[8:16].iterrows()
    )

    batteries = []
    for _, row in aircraft_raw.iloc[19:22].iterrows():
        model_id = str(row.iloc[0])
        count = int(row.iloc[1])
        full_charge_time_s = float(row.iloc[2])
        batteries.extend(
            BatteryUnit(
                battery_id=f"B{model_id}{index:02d}",
                model_id=model_id,
                full_charge_time_s=full_charge_time_s,
            )
            for index in range(1, count + 1)
        )

    return Q2Data(
        problem_dir=problem_dir.resolve(),
        dem_path=dem_file.resolve(),
        nodes=nodes,
        boxes=boxes,
        aircraft_models=models,
        aircraft_units=aircraft_units,
        batteries=tuple(batteries),
    )
