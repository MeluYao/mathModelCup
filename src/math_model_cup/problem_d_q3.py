"""Problem D, question 3 domain data and workbook loading.

This program and code were completed with assistance from OpenAI Codex.
The authors reviewed and validated the resulting implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import pandas as pd

from .problem_d_q2 import Q2Data, Q2Solution, load_q2_data


@dataclass(frozen=True)
class Position3D:
    longitude: float
    latitude: float
    altitude_m: float


@dataclass(frozen=True)
class TrajectorySegment:
    transport_trip_id: str
    phase: str
    start_offset_s: float
    end_offset_s: float
    start_position: Position3D
    end_position: Position3D
    trip_start_time_s: float = 0.0

    @property
    def duration_s(self) -> float:
        return self.end_offset_s - self.start_offset_s

    def position_at(self, offset_s: float) -> Position3D:
        if self.duration_s <= 0.0:
            return self.end_position
        fraction = min(1.0, max(0.0, (offset_s - self.start_offset_s) / self.duration_s))
        return Position3D(
            longitude=self.start_position.longitude
            + fraction * (self.end_position.longitude - self.start_position.longitude),
            latitude=self.start_position.latitude
            + fraction * (self.end_position.latitude - self.start_position.latitude),
            altitude_m=self.start_position.altitude_m
            + fraction * (self.end_position.altitude_m - self.start_position.altitude_m),
        )


@dataclass(frozen=True)
class CommunicationSegment:
    segment_id: str
    transport_trip_id: str
    phase: str
    start_time_s: float
    end_time_s: float
    start_position: Position3D
    end_position: Position3D
    direct_available: bool
    minimum_direct_margin_db: float

    @property
    def duration_s(self) -> float:
        return self.end_time_s - self.start_time_s

    def position_at(self, time_s: float) -> Position3D:
        if self.duration_s <= 0.0:
            return self.end_position
        fraction = min(1.0, max(0.0, (time_s - self.start_time_s) / self.duration_s))
        return Position3D(
            longitude=self.start_position.longitude
            + fraction * (self.end_position.longitude - self.start_position.longitude),
            latitude=self.start_position.latitude
            + fraction * (self.end_position.latitude - self.start_position.latitude),
            altitude_m=self.start_position.altitude_m
            + fraction * (self.end_position.altitude_m - self.start_position.altitude_m),
        )


@dataclass(frozen=True)
class RelayState:
    state_id: str
    longitude: float
    latitude: float
    ground_elevation_m: float
    agl_m: float
    altitude_m: float
    backhaul_available: bool
    round_trip_time_s: float
    round_trip_energy_kwh: float
    backhaul_margin_db: float

    @property
    def position(self) -> Position3D:
        return Position3D(self.longitude, self.latitude, self.altitude_m)


@dataclass(frozen=True)
class RelaySortieExecution:
    relay_sortie_id: str
    state_id: str
    relay_id: str
    energy_id: str
    preparation_start_time_s: float
    takeoff_time_s: float
    link_ready_time_s: float
    service_end_time_s: float
    return_time_s: float
    energy_ready_time_s: float
    longitude: float
    latitude: float
    altitude_m: float
    energy_kwh: float
    return_soc_percent: float


@dataclass(frozen=True)
class CommunicationAssignment:
    segment_id: str
    transport_trip_id: str
    mode: str
    relay_sortie_id: str | None
    start_time_s: float
    end_time_s: float


@dataclass(frozen=True)
class RelayModel:
    model_id: str
    takeoff_mass_kg: float
    cruise_speed_mps: float
    cruise_power_kw: float
    usable_energy_kwh: float
    reserve_ratio: float
    preparation_time_s: float
    link_setup_time_s: float
    turnaround_time_s: float
    climb_speed_mps: float
    descent_speed_mps: float
    climb_efficiency: float
    hover_power_kw: float
    communication_power_kw: float
    max_hover_agl_m: float
    full_charge_time_s: float


@dataclass(frozen=True)
class RelayUnit:
    relay_id: str
    model_id: str
    initial_node_id: str


@dataclass(frozen=True)
class RelayEnergyUnit:
    energy_id: str
    model_id: str
    full_charge_time_s: float


@dataclass(frozen=True)
class LinkBudget:
    frequency_mhz: float
    system_loss_db: float
    obstruction_loss_db: float
    sensitivity_dbm: float
    fade_margin_db: float
    transport_tx_dbm: float
    transport_gain_dbi: float
    relay_access_tx_dbm: float
    relay_access_gain_dbi: float
    relay_backhaul_tx_dbm: float
    relay_backhaul_gain_dbi: float
    gateway_tx_dbm: float
    gateway_gain_dbi: float
    gateway_height_agl_m: float


@dataclass(frozen=True)
class Q3Data:
    transport: Q2Data
    relay_model: RelayModel
    relay_units: tuple[RelayUnit, ...]
    energy_units: tuple[RelayEnergyUnit, ...]
    link_budget: LinkBudget


@dataclass(frozen=True)
class Q3Objective:
    weighted_delivery_time: float
    joint_makespan_s: float
    total_energy_kwh: float
    transport_trip_count: int
    relay_trip_count: int

    def as_tuple(self) -> tuple[float, float, float, int, int]:
        return (
            self.weighted_delivery_time,
            self.joint_makespan_s,
            self.total_energy_kwh,
            self.transport_trip_count,
            self.relay_trip_count,
        )


@dataclass(frozen=True)
class Q3Solution:
    method: str
    transport: Q2Solution
    relay_sorties: tuple[RelaySortieExecution, ...]
    communication_assignments: tuple[CommunicationAssignment, ...]
    objective: Q3Objective
    runtime_s: float
    solver_status: str
    diagnostics: Mapping[str, object] = field(default_factory=dict)


def _required_file(root: Path, name: str) -> Path:
    matches = list(Path(root).rglob(name))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected exactly one {name!r} below {root}, found {len(matches)}"
        )
    return matches[0]


def _parameter_map(frame: pd.DataFrame) -> dict[tuple[str, str], float]:
    result: dict[tuple[str, str], float] = {}
    for _, row in frame.iterrows():
        category = row.iloc[0]
        name = row.iloc[1]
        value = row.iloc[4]
        if pd.notna(category) and pd.notna(name) and pd.notna(value):
            result[(str(category), str(name))] = float(value)
    return result


def _row_mapping(
    frame: pd.DataFrame,
    first_header: str,
    *,
    occurrence: int = 0,
) -> dict[str, object]:
    header_matches = frame.index[frame.iloc[:, 0].astype(str) == first_header].tolist()
    if occurrence >= len(header_matches):
        raise ValueError(
            f"missing header occurrence {occurrence} beginning with {first_header!r}"
        )
    header_index = int(header_matches[occurrence])
    headers = frame.loc[header_index].tolist()
    values = frame.loc[header_index + 1].tolist()
    return {
        str(header): value
        for header, value in zip(headers, values)
        if pd.notna(header)
    }


def load_q3_data(problem_dir: Path) -> Q3Data:
    """Load question 3 transport, relay, and communication data read-only."""
    problem_dir = Path(problem_dir)
    transport = load_q2_data(problem_dir)

    relay_path = _required_file(problem_dir, "中继无人机数据.xlsx")
    relay_raw = pd.read_excel(relay_path, sheet_name="数据", header=None)
    model_row = _row_mapping(relay_raw, "机型编号", occurrence=0)
    energy_row = _row_mapping(relay_raw, "机型编号", occurrence=1)
    relay_model = RelayModel(
        model_id=str(model_row["机型编号"]),
        takeoff_mass_kg=float(model_row["计划起飞总质量（kg）"]),
        cruise_speed_mps=float(model_row["计划巡航速度（m/s）"]),
        cruise_power_kw=float(model_row["巡航功率（kW）"]),
        usable_energy_kwh=float(model_row["能源组件可用能量（kWh）"]),
        reserve_ratio=float(model_row["返航电量下限（%）"]) / 100.0,
        preparation_time_s=float(model_row["工位固定准备时间（s）"]),
        link_setup_time_s=float(model_row["建链时间（s）"]),
        turnaround_time_s=float(model_row["架次周转时间（s）"]),
        climb_speed_mps=float(model_row["最大爬升速度（m/s）"]),
        descent_speed_mps=float(model_row["最大下降速度（m/s）"]),
        climb_efficiency=float(model_row["爬升能耗效率"]),
        hover_power_kw=float(model_row["悬停功率（kW）"]),
        communication_power_kw=float(model_row["通信附加功率（kW）"]),
        max_hover_agl_m=float(model_row["最大悬停离地高度（m）"]),
        full_charge_time_s=float(energy_row["等效完全充电时间（s）"]),
    )
    unit_header_index = int(
        relay_raw.index[relay_raw.iloc[:, 0].astype(str) == "中继无人机编号"][0]
    )
    unit_headers = relay_raw.loc[unit_header_index].tolist()
    relay_units_list = []
    for _, row in relay_raw.loc[unit_header_index + 1 :].iterrows():
        if pd.isna(row.iloc[0]):
            break
        record = {
            str(header): value
            for header, value in zip(unit_headers, row.tolist())
            if pd.notna(header)
        }
        relay_units_list.append(
            RelayUnit(
                relay_id=str(record["中继无人机编号"]),
                model_id=str(record["机型编号"]),
                initial_node_id=str(record["初始位置"]),
            )
        )
    relay_units = tuple(relay_units_list)
    energy_count = int(energy_row["共享能源组件总数（组）"])
    energy_units = tuple(
        RelayEnergyUnit(
            energy_id=f"ER{index:02d}",
            model_id=str(energy_row["机型编号"]),
            full_charge_time_s=float(energy_row["等效完全充电时间（s）"]),
        )
        for index in range(1, energy_count + 1)
    )

    link_path = _required_file(problem_dir, "通信链路参数.xlsx")
    link_raw = pd.read_excel(link_path, sheet_name="数据", header=None)
    parameters = _parameter_map(link_raw.iloc[2:])

    def value(category: str, name: str) -> float:
        try:
            return parameters[(category, name)]
        except KeyError as error:
            raise ValueError(f"missing communication parameter: {category}/{name}") from error

    link_budget = LinkBudget(
        frequency_mhz=value("传播参数", "载波频率（MHz）"),
        system_loss_db=value("传播参数", "系统损耗（dB）"),
        obstruction_loss_db=value("传播参数", "地形遮挡附加损耗（dB）"),
        sensitivity_dbm=value("接收参数", "接收灵敏度（dBm）"),
        fade_margin_db=value("接收参数", "衰落裕量（dB）"),
        transport_tx_dbm=value("运输无人机", "发射功率（dBm）"),
        transport_gain_dbi=value("运输无人机", "天线增益（dBi）"),
        relay_access_tx_dbm=value("中继接入端", "发射功率（dBm）"),
        relay_access_gain_dbi=value("中继接入端", "天线增益（dBi）"),
        relay_backhaul_tx_dbm=value("中继回传端", "发射功率（dBm）"),
        relay_backhaul_gain_dbi=value("中继回传端", "天线增益（dBi）"),
        gateway_tx_dbm=value("固定网关 G01", "发射功率（dBm）"),
        gateway_gain_dbi=value("固定网关 G01", "天线增益（dBi）"),
        gateway_height_agl_m=value("固定网关 G01", "天线离地高度（m）"),
    )
    return Q3Data(
        transport=transport,
        relay_model=relay_model,
        relay_units=relay_units,
        energy_units=energy_units,
        link_budget=link_budget,
    )
