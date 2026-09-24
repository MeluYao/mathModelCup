"""Exact and heuristic solvers for Problem D, question 1.

The statement specifies total segment energy as horizontal plus climb energy but
does not print the two component formulae.  This module uses the dimensionally
consistent interpretation documented in the accompanying design:

* horizontal energy = usable battery energy * distance / equivalent range;
* climb energy = mass * gravity * climb height / climb efficiency.

The return leg is empty after all boxes have been delivered.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from math import asin, cos, floor, inf, radians, sin, sqrt
from pathlib import Path
from time import perf_counter
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from pyproj import Geod
from scipy.optimize import Bounds, LinearConstraint, milp


GRAVITY_MPS2 = 9.81
JOULES_PER_KWH = 3_600_000.0
NUMERIC_TOLERANCE = 1e-9


@dataclass(frozen=True)
class Aircraft:
    model_id: str
    name: str
    empty_mass_kg: float
    max_payload_kg: float
    volume_capacity_m3: float
    cruise_speed_mps: float
    empty_range_m: float
    full_range_m: float
    usable_energy_kwh: float
    default_reserve_ratio: float
    preparation_time_s: float
    load_time_per_box_s: float
    handoff_base_time_s: float
    handoff_time_per_box_s: float
    climb_speed_mps: float
    descent_speed_mps: float
    climb_efficiency: float


@dataclass(frozen=True)
class Box:
    box_id: str
    service_id: str
    material_type: str
    mass_kg: float
    volume_m3: float


@dataclass(frozen=True)
class ServiceGeometry:
    service_id: str
    distance_m: float
    peak_ground_m: float
    cruise_altitude_m: float
    center_climb_m: float
    service_climb_m: float
    haversine_distance_m: float = 0.0
    peak_row: int = -1
    peak_column: int = -1
    peak_longitude: float = float("nan")
    peak_latitude: float = float("nan")
    service_longitude: float = float("nan")
    service_latitude: float = float("nan")
    service_ground_m: float = float("nan")

    @property
    def outbound_descent_m(self) -> float:
        return self.service_climb_m

    @property
    def return_climb_m(self) -> float:
        return self.service_climb_m

    @property
    def return_descent_m(self) -> float:
        return self.center_climb_m


@dataclass(frozen=True)
class GeoTiffMetadata:
    path: Path
    columns: int
    rows: int
    epsg: int
    crs_name: str
    pixel_width_degrees: float
    pixel_height_degrees: float
    nodata: float | None
    affine: Tuple[float, float, float, float, float, float]
    bounds: Tuple[float, float, float, float]


@dataclass(frozen=True)
class ProblemData:
    aircraft: Mapping[str, Aircraft]
    boxes: Tuple[Box, ...]
    geometries: Mapping[str, ServiceGeometry]
    material_types: Tuple[str, ...]
    dem_metadata: GeoTiffMetadata | None = None
    center_longitude: float = float("nan")
    center_latitude: float = float("nan")
    center_elevation_m: float = float("nan")


@dataclass(frozen=True)
class CandidateTrip:
    service_id: str
    model_id: str
    counts: Tuple[int, ...]
    mass_kg: float
    volume_m3: float
    time_s: float
    energy_kwh: float
    return_soc_percent: float

    @property
    def box_count(self) -> int:
        return sum(self.counts)


@dataclass(frozen=True)
class EnergyBreakdown:
    outbound_payload_kg: float
    return_payload_kg: float
    outbound_equivalent_range_m: float
    return_equivalent_range_m: float
    outbound_horizontal_kwh: float
    return_horizontal_kwh: float
    outbound_climb_kwh: float
    return_climb_kwh: float
    descent_energy_kwh: float

    @property
    def horizontal_kwh(self) -> float:
        return self.outbound_horizontal_kwh + self.return_horizontal_kwh

    @property
    def climb_kwh(self) -> float:
        return self.outbound_climb_kwh + self.return_climb_kwh

    @property
    def total_kwh(self) -> float:
        return self.horizontal_kwh + self.climb_kwh + self.descent_energy_kwh


@dataclass(frozen=True)
class TripResult:
    trip_id: str
    service_id: str
    model_id: str
    box_ids: Tuple[str, ...]
    counts: Tuple[int, ...]
    mass_kg: float
    volume_m3: float
    time_s: float
    energy_kwh: float
    return_soc_percent: float


@dataclass(frozen=True)
class MethodSolution:
    method: str
    trips: Tuple[TripResult, ...]
    runtime_s: float

    @property
    def trip_count(self) -> int:
        return len(self.trips)

    @property
    def total_time_s(self) -> float:
        return sum(trip.time_s for trip in self.trips)

    @property
    def total_energy_kwh(self) -> float:
        return sum(trip.energy_kwh for trip in self.trips)

    @property
    def objective(self) -> Tuple[int, float, float]:
        return (self.trip_count, self.total_time_s, self.total_energy_kwh)


class InfeasibleProblemError(RuntimeError):
    """Raised when at least one service cannot be covered."""


def equivalent_range(aircraft: Aircraft, payload_kg: float) -> float:
    """Return load-dependent standard range in metres."""
    if payload_kg < -NUMERIC_TOLERANCE or payload_kg > aircraft.max_payload_kg + NUMERIC_TOLERANCE:
        raise ValueError("payload is outside aircraft limits")
    payload = min(max(payload_kg, 0.0), aircraft.max_payload_kg)
    load_fraction = payload / aircraft.max_payload_kg
    return aircraft.empty_range_m - (
        aircraft.empty_range_m - aircraft.full_range_m
    ) * load_fraction**1.5


def _horizontal_energy(
    aircraft: Aircraft,
    distance_m: float,
    payload_kg: float,
    range_energy_fraction: float = 1.0,
) -> float:
    return (
        range_energy_fraction
        * aircraft.usable_energy_kwh
        * distance_m
        / equivalent_range(aircraft, payload_kg)
    )


def _climb_energy(aircraft: Aircraft, climb_m: float, payload_kg: float) -> float:
    return (
        (aircraft.empty_mass_kg + payload_kg)
        * GRAVITY_MPS2
        * climb_m
        / (JOULES_PER_KWH * aircraft.climb_efficiency)
    )


def trip_energy_breakdown(
    aircraft: Aircraft,
    geometry: ServiceGeometry,
    payload_kg: float,
    *,
    range_energy_fraction: float = 1.0,
) -> EnergyBreakdown:
    """Return a dimensionally explicit round-trip energy decomposition.

    ``range_energy_fraction=1`` means the tabulated standard range consumes the
    full usable battery energy.  A value of 0.8 represents the alternative
    interpretation that the published range already preserves a 20% reserve.
    """
    if not 0.0 < range_energy_fraction <= 1.0:
        raise ValueError("range_energy_fraction must be in (0, 1]")
    return EnergyBreakdown(
        outbound_payload_kg=payload_kg,
        return_payload_kg=0.0,
        outbound_equivalent_range_m=equivalent_range(aircraft, payload_kg),
        return_equivalent_range_m=equivalent_range(aircraft, 0.0),
        outbound_horizontal_kwh=_horizontal_energy(
            aircraft, geometry.distance_m, payload_kg, range_energy_fraction
        ),
        return_horizontal_kwh=_horizontal_energy(
            aircraft, geometry.distance_m, 0.0, range_energy_fraction
        ),
        outbound_climb_kwh=_climb_energy(
            aircraft, geometry.center_climb_m, payload_kg
        ),
        return_climb_kwh=_climb_energy(
            aircraft, geometry.return_climb_m, 0.0
        ),
        descent_energy_kwh=0.0,
    )


def trip_energy(
    aircraft: Aircraft,
    geometry: ServiceGeometry,
    payload_kg: float,
    *,
    range_energy_fraction: float = 1.0,
) -> float:
    """Return round-trip energy with loaded outbound and empty return legs."""
    return trip_energy_breakdown(
        aircraft,
        geometry,
        payload_kg,
        range_energy_fraction=range_energy_fraction,
    ).total_kwh


def trip_time(aircraft: Aircraft, geometry: ServiceGeometry, box_count: int) -> float:
    """Return cumulative operating time for one direct round trip."""
    outbound_flight = (
        geometry.center_climb_m / aircraft.climb_speed_mps
        + geometry.distance_m / aircraft.cruise_speed_mps
        + geometry.service_climb_m / aircraft.descent_speed_mps
    )
    inbound_flight = (
        geometry.service_climb_m / aircraft.climb_speed_mps
        + geometry.distance_m / aircraft.cruise_speed_mps
        + geometry.center_climb_m / aircraft.descent_speed_mps
    )
    return (
        aircraft.preparation_time_s
        + box_count * aircraft.load_time_per_box_s
        + outbound_flight
        + aircraft.handoff_base_time_s
        + box_count * aircraft.handoff_time_per_box_s
        + inbound_flight
    )


def safe_payload(
    aircraft: Aircraft,
    geometry: ServiceGeometry,
    reserve_ratio: float,
    *,
    iterations: int = 80,
    range_energy_fraction: float = 1.0,
) -> float:
    """Return the largest continuous payload satisfying the reserve constraint."""
    if not 0.0 <= reserve_ratio < 1.0:
        raise ValueError("reserve_ratio must be in [0, 1)")
    energy_limit = (1.0 - reserve_ratio) * aircraft.usable_energy_kwh
    if trip_energy(
        aircraft, geometry, 0.0, range_energy_fraction=range_energy_fraction
    ) > energy_limit + NUMERIC_TOLERANCE:
        return 0.0
    if trip_energy(
        aircraft,
        geometry,
        aircraft.max_payload_kg,
        range_energy_fraction=range_energy_fraction,
    ) <= energy_limit + NUMERIC_TOLERANCE:
        return aircraft.max_payload_kg
    lower = 0.0
    upper = aircraft.max_payload_kg
    for _ in range(iterations):
        middle = (lower + upper) / 2.0
        if trip_energy(
            aircraft, geometry, middle, range_energy_fraction=range_energy_fraction
        ) <= energy_limit:
            lower = middle
        else:
            upper = middle
    return lower


def _great_circle_distance_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    earth_radius_m = 6_371_008.8
    phi1 = radians(lat1)
    phi2 = radians(lat2)
    delta_phi = phi2 - phi1
    delta_lambda = radians(lon2 - lon1)
    haversine = (
        sin(delta_phi / 2.0) ** 2
        + cos(phi1) * cos(phi2) * sin(delta_lambda / 2.0) ** 2
    )
    return 2.0 * earth_radius_m * asin(sqrt(haversine))


_WGS84_GEOD = Geod(ellps="WGS84")


def _ellipsoidal_distance_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Return the WGS84 inverse-geodesic distance in metres."""
    _, _, distance_m = _WGS84_GEOD.inv(lon1, lat1, lon2, lat2)
    return float(distance_m)


def _segment_intersects_closed_rectangle(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> bool:
    """Liang-Barsky intersection including boundary and corner touches."""
    dx = x2 - x1
    dy = y2 - y1
    lower = 0.0
    upper = 1.0
    tolerance = 1e-12
    for direction, offset in (
        (-dx, x1 - x_min),
        (dx, x_max - x1),
        (-dy, y1 - y_min),
        (dy, y_max - y1),
    ):
        if abs(direction) <= tolerance:
            if offset < -tolerance:
                return False
            continue
        ratio = offset / direction
        if direction < 0.0:
            lower = max(lower, ratio)
        else:
            upper = min(upper, ratio)
        if lower > upper + tolerance:
            return False
    return True


def _line_cells(
    lon1: float,
    lat1: float,
    lon2: float,
    lat2: float,
    *,
    x_origin: float,
    y_origin: float,
    x_scale: float,
    y_scale: float,
    rows: int,
    columns: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return every DEM cell touched by the closed route segment."""
    column1 = (lon1 - x_origin) / x_scale
    row1 = (y_origin - lat1) / y_scale
    column2 = (lon2 - x_origin) / x_scale
    row2 = (y_origin - lat2) / y_scale
    row_start = max(0, floor(min(row1, row2)) - 1)
    row_stop = min(rows - 1, floor(max(row1, row2)) + 1)
    column_start = max(0, floor(min(column1, column2)) - 1)
    column_stop = min(columns - 1, floor(max(column1, column2)) + 1)
    pairs = [
        (row, column)
        for row in range(row_start, row_stop + 1)
        for column in range(column_start, column_stop + 1)
        if _segment_intersects_closed_rectangle(
            column1,
            row1,
            column2,
            row2,
            float(column),
            float(column + 1),
            float(row),
            float(row + 1),
        )
    ]
    if not pairs:
        raise ValueError("route lies outside DEM coverage")
    pair_array = np.asarray(sorted(set(pairs)), dtype=int)
    return pair_array[:, 0], pair_array[:, 1]


def _required_file(root: Path, name: str) -> Path:
    matches = [path for path in root.rglob(name) if not path.name.startswith("~$")]
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one {name!r} below {root}, found {len(matches)}")
    return matches[0]


def _geotiff_epsg(geo_keys: Sequence[int]) -> int:
    """Extract the geographic or projected EPSG code from a GeoKey directory."""
    if len(geo_keys) < 4:
        raise ValueError("invalid GeoTIFF GeoKey directory")
    key_count = int(geo_keys[3])
    for index in range(key_count):
        offset = 4 + index * 4
        key_id, tag_location, count, value_offset = geo_keys[offset : offset + 4]
        if key_id in (2048, 3072) and tag_location == 0 and count == 1:
            return int(value_offset)
    raise ValueError("GeoTIFF does not declare a geographic/projected EPSG code")


def _parse_nodata(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (tuple, list)):
        value = value[0]
    text = value.decode("ascii") if isinstance(value, bytes) else str(value)
    text = text.strip().strip("\x00")
    if not text:
        return None
    return float(text)


def load_problem_data(problem_dir: Path) -> ProblemData:
    """Load the supplied D-problem workbooks and GeoTIFF without modifying them."""
    problem_dir = Path(problem_dir)
    node_file = _required_file(problem_dir, "调度中心与服务区.xlsx")
    aircraft_file = _required_file(problem_dir, "运输无人机数据.xlsx")
    demand_file = _required_file(problem_dir, "物资需求与配送时限.xlsx")
    dem_file = _required_file(problem_dir, "镇龙乡及周边30米DEM.tif")

    aircraft_raw = pd.read_excel(aircraft_file, sheet_name="数据", header=None)
    aircraft_by_id: Dict[str, Aircraft] = {}
    for _, row in aircraft_raw.iloc[2:5].iterrows():
        model_id = str(row.iloc[0])
        aircraft_by_id[model_id] = Aircraft(
            model_id=model_id,
            name=str(row.iloc[1]),
            empty_mass_kg=float(row.iloc[2]),
            max_payload_kg=float(row.iloc[3]),
            volume_capacity_m3=float(row.iloc[4]),
            cruise_speed_mps=float(row.iloc[5]),
            empty_range_m=float(row.iloc[6]),
            full_range_m=float(row.iloc[7]),
            usable_energy_kwh=float(row.iloc[8]),
            default_reserve_ratio=float(row.iloc[9]) / 100.0,
            preparation_time_s=float(row.iloc[10]),
            load_time_per_box_s=float(row.iloc[11]),
            handoff_base_time_s=float(row.iloc[12]),
            handoff_time_per_box_s=float(row.iloc[13]),
            climb_speed_mps=float(row.iloc[14]),
            descent_speed_mps=float(row.iloc[15]),
            climb_efficiency=float(row.iloc[16]),
        )

    box_frame = pd.read_excel(demand_file, sheet_name="逐箱货箱清单")
    boxes = tuple(
        Box(
            box_id=str(row["货箱编号"]),
            service_id=str(row["服务区编号"]),
            material_type=str(row["物资类型"]),
            mass_kg=float(row["单箱质量（kg）"]),
            volume_m3=float(row["单箱体积（m³）"]),
        )
        for _, row in box_frame.iterrows()
    )
    material_types = tuple(dict.fromkeys(box.material_type for box in boxes))

    node_raw = pd.read_excel(node_file, sheet_name="数据", header=None)
    center_lon = float(node_raw.iloc[2, 2])
    center_lat = float(node_raw.iloc[2, 3])
    center_elevation = float(node_raw.iloc[2, 4])

    with Image.open(dem_file) as image:
        dem = np.asarray(image, dtype=float)
        tie_point = image.tag_v2[33922]
        pixel_scale = image.tag_v2[33550]
        geo_keys = image.tag_v2[34735]
        nodata = _parse_nodata(image.tag_v2.get(42113))
    x_origin = float(tie_point[3])
    y_origin = float(tie_point[4])
    x_scale = float(pixel_scale[0])
    y_scale = float(pixel_scale[1])
    epsg = _geotiff_epsg(geo_keys)
    if epsg != 4326:
        raise ValueError(f"expected WGS84 geographic DEM (EPSG:4326), got EPSG:{epsg}")
    if x_scale <= 0.0 or y_scale <= 0.0:
        raise ValueError("GeoTIFF pixel scales must be positive")
    bounds = (
        x_origin,
        y_origin - dem.shape[0] * y_scale,
        x_origin + dem.shape[1] * x_scale,
        y_origin,
    )
    dem_metadata = GeoTiffMetadata(
        path=dem_file,
        columns=int(dem.shape[1]),
        rows=int(dem.shape[0]),
        epsg=epsg,
        crs_name="WGS 84",
        pixel_width_degrees=x_scale,
        pixel_height_degrees=y_scale,
        nodata=nodata,
        affine=(x_scale, 0.0, x_origin, 0.0, -y_scale, y_origin),
        bounds=bounds,
    )
    if not (bounds[0] <= center_lon <= bounds[2] and bounds[1] <= center_lat <= bounds[3]):
        raise ValueError("dispatch center lies outside DEM coverage")

    geometries: Dict[str, ServiceGeometry] = {}
    for _, row in node_raw.iloc[6:21].iterrows():
        service_id = str(row.iloc[0])
        longitude = float(row.iloc[2])
        latitude = float(row.iloc[3])
        elevation = float(row.iloc[4])
        if not (bounds[0] <= longitude <= bounds[2] and bounds[1] <= latitude <= bounds[3]):
            raise ValueError(f"service {service_id} lies outside DEM coverage")
        dem_rows, dem_columns = _line_cells(
            center_lon,
            center_lat,
            longitude,
            latitude,
            x_origin=x_origin,
            y_origin=y_origin,
            x_scale=x_scale,
            y_scale=y_scale,
            rows=dem.shape[0],
            columns=dem.shape[1],
        )
        route_elevations = dem[dem_rows, dem_columns]
        valid = np.isfinite(route_elevations)
        if nodata is not None:
            valid &= ~np.isclose(route_elevations, nodata, rtol=0.0, atol=1e-12)
        if not valid.any():
            raise ValueError(f"route {service_id} has no valid DEM pixels")
        valid_elevations = route_elevations[valid]
        valid_rows = dem_rows[valid]
        valid_columns = dem_columns[valid]
        peak_index = int(np.argmax(valid_elevations))
        peak_ground = float(valid_elevations[peak_index])
        peak_row = int(valid_rows[peak_index])
        peak_column = int(valid_columns[peak_index])
        cruise_altitude = peak_ground + 50.0
        haversine_distance = _great_circle_distance_m(
            center_lon, center_lat, longitude, latitude
        )
        geometries[service_id] = ServiceGeometry(
            service_id=service_id,
            distance_m=_ellipsoidal_distance_m(
                center_lon, center_lat, longitude, latitude
            ),
            peak_ground_m=peak_ground,
            cruise_altitude_m=cruise_altitude,
            center_climb_m=max(0.0, cruise_altitude - center_elevation),
            service_climb_m=max(0.0, cruise_altitude - (elevation + 30.0)),
            haversine_distance_m=haversine_distance,
            peak_row=peak_row,
            peak_column=peak_column,
            peak_longitude=x_origin + (peak_column + 0.5) * x_scale,
            peak_latitude=y_origin - (peak_row + 0.5) * y_scale,
            service_longitude=longitude,
            service_latitude=latitude,
            service_ground_m=elevation,
        )

    return ProblemData(
        aircraft=aircraft_by_id,
        boxes=boxes,
        geometries=geometries,
        material_types=material_types,
        dem_metadata=dem_metadata,
        center_longitude=center_lon,
        center_latitude=center_lat,
        center_elevation_m=center_elevation,
    )


def _boxes_for_service(data: ProblemData, service_id: str) -> Tuple[Box, ...]:
    return tuple(box for box in data.boxes if box.service_id == service_id)


def _target_counts(data: ProblemData, service_id: str) -> Tuple[int, ...]:
    boxes = _boxes_for_service(data, service_id)
    return tuple(
        sum(box.material_type == material_type for box in boxes)
        for material_type in data.material_types
    )


def _type_properties(data: ProblemData) -> Dict[str, Tuple[float, float]]:
    properties: Dict[str, Tuple[float, float]] = {}
    for box in data.boxes:
        value = (box.mass_kg, box.volume_m3)
        if box.material_type in properties and properties[box.material_type] != value:
            raise ValueError(f"material type {box.material_type} has inconsistent box properties")
        properties[box.material_type] = value
    return properties


def generate_candidates(
    data: ProblemData,
    service_id: str,
    reserve_ratio: float = 0.20,
    *,
    range_energy_fraction: float = 1.0,
) -> List[CandidateTrip]:
    """Enumerate feasible type-count patterns and remove same-pattern domination."""
    geometry = data.geometries[service_id]
    target = _target_counts(data, service_id)
    properties = _type_properties(data)
    unpruned: List[CandidateTrip] = []
    for counts in product(*(range(count + 1) for count in target)):
        if not any(counts):
            continue
        mass = sum(
            counts[index] * properties[material_type][0]
            for index, material_type in enumerate(data.material_types)
        )
        volume = sum(
            counts[index] * properties[material_type][1]
            for index, material_type in enumerate(data.material_types)
        )
        box_count = sum(counts)
        for model_id in sorted(data.aircraft):
            aircraft = data.aircraft[model_id]
            if mass > aircraft.max_payload_kg + NUMERIC_TOLERANCE:
                continue
            if volume > aircraft.volume_capacity_m3 + NUMERIC_TOLERANCE:
                continue
            energy = trip_energy(
                aircraft,
                geometry,
                mass,
                range_energy_fraction=range_energy_fraction,
            )
            energy_limit = (1.0 - reserve_ratio) * aircraft.usable_energy_kwh
            if energy > energy_limit + NUMERIC_TOLERANCE:
                continue
            unpruned.append(
                CandidateTrip(
                    service_id=service_id,
                    model_id=model_id,
                    counts=tuple(int(value) for value in counts),
                    mass_kg=mass,
                    volume_m3=volume,
                    time_s=trip_time(aircraft, geometry, box_count),
                    energy_kwh=energy,
                    return_soc_percent=100.0 * (1.0 - energy / aircraft.usable_energy_kwh),
                )
            )

    by_counts: Dict[Tuple[int, ...], List[CandidateTrip]] = {}
    for candidate in unpruned:
        by_counts.setdefault(candidate.counts, []).append(candidate)
    candidates: List[CandidateTrip] = []
    for counts in sorted(by_counts):
        group = by_counts[counts]
        for candidate in group:
            dominated = any(
                other is not candidate
                and other.time_s <= candidate.time_s + NUMERIC_TOLERANCE
                and other.energy_kwh <= candidate.energy_kwh + NUMERIC_TOLERANCE
                and (
                    other.time_s < candidate.time_s - NUMERIC_TOLERANCE
                    or other.energy_kwh < candidate.energy_kwh - NUMERIC_TOLERANCE
                )
                for other in group
            )
            if not dominated:
                candidates.append(candidate)
    candidates.sort(
        key=lambda item: (
            item.service_id,
            -item.box_count,
            item.time_s,
            item.energy_kwh,
            item.model_id,
            item.counts,
        )
    )
    return candidates


def materialize_box_ids(
    data: ProblemData,
    selected_candidates: Sequence[CandidateTrip],
    method: str,
) -> Tuple[TripResult, ...]:
    available: Dict[Tuple[str, str], List[str]] = {}
    for box in sorted(data.boxes, key=lambda item: item.box_id):
        available.setdefault((box.service_id, box.material_type), []).append(box.box_id)

    method_code = {"greedy": "GRE", "dynamic_programming": "DP", "milp": "MILP"}.get(
        method, method.upper()[:4]
    )
    trips: List[TripResult] = []
    ordered = sorted(
        selected_candidates,
        key=lambda item: (item.service_id, -item.mass_kg, item.model_id, item.counts),
    )
    for index, candidate in enumerate(ordered, start=1):
        box_ids: List[str] = []
        for type_index, count in enumerate(candidate.counts):
            if count == 0:
                continue
            material_type = data.material_types[type_index]
            key = (candidate.service_id, material_type)
            if len(available.get(key, [])) < count:
                raise ValueError("selected candidates exceed available boxes")
            box_ids.extend(available[key][:count])
            del available[key][:count]
        trips.append(
            TripResult(
                trip_id=f"Q1-{method_code}-{index:03d}",
                service_id=candidate.service_id,
                model_id=candidate.model_id,
                box_ids=tuple(box_ids),
                counts=candidate.counts,
                mass_kg=candidate.mass_kg,
                volume_m3=candidate.volume_m3,
                time_s=candidate.time_s,
                energy_kwh=candidate.energy_kwh,
                return_soc_percent=candidate.return_soc_percent,
            )
        )
    return tuple(trips)


def _solution_from_candidates(
    data: ProblemData,
    method: str,
    candidates: Sequence[CandidateTrip],
    runtime_s: float,
) -> MethodSolution:
    solution = MethodSolution(
        method=method,
        trips=materialize_box_ids(data, candidates, method),
        runtime_s=runtime_s,
    )
    return solution


def _candidate_lookup(
    candidates: Sequence[CandidateTrip],
) -> Dict[Tuple[int, ...], List[CandidateTrip]]:
    lookup: Dict[Tuple[int, ...], List[CandidateTrip]] = {}
    for candidate in candidates:
        lookup.setdefault(candidate.counts, []).append(candidate)
    return lookup


def solve_greedy(
    data: ProblemData,
    reserve_ratio: float = 0.20,
    *,
    range_energy_fraction: float = 1.0,
) -> MethodSolution:
    """Solve by best-fit decreasing followed by deterministic pair merging."""
    started = perf_counter()
    properties = _type_properties(data)
    selected: List[CandidateTrip] = []
    for service_id in sorted(data.geometries):
        candidates = generate_candidates(
            data,
            service_id,
            reserve_ratio,
            range_energy_fraction=range_energy_fraction,
        )
        if not candidates:
            raise InfeasibleProblemError(f"no feasible trip for {service_id}")
        lookup = _candidate_lookup(candidates)
        items: List[int] = []
        target = _target_counts(data, service_id)
        for type_index, count in enumerate(target):
            items.extend([type_index] * count)
        items.sort(
            key=lambda index: (
                properties[data.material_types[index]][0],
                properties[data.material_types[index]][1],
                -index,
            ),
            reverse=True,
        )
        batches: List[CandidateTrip] = []
        for type_index in items:
            placement_options: List[Tuple[Tuple[float, float, float, int], int, CandidateTrip]] = []
            for batch_index, batch in enumerate(batches):
                counts = list(batch.counts)
                counts[type_index] += 1
                for option in lookup.get(tuple(counts), []):
                    aircraft = data.aircraft[option.model_id]
                    payload_capacity = min(
                        aircraft.max_payload_kg,
                        safe_payload(
                            aircraft,
                            data.geometries[service_id],
                            reserve_ratio,
                            range_energy_fraction=range_energy_fraction,
                        ),
                    )
                    residual = (
                        (payload_capacity - option.mass_kg) / max(payload_capacity, 1e-12)
                        + (aircraft.volume_capacity_m3 - option.volume_m3)
                        / aircraft.volume_capacity_m3
                    )
                    placement_options.append(
                        ((residual, option.time_s, option.energy_kwh, batch_index), batch_index, option)
                    )
            if placement_options:
                _, batch_index, option = min(placement_options, key=lambda item: item[0])
                batches[batch_index] = option
            else:
                singleton = [0] * len(data.material_types)
                singleton[type_index] = 1
                options = lookup.get(tuple(singleton), [])
                if not options:
                    raise InfeasibleProblemError(
                        f"box type {data.material_types[type_index]} cannot reach {service_id}"
                    )
                batches.append(min(options, key=lambda item: (item.time_s, item.energy_kwh)))

        improved = True
        while improved:
            improved = False
            merge_options: List[Tuple[Tuple[float, float, int, int], int, int, CandidateTrip]] = []
            for left in range(len(batches)):
                for right in range(left + 1, len(batches)):
                    counts = tuple(
                        batches[left].counts[index] + batches[right].counts[index]
                        for index in range(len(data.material_types))
                    )
                    for option in lookup.get(counts, []):
                        merge_options.append(
                            (
                                (option.time_s, option.energy_kwh, left, right),
                                left,
                                right,
                                option,
                            )
                        )
            if merge_options:
                _, left, right, option = min(merge_options, key=lambda item: item[0])
                batches[left] = option
                del batches[right]
                improved = True

        batches = [
            min(lookup[batch.counts], key=lambda item: (item.time_s, item.energy_kwh))
            for batch in batches
        ]
        selected.extend(batches)

    solution = _solution_from_candidates(
        data, "greedy", selected, perf_counter() - started
    )
    validate_solution(
        data,
        solution.trips,
        reserve_ratio,
        range_energy_fraction=range_energy_fraction,
    )
    return solution


def solve_dynamic_programming(
    data: ProblemData,
    reserve_ratio: float = 0.20,
    *,
    range_energy_fraction: float = 1.0,
) -> MethodSolution:
    """Solve every independent service exactly by acyclic count-state DP."""
    started = perf_counter()
    selected: List[CandidateTrip] = []
    for service_id in sorted(data.geometries):
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
        def best_from(state: Tuple[int, ...]) -> Tuple[Tuple[int, float, float], Tuple[CandidateTrip, ...]]:
            if state == target:
                return (0, 0.0, 0.0), tuple()
            best_objective: Tuple[int, float, float] = (10**9, inf, inf)
            best_path: Tuple[CandidateTrip, ...] = tuple()
            for candidate in candidates:
                next_state = tuple(
                    state[index] + candidate.counts[index]
                    for index in range(len(target))
                )
                if any(next_state[index] > target[index] for index in range(len(target))):
                    continue
                sub_objective, sub_path = best_from(next_state)
                objective = (
                    1 + sub_objective[0],
                    candidate.time_s + sub_objective[1],
                    candidate.energy_kwh + sub_objective[2],
                )
                if objective < best_objective:
                    best_objective = objective
                    best_path = (candidate,) + sub_path
            return best_objective, best_path

        objective, path = best_from(tuple(0 for _ in target))
        if objective[0] >= 10**9:
            raise InfeasibleProblemError(f"cannot cover all boxes for {service_id}")
        selected.extend(path)

    solution = _solution_from_candidates(
        data, "dynamic_programming", selected, perf_counter() - started
    )
    validate_solution(
        data,
        solution.trips,
        reserve_ratio,
        range_energy_fraction=range_energy_fraction,
    )
    return solution


def _run_milp(
    cost: np.ndarray,
    integrality: np.ndarray,
    bounds: Bounds,
    constraints: Sequence[LinearConstraint],
) -> np.ndarray:
    result = milp(
        c=cost,
        integrality=integrality,
        bounds=bounds,
        constraints=list(constraints),
        options={"presolve": True},
    )
    if not result.success or result.x is None:
        raise InfeasibleProblemError(f"MILP failed: {result.message}")
    return result.x


def solve_milp(
    data: ProblemData,
    reserve_ratio: float = 0.20,
    *,
    range_energy_fraction: float = 1.0,
) -> MethodSolution:
    """Solve global set partitioning with three lexicographic MILP stages."""
    started = perf_counter()
    candidates: List[CandidateTrip] = []
    for service_id in sorted(data.geometries):
        service_candidates = generate_candidates(
            data,
            service_id,
            reserve_ratio,
            range_energy_fraction=range_energy_fraction,
        )
        if not service_candidates:
            raise InfeasibleProblemError(f"no feasible trip for {service_id}")
        candidates.extend(service_candidates)

    row_keys = [
        (service_id, type_index)
        for service_id in sorted(data.geometries)
        for type_index in range(len(data.material_types))
    ]
    row_index = {key: index for index, key in enumerate(row_keys)}
    coverage = np.zeros((len(row_keys), len(candidates)), dtype=float)
    target = np.zeros(len(row_keys), dtype=float)
    for service_id in sorted(data.geometries):
        counts = _target_counts(data, service_id)
        for type_index, count in enumerate(counts):
            target[row_index[(service_id, type_index)]] = count
    for column, candidate in enumerate(candidates):
        for type_index, count in enumerate(candidate.counts):
            coverage[row_index[(candidate.service_id, type_index)], column] = count

    count_constraint = LinearConstraint(coverage, target, target)
    variable_count = len(candidates)
    integrality = np.ones(variable_count, dtype=int)
    upper_bound = max(target.max(), 1.0)
    bounds = Bounds(np.zeros(variable_count), np.full(variable_count, upper_bound))
    trip_cost = np.ones(variable_count)

    stage_one = _run_milp(
        trip_cost, integrality, bounds, (count_constraint,)
    )
    minimum_trips = int(round(float(trip_cost @ stage_one)))
    trip_constraint = LinearConstraint(trip_cost, minimum_trips, minimum_trips)

    time_hours = np.array([candidate.time_s / 3600.0 for candidate in candidates])
    stage_two = _run_milp(
        time_hours,
        integrality,
        bounds,
        (count_constraint, trip_constraint),
    )
    minimum_time_hours = float(time_hours @ stage_two)
    time_constraint = LinearConstraint(
        time_hours, -np.inf, minimum_time_hours + 1e-8
    )

    energy = np.array([candidate.energy_kwh for candidate in candidates])
    stage_three = _run_milp(
        energy,
        integrality,
        bounds,
        (count_constraint, trip_constraint, time_constraint),
    )
    integer_solution = np.rint(stage_three).astype(int)
    if not np.array_equal(coverage @ integer_solution, target):
        raise RuntimeError("rounded MILP solution does not satisfy exact coverage")
    selected = [
        candidate
        for candidate, repeat_count in zip(candidates, integer_solution)
        for _ in range(int(repeat_count))
    ]
    solution = _solution_from_candidates(
        data, "milp", selected, perf_counter() - started
    )
    validate_solution(
        data,
        solution.trips,
        reserve_ratio,
        range_energy_fraction=range_energy_fraction,
    )
    return solution


def validate_solution(
    data: ProblemData,
    trips: Sequence[TripResult],
    reserve_ratio: float = 0.20,
    *,
    range_energy_fraction: float = 1.0,
) -> None:
    """Raise ValueError unless the result satisfies all question-1 constraints."""
    boxes_by_id = {box.box_id: box for box in data.boxes}
    seen: List[str] = []
    for trip in trips:
        if not trip.box_ids:
            raise ValueError(f"{trip.trip_id} is empty")
        aircraft = data.aircraft[trip.model_id]
        geometry = data.geometries[trip.service_id]
        trip_boxes = [boxes_by_id[box_id] for box_id in trip.box_ids]
        if any(box.service_id != trip.service_id for box in trip_boxes):
            raise ValueError(f"{trip.trip_id} mixes service areas")
        mass = sum(box.mass_kg for box in trip_boxes)
        volume = sum(box.volume_m3 for box in trip_boxes)
        energy = trip_energy(
            aircraft,
            geometry,
            mass,
            range_energy_fraction=range_energy_fraction,
        )
        duration = trip_time(aircraft, geometry, len(trip_boxes))
        return_soc = 100.0 * (1.0 - energy / aircraft.usable_energy_kwh)
        if abs(mass - trip.mass_kg) > 1e-7:
            raise ValueError(f"{trip.trip_id} mass mismatch")
        if abs(volume - trip.volume_m3) > 1e-9:
            raise ValueError(f"{trip.trip_id} volume mismatch")
        if mass > aircraft.max_payload_kg + NUMERIC_TOLERANCE:
            raise ValueError(f"{trip.trip_id} exceeds mass capacity")
        if volume > aircraft.volume_capacity_m3 + NUMERIC_TOLERANCE:
            raise ValueError(f"{trip.trip_id} exceeds volume capacity")
        if abs(energy - trip.energy_kwh) > 1e-7:
            raise ValueError(f"{trip.trip_id} energy mismatch")
        if abs(duration - trip.time_s) > 1e-7:
            raise ValueError(f"{trip.trip_id} time mismatch")
        if abs(return_soc - trip.return_soc_percent) > 1e-7:
            raise ValueError(f"{trip.trip_id} return SOC mismatch")
        if return_soc < reserve_ratio * 100.0 - 1e-7:
            raise ValueError(f"{trip.trip_id} violates reserve SOC")
        seen.extend(trip.box_ids)

    expected = {box.box_id for box in data.boxes}
    if len(seen) != len(set(seen)):
        raise ValueError("at least one box is assigned more than once")
    if set(seen) != expected:
        missing = sorted(expected - set(seen))
        extra = sorted(set(seen) - expected)
        raise ValueError(f"box coverage mismatch; missing={missing}, extra={extra}")


def solution_rows(solution: MethodSolution) -> List[Dict[str, object]]:
    """Convert a method solution to serializable row dictionaries."""
    return [
        {
            "架次编号": trip.trip_id,
            "服务区编号": trip.service_id,
            "机型编号": trip.model_id,
            "货箱编号列表": ";".join(trip.box_ids),
            "总质量（kg）": trip.mass_kg,
            "总体积（m³）": trip.volume_m3,
            "往返时间（s）": trip.time_s,
            "架次能耗（kWh）": trip.energy_kwh,
            "返航SOC（%）": trip.return_soc_percent,
        }
        for trip in solution.trips
    ]


def model_trip_counts(solution: MethodSolution) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for trip in solution.trips:
        counts[trip.model_id] = counts.get(trip.model_id, 0) + 1
    return counts
