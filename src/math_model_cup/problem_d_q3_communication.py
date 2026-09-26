"""Communication physics for Problem D, question 3.

This program and code were completed with assistance from OpenAI Codex.
The authors reviewed and validated the resulting implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor, hypot, log10
from pathlib import Path

import numpy as np
from PIL import Image

from .problem_d_q1 import _ellipsoidal_distance_m, _parse_nodata
from .problem_d_q3 import LinkBudget, Position3D, Q3Data


@dataclass(frozen=True)
class DemGrid:
    elevations_m: np.ndarray
    x_origin: float
    y_origin: float
    x_scale: float
    y_scale: float
    nodata: float | None = None

    @property
    def rows(self) -> int:
        return int(self.elevations_m.shape[0])

    @property
    def columns(self) -> int:
        return int(self.elevations_m.shape[1])


@dataclass(frozen=True)
class LinkResult:
    link_type: str
    distance_km: float
    loss_db: float
    limit_db: float
    margin_db: float
    obstructed: bool
    available: bool


@dataclass(frozen=True)
class RelayLinkResult:
    access: LinkResult
    backhaul: LinkResult

    @property
    def available(self) -> bool:
        return self.access.available and self.backhaul.available

    @property
    def minimum_margin_db(self) -> float:
        return min(self.access.margin_db, self.backhaul.margin_db)


def load_dem_grid(path: Path) -> DemGrid:
    """Load the supplied WGS84 GeoTIFF into an immutable calculation object."""
    with Image.open(path) as image:
        elevations = np.asarray(image, dtype=float).copy()
        tie_point = image.tag_v2[33922]
        pixel_scale = image.tag_v2[33550]
        nodata = _parse_nodata(image.tag_v2.get(42113))
    x_scale = float(pixel_scale[0])
    y_scale = float(pixel_scale[1])
    if x_scale <= 0.0 or y_scale <= 0.0:
        raise ValueError("DEM pixel scales must be positive")
    return DemGrid(
        elevations_m=elevations,
        x_origin=float(tie_point[3]),
        y_origin=float(tie_point[4]),
        x_scale=x_scale,
        y_scale=y_scale,
        nodata=nodata,
    )


def fspl_db(frequency_mhz: float, distance_km: float) -> float:
    """Free-space path loss using MHz and km units."""
    if frequency_mhz <= 0.0 or distance_km <= 0.0:
        raise ValueError("frequency and distance must be positive")
    return 32.44 + 20.0 * log10(frequency_mhz) + 20.0 * log10(distance_km)


def raster_line_cells(
    dem: DemGrid,
    longitude1: float,
    latitude1: float,
    longitude2: float,
    latitude2: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Enumerate raster cells along a segment in time linear in line length."""
    x1 = (longitude1 - dem.x_origin) / dem.x_scale
    y1 = (dem.y_origin - latitude1) / dem.y_scale
    x2 = (longitude2 - dem.x_origin) / dem.x_scale
    y2 = (dem.y_origin - latitude2) / dem.y_scale
    sample_count = max(1, int(ceil(max(abs(x2 - x1), abs(y2 - y1)) * 4.0)))
    columns = np.floor(np.linspace(x1, x2, sample_count + 1)).astype(int)
    rows = np.floor(np.linspace(y1, y2, sample_count + 1)).astype(int)
    valid = (
        (rows >= 0)
        & (rows < dem.rows)
        & (columns >= 0)
        & (columns < dem.columns)
    )
    if not valid.any():
        raise ValueError("communication path lies outside DEM coverage")
    pairs = np.unique(np.column_stack((rows[valid], columns[valid])), axis=0)
    return pairs[:, 0], pairs[:, 1]


def _directional_limit_db(
    budget: LinkBudget,
    transmitter_power_dbm: float,
    transmitter_gain_dbi: float,
    receiver_gain_dbi: float,
) -> float:
    required_received_power_dbm = budget.sensitivity_dbm + budget.fade_margin_db
    return (
        transmitter_power_dbm
        + transmitter_gain_dbi
        + receiver_gain_dbi
        - budget.system_loss_db
        - required_received_power_dbm
    )


def link_loss_limit_db(budget: LinkBudget, link_type: str) -> float:
    """Return the stricter of the two directional link-budget limits."""
    if link_type == "direct":
        forward = _directional_limit_db(
            budget,
            budget.transport_tx_dbm,
            budget.transport_gain_dbi,
            budget.gateway_gain_dbi,
        )
        reverse = _directional_limit_db(
            budget,
            budget.gateway_tx_dbm,
            budget.gateway_gain_dbi,
            budget.transport_gain_dbi,
        )
    elif link_type == "access":
        forward = _directional_limit_db(
            budget,
            budget.transport_tx_dbm,
            budget.transport_gain_dbi,
            budget.relay_access_gain_dbi,
        )
        reverse = _directional_limit_db(
            budget,
            budget.relay_access_tx_dbm,
            budget.relay_access_gain_dbi,
            budget.transport_gain_dbi,
        )
    elif link_type == "backhaul":
        forward = _directional_limit_db(
            budget,
            budget.relay_backhaul_tx_dbm,
            budget.relay_backhaul_gain_dbi,
            budget.gateway_gain_dbi,
        )
        reverse = _directional_limit_db(
            budget,
            budget.gateway_tx_dbm,
            budget.gateway_gain_dbi,
            budget.relay_backhaul_gain_dbi,
        )
    else:
        raise ValueError(f"unknown link type: {link_type}")
    return min(forward, reverse)


def distance_3d_km(origin: Position3D, destination: Position3D) -> float:
    horizontal_m = _ellipsoidal_distance_m(
        origin.longitude,
        origin.latitude,
        destination.longitude,
        destination.latitude,
    )
    return hypot(horizontal_m, destination.altitude_m - origin.altitude_m) / 1000.0


def path_is_obstructed(dem: DemGrid, origin: Position3D, destination: Position3D) -> bool:
    """Conservatively test all DEM cells touched by the ground projection."""
    rows, columns = raster_line_cells(
        dem,
        origin.longitude,
        origin.latitude,
        destination.longitude,
        destination.latitude,
    )
    delta_lon = destination.longitude - origin.longitude
    delta_lat = destination.latitude - origin.latitude
    denominator = delta_lon * delta_lon + delta_lat * delta_lat
    endpoint_cells = {
        (
            floor((dem.y_origin - point.latitude) / dem.y_scale),
            floor((point.longitude - dem.x_origin) / dem.x_scale),
        )
        for point in (origin, destination)
    }
    for row, column in zip(rows.tolist(), columns.tolist()):
        if (row, column) in endpoint_cells:
            continue
        terrain_m = float(dem.elevations_m[row, column])
        if not np.isfinite(terrain_m):
            continue
        if dem.nodata is not None and np.isclose(
            terrain_m, dem.nodata, rtol=0.0, atol=1e-12
        ):
            continue
        cell_lon = dem.x_origin + (column + 0.5) * dem.x_scale
        cell_lat = dem.y_origin - (row + 0.5) * dem.y_scale
        if denominator <= 1e-24:
            fraction = 0.0
        else:
            fraction = (
                (cell_lon - origin.longitude) * delta_lon
                + (cell_lat - origin.latitude) * delta_lat
            ) / denominator
            fraction = min(1.0, max(0.0, fraction))
        line_altitude_m = origin.altitude_m + fraction * (
            destination.altitude_m - origin.altitude_m
        )
        if terrain_m >= line_altitude_m:
            return True
    return False


def evaluate_link(
    budget: LinkBudget,
    dem: DemGrid,
    origin: Position3D,
    destination: Position3D,
    link_type: str,
) -> LinkResult:
    distance_km = distance_3d_km(origin, destination)
    # Co-located radios are represented by a one-millimetre separation to keep
    # the logarithmic model finite without changing any operational decision.
    propagation_distance_km = max(distance_km, 1e-6)
    obstructed = path_is_obstructed(dem, origin, destination)
    loss_db = fspl_db(budget.frequency_mhz, propagation_distance_km)
    if obstructed:
        loss_db += budget.obstruction_loss_db
    limit_db = link_loss_limit_db(budget, link_type)
    margin_db = limit_db - loss_db
    return LinkResult(
        link_type=link_type,
        distance_km=distance_km,
        loss_db=loss_db,
        limit_db=limit_db,
        margin_db=margin_db,
        obstructed=obstructed,
        available=margin_db >= -1e-9,
    )


def evaluate_relay_link(
    budget: LinkBudget,
    dem: DemGrid,
    transport: Position3D,
    relay: Position3D,
    gateway: Position3D,
) -> RelayLinkResult:
    return RelayLinkResult(
        access=evaluate_link(budget, dem, transport, relay, "access"),
        backhaul=evaluate_link(budget, dem, relay, gateway, "backhaul"),
    )


class CommunicationEngine:
    """Terrain-aware direct and two-hop communication evaluator."""

    def __init__(self, data: Q3Data, dem: DemGrid | None = None) -> None:
        self.data = data
        self.dem = dem if dem is not None else load_dem_grid(data.transport.dem_path)
        center = data.transport.nodes["O01"]
        self.gateway_position = Position3D(
            center.longitude,
            center.latitude,
            center.elevation_m + data.link_budget.gateway_height_agl_m,
        )

    @property
    def direct_limit_db(self) -> float:
        return link_loss_limit_db(self.data.link_budget, "direct")

    @property
    def access_limit_db(self) -> float:
        return link_loss_limit_db(self.data.link_budget, "access")

    @property
    def backhaul_limit_db(self) -> float:
        return link_loss_limit_db(self.data.link_budget, "backhaul")

    def total_loss_db(self, distance_km: float, obstructed: bool) -> float:
        loss = fspl_db(self.data.link_budget.frequency_mhz, distance_km)
        if obstructed:
            loss += self.data.link_budget.obstruction_loss_db
        return loss

    def direct_available(self, transport: Position3D) -> LinkResult:
        return evaluate_link(
            self.data.link_budget,
            self.dem,
            transport,
            self.gateway_position,
            "direct",
        )

    def relay_available(
        self,
        transport: Position3D,
        relay: Position3D,
    ) -> RelayLinkResult:
        return evaluate_relay_link(
            self.data.link_budget,
            self.dem,
            transport,
            relay,
            self.gateway_position,
        )
