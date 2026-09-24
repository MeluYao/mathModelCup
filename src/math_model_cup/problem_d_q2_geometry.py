"""All-pairs terrain-aware route geometry for Problem D, question 2."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Mapping, Tuple

import numpy as np
from PIL import Image

from .problem_d_q1 import (
    _ellipsoidal_distance_m,
    _geotiff_epsg,
    _line_cells,
    _parse_nodata,
)
from .problem_d_q2 import Node, Q2Data


@dataclass(frozen=True)
class ArcGeometry:
    origin_id: str
    destination_id: str
    distance_m: float
    peak_ground_m: float
    cruise_altitude_m: float
    climb_m: float
    descent_m: float
    peak_row: int
    peak_column: int


def _arc(
    origin: Node,
    destination: Node,
    *,
    distance_m: float,
    peak_ground_m: float,
    peak_row: int,
    peak_column: int,
) -> ArcGeometry:
    cruise_altitude_m = peak_ground_m + 50.0
    return ArcGeometry(
        origin_id=origin.node_id,
        destination_id=destination.node_id,
        distance_m=distance_m,
        peak_ground_m=peak_ground_m,
        cruise_altitude_m=cruise_altitude_m,
        climb_m=max(0.0, cruise_altitude_m - origin.operation_altitude_m),
        descent_m=max(0.0, cruise_altitude_m - destination.operation_altitude_m),
        peak_row=peak_row,
        peak_column=peak_column,
    )


def build_arc_matrix(
    data: Q2Data,
) -> Mapping[Tuple[str, str], ArcGeometry]:
    """Return terrain-aware geometry for every directed pair of distinct nodes."""
    with Image.open(data.dem_path) as image:
        dem = np.asarray(image, dtype=float)
        tie_point = image.tag_v2[33922]
        pixel_scale = image.tag_v2[33550]
        geo_keys = image.tag_v2[34735]
        nodata = _parse_nodata(image.tag_v2.get(42113))

    if _geotiff_epsg(geo_keys) != 4326:
        raise ValueError("question 2 requires a WGS84 geographic DEM")
    x_origin = float(tie_point[3])
    y_origin = float(tie_point[4])
    x_scale = float(pixel_scale[0])
    y_scale = float(pixel_scale[1])

    arcs: Dict[Tuple[str, str], ArcGeometry] = {}
    ordered_nodes = [data.nodes[node_id] for node_id in sorted(data.nodes)]
    for left, right in combinations(ordered_nodes, 2):
        rows, columns = _line_cells(
            left.longitude,
            left.latitude,
            right.longitude,
            right.latitude,
            x_origin=x_origin,
            y_origin=y_origin,
            x_scale=x_scale,
            y_scale=y_scale,
            rows=dem.shape[0],
            columns=dem.shape[1],
        )
        elevations = dem[rows, columns]
        valid = np.isfinite(elevations)
        if nodata is not None:
            valid &= ~np.isclose(elevations, nodata, rtol=0.0, atol=1e-12)
        if not valid.any():
            raise ValueError(f"route {left.node_id}-{right.node_id} has no valid DEM pixels")
        valid_elevations = elevations[valid]
        valid_rows = rows[valid]
        valid_columns = columns[valid]
        peak_index = int(np.argmax(valid_elevations))
        peak_ground_m = float(valid_elevations[peak_index])
        peak_row = int(valid_rows[peak_index])
        peak_column = int(valid_columns[peak_index])
        distance_m = _ellipsoidal_distance_m(
            left.longitude,
            left.latitude,
            right.longitude,
            right.latitude,
        )

        arcs[(left.node_id, right.node_id)] = _arc(
            left,
            right,
            distance_m=distance_m,
            peak_ground_m=peak_ground_m,
            peak_row=peak_row,
            peak_column=peak_column,
        )
        arcs[(right.node_id, left.node_id)] = _arc(
            right,
            left,
            distance_m=distance_m,
            peak_ground_m=peak_ground_m,
            peak_row=peak_row,
            peak_column=peak_column,
        )

    return arcs
