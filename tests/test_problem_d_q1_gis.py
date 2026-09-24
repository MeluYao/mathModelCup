from __future__ import annotations

import math
import os
from pathlib import Path

import pytest

from math_model_cup.problem_d_q1 import (
    ServiceGeometry,
    _ellipsoidal_distance_m,
    _great_circle_distance_m,
    _line_cells,
)


def _pairs(rows, columns):
    return set(zip(rows.tolist(), columns.tolist()))


def test_all_touched_diagonal_includes_corner_neighbours() -> None:
    rows, columns = _line_cells(
        0.2,
        2.8,
        2.8,
        0.2,
        x_origin=0.0,
        y_origin=3.0,
        x_scale=1.0,
        y_scale=1.0,
        rows=3,
        columns=3,
    )

    assert _pairs(rows, columns) == {
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
        (1, 2),
        (2, 1),
        (2, 2),
    }


def test_all_touched_boundary_line_includes_both_pixel_rows() -> None:
    rows, columns = _line_cells(
        0.2,
        2.0,
        2.8,
        2.0,
        x_origin=0.0,
        y_origin=3.0,
        x_scale=1.0,
        y_scale=1.0,
        rows=3,
        columns=3,
    )

    assert _pairs(rows, columns) == {
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 1),
        (1, 2),
    }


def test_wgs84_distance_is_close_to_but_not_identical_to_haversine() -> None:
    ellipsoidal = _ellipsoidal_distance_m(109.0, 23.0, 109.2, 23.1)
    spherical = _great_circle_distance_m(109.0, 23.0, 109.2, 23.1)

    assert 22_000.0 < ellipsoidal < 24_000.0
    assert not math.isclose(ellipsoidal, spherical, rel_tol=1e-8)
    assert abs(ellipsoidal - spherical) / ellipsoidal < 0.01


def test_geometry_records_descent_and_peak_pixel_coordinates() -> None:
    geometry = ServiceGeometry(
        service_id="S001",
        distance_m=1_000.0,
        peak_ground_m=120.0,
        cruise_altitude_m=170.0,
        center_climb_m=70.0,
        service_climb_m=40.0,
        haversine_distance_m=999.0,
        peak_row=5,
        peak_column=6,
        peak_longitude=109.1,
        peak_latitude=23.1,
    )

    assert geometry.outbound_descent_m == 40.0
    assert geometry.return_descent_m == 70.0
    assert geometry.return_climb_m == 40.0


@pytest.mark.skipif(
    not os.environ.get("D_PROBLEM_DIR"), reason="D_PROBLEM_DIR is not configured"
)
def test_real_dem_metadata_and_routes_are_fully_recorded() -> None:
    from math_model_cup.problem_d_q1 import load_problem_data

    data = load_problem_data(Path(os.environ["D_PROBLEM_DIR"]))

    assert data.dem_metadata is not None
    assert data.dem_metadata.epsg == 4326
    assert data.dem_metadata.columns == 1486
    assert data.dem_metadata.rows == 1309
    assert data.dem_metadata.nodata is None
    assert math.isclose(data.dem_metadata.pixel_width_degrees, 1.0 / 3600.0)
    assert len(data.geometries) == 15
    assert all(geometry.peak_row >= 0 for geometry in data.geometries.values())
    assert all(geometry.peak_column >= 0 for geometry in data.geometries.values())
    assert all(geometry.distance_m > 0.0 for geometry in data.geometries.values())
