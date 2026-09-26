from __future__ import annotations

from dataclasses import replace

import pytest

from math_model_cup.problem_d_q2 import (
    DeliveryRecord,
    InfeasibleQ2Error,
    Q2Box,
    Q2Solution,
)
from math_model_cup.problem_d_q2_urgent import (
    URGENT_PRIORITY_MODE,
    is_urgent_box,
    required_deadline_s,
    urgent_lateness,
)
from math_model_cup.problem_d_q2_schedule import schedule_trips_greedy
from math_model_cup.problem_d_q2_validation import validate_q2_solution

from test_problem_d_q2_schedule import schedule_case


def _box(
    box_id: str,
    *,
    material_type: str = "饮用水",
    first_batch: bool = False,
    first_deadline_s: float | None = None,
    expected_time_s: float = 3600.0,
) -> Q2Box:
    return Q2Box(
        box_id=box_id,
        service_id=f"S{box_id}",
        material_type=material_type,
        mass_kg=1.0,
        volume_m3=0.01,
        first_batch=first_batch,
        first_deadline_s=first_deadline_s,
        expected_time_s=expected_time_s,
        priority_weight=1.0,
    )


def test_first_batch_or_medical_boxes_are_urgent() -> None:
    assert is_urgent_box(_box("1", first_batch=True, first_deadline_s=1800.0))
    assert is_urgent_box(_box("2", material_type="医疗物资"))
    assert not is_urgent_box(_box("3"))


def test_required_deadline_uses_earliest_urgent_limit_and_normal_expected_time() -> None:
    urgent = _box(
        "1",
        material_type="医疗物资",
        first_batch=True,
        first_deadline_s=1800.0,
        expected_time_s=2400.0,
    )
    normal = _box("2", expected_time_s=5400.0)

    assert required_deadline_s(urgent) == 1800.0
    assert required_deadline_s(normal) == 5400.0


def test_urgent_lateness_ignores_normal_box_lateness() -> None:
    boxes = {
        "U": _box("U", first_batch=True, first_deadline_s=100.0),
        "N": _box("N", expected_time_s=100.0),
    }
    data = type("Data", (), {"boxes": boxes})()
    solution = Q2Solution(
        method="test",
        trips=(),
        deliveries=(
            DeliveryRecord("U", "T1", "SU", 112.5),
            DeliveryRecord("N", "T2", "SN", 999.0),
        ),
        objective=(0.0, 0.0, 0, 0.0),
        runtime_s=0.0,
        solver_status="FEASIBLE",
    )

    assert urgent_lateness(data, solution) == (1, 12.5)


def test_urgent_schedule_uses_new_objective_and_all_box_deadlines(schedule_case) -> None:
    data, _, plans = schedule_case
    normal_box = replace(
        data.boxes["B2"],
        first_batch=False,
        first_deadline_s=None,
        expected_time_s=3600.0,
    )
    urgent_data = replace(data, boxes={**data.boxes, "B2": normal_box})

    solution = schedule_trips_greedy(
        urgent_data,
        plans,
        method="urgent",
        evaluation_mode=URGENT_PRIORITY_MODE,
    )

    assert len(solution.objective) == 4
    assert solution.objective[2] == 2.0


def test_urgent_schedule_rejects_normal_box_that_cannot_meet_expected_time(
    schedule_case,
) -> None:
    data, _, plans = schedule_case
    normal_box = replace(
        data.boxes["B2"],
        first_batch=False,
        first_deadline_s=None,
        expected_time_s=1.0,
    )
    urgent_data = replace(data, boxes={**data.boxes, "B2": normal_box})

    with pytest.raises(InfeasibleQ2Error):
        schedule_trips_greedy(
            urgent_data,
            plans,
            method="urgent",
            evaluation_mode=URGENT_PRIORITY_MODE,
        )


def test_urgent_validator_classifies_normal_deadline_violation(schedule_case) -> None:
    data, arcs, plans = schedule_case
    solution = schedule_trips_greedy(data, plans, method="published")
    b2_delivery = next(record for record in solution.deliveries if record.box_id == "B2")
    normal_box = replace(
        data.boxes["B2"],
        first_batch=False,
        first_deadline_s=None,
        expected_time_s=b2_delivery.delivery_time_s - 1.0,
    )
    urgent_data = replace(data, boxes={**data.boxes, "B2": normal_box})

    report = validate_q2_solution(
        urgent_data,
        arcs,
        solution,
        evaluation_mode=URGENT_PRIORITY_MODE,
    )

    assert "normal_deadline" in {issue.code for issue in report.issues}
