"""Small, dependency-free evaluation metrics."""

from __future__ import annotations

from collections.abc import Sequence


def mean_absolute_error(actual: Sequence[float], predicted: Sequence[float]) -> float:
    """Return the average absolute difference between paired numeric values."""
    if len(actual) != len(predicted):
        raise ValueError("actual and predicted must have the same length")
    if not actual:
        raise ValueError("actual and predicted must not be empty")
    return sum(abs(value - estimate) for value, estimate in zip(actual, predicted)) / len(actual)
