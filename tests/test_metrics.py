from math_model_cup.metrics import mean_absolute_error


def test_mean_absolute_error_returns_average_absolute_difference() -> None:
    assert mean_absolute_error([1.0, 2.0, 3.0], [2.0, 2.0, 5.0]) == 1.0


def test_mean_absolute_error_rejects_sequences_with_different_lengths() -> None:
    try:
        mean_absolute_error([1.0], [1.0, 2.0])
    except ValueError as error:
        assert str(error) == "actual and predicted must have the same length"
    else:
        raise AssertionError("expected ValueError")
