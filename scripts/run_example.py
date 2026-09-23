"""Run a deterministic environment smoke test."""

from math_model_cup.metrics import mean_absolute_error


if __name__ == "__main__":
    score = mean_absolute_error([1.0, 2.0, 3.0], [2.0, 2.0, 5.0])
    print(f"Example MAE: {score:.2f}")
