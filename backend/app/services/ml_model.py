from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import Sequence

from app.services.ml_dataset import MLDataset, MLDatasetRow


@dataclass(frozen=True)
class MLPrediction:
    direction: str
    confidence: float
    expected_return_percent: float


@dataclass(frozen=True)
class MLModel:
    feature_names: tuple[str, ...]
    weights: tuple[float, ...]
    bias: float
    mean_values: tuple[float, ...]
    scale_values: tuple[float, ...]


def _safe_scale(value: float) -> float:
    if abs(value) < 1e-12:
        return 1.0

    return value


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = exp(-value)
        return 1.0 / (1.0 + z)

    z = exp(value)
    return z / (1.0 + z)


def _standardize(
    values: Sequence[float],
    means: Sequence[float],
    scales: Sequence[float],
) -> list[float]:
    return [
        (float(value) - float(mean))
        / _safe_scale(float(scale))
        for value, mean, scale in zip(
            values,
            means,
            scales,
        )
    ]


def _dataset_matrix(
    dataset: MLDataset,
) -> tuple[list[list[float]], list[float]]:
    matrix: list[list[float]] = []
    targets: list[float] = []

    for row in dataset.rows:
        matrix.append(
            [
                row.features[name]
                for name in dataset.feature_names
            ]
        )
        targets.append(row.target_return)

    return matrix, targets


def _calculate_statistics(
    matrix: Sequence[Sequence[float]],
) -> tuple[list[float], list[float]]:
    if not matrix:
        raise ValueError(
            "Training dataset cannot be empty."
        )

    feature_count = len(matrix[0])

    means: list[float] = []
    scales: list[float] = []

    for column in range(feature_count):
        values = [
            float(row[column])
            for row in matrix
        ]

        mean = sum(values) / len(values)

        variance = sum(
            (value - mean) ** 2
            for value in values
        ) / len(values)

        scale = variance ** 0.5

        means.append(mean)
        scales.append(
            _safe_scale(scale)
        )

    return means, scales


def train_ml_model(
    dataset: MLDataset,
    *,
    learning_rate: float = 0.01,
    epochs: int = 500,
) -> MLModel:
    if not dataset.rows:
        raise ValueError(
            "Training dataset cannot be empty."
        )

    if learning_rate <= 0:
        raise ValueError(
            "Learning rate must be greater than zero."
        )

    if epochs <= 0:
        raise ValueError(
            "Epochs must be greater than zero."
        )

    matrix, targets = _dataset_matrix(dataset)

    means, scales = _calculate_statistics(matrix)

    standardized = [
        _standardize(
            row,
            means,
            scales,
        )
        for row in matrix
    ]

    feature_count = len(dataset.feature_names)

    weights = [0.0] * feature_count
    bias = 0.0

    # Normalize continuous returns into a bounded target.
    target_values = [
        max(
            -1.0,
            min(
                1.0,
                target / 10.0,
            ),
        )
        for target in targets
    ]

    for _ in range(epochs):
        weight_gradients = [
            0.0
        ] * feature_count

        bias_gradient = 0.0

        for values, target in zip(
            standardized,
            target_values,
        ):
            prediction = bias

            for weight, value in zip(
                weights,
                values,
            ):
                prediction += weight * value

            error = prediction - target

            for index, value in enumerate(values):
                weight_gradients[index] += (
                    error * value
                )

            bias_gradient += error

        sample_count = len(standardized)

        for index in range(feature_count):
            weights[index] -= (
                learning_rate
                * weight_gradients[index]
                / sample_count
            )

        bias -= (
            learning_rate
            * bias_gradient
            / sample_count
        )

    return MLModel(
        feature_names=dataset.feature_names,
        weights=tuple(weights),
        bias=bias,
        mean_values=tuple(means),
        scale_values=tuple(scales),
    )


def predict_ml_model(
    model: MLModel,
    row: MLDatasetRow,
) -> MLPrediction:
    values = [
        row.features[name]
        for name in model.feature_names
    ]

    standardized = _standardize(
        values,
        model.mean_values,
        model.scale_values,
    )

    prediction = model.bias

    for weight, value in zip(
        model.weights,
        standardized,
    ):
        prediction += weight * value

    expected_return_percent = prediction * 10.0

    probability = _sigmoid(
        abs(prediction)
    )

    if prediction > 0:
        direction = "BUY"
    elif prediction < 0:
        direction = "SELL"
    else:
        direction = "HOLD"

    confidence = min(
        max(probability, 0.0),
        1.0,
    )

    return MLPrediction(
        direction=direction,
        confidence=confidence,
        expected_return_percent=expected_return_percent,
    )