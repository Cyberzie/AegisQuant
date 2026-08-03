from __future__ import annotations

from dataclasses import dataclass

from app.services.ml_dataset import MLDataset
from app.services.ml_model import (
    MLModel,
    predict_ml_model,
    train_ml_model,
)


@dataclass(frozen=True)
class MLValidationResult:
    training_rows: int
    validation_rows: int
    correct_direction_predictions: int
    incorrect_direction_predictions: int
    direction_accuracy: float
    average_absolute_error_percent: float


def split_dataset(
    dataset: MLDataset,
    *,
    validation_fraction: float = 0.2,
) -> tuple[MLDataset, MLDataset]:
    if not dataset.rows:
        raise ValueError(
            "Dataset cannot be empty."
        )

    if not 0 < validation_fraction < 1:
        raise ValueError(
            "Validation fraction must be between 0 and 1."
        )

    split_index = int(
        len(dataset.rows)
        * (1.0 - validation_fraction)
    )

    split_index = max(
        1,
        min(
            split_index,
            len(dataset.rows) - 1,
        ),
    )

    training_rows = dataset.rows[:split_index]
    validation_rows = dataset.rows[split_index:]

    training_dataset = MLDataset(
        rows=tuple(training_rows),
        feature_names=dataset.feature_names,
    )

    validation_dataset = MLDataset(
        rows=tuple(validation_rows),
        feature_names=dataset.feature_names,
    )

    return (
        training_dataset,
        validation_dataset,
    )


def validate_ml_model(
    model: MLModel,
    dataset: MLDataset,
) -> MLValidationResult:
    if not dataset.rows:
        raise ValueError(
            "Validation dataset cannot be empty."
        )

    correct = 0
    incorrect = 0
    absolute_errors: list[float] = []

    for row in dataset.rows:
        prediction = predict_ml_model(
            model,
            row,
        )

        if prediction.direction == row.target_direction:
            correct += 1
        else:
            incorrect += 1

        absolute_errors.append(
            abs(
                prediction.expected_return_percent
                - row.target_return
            )
        )

    total = len(dataset.rows)

    return MLValidationResult(
        training_rows=0,
        validation_rows=total,
        correct_direction_predictions=correct,
        incorrect_direction_predictions=incorrect,
        direction_accuracy=correct / total,
        average_absolute_error_percent=(
            sum(absolute_errors) / total
        ),
    )


def train_and_validate(
    dataset: MLDataset,
    *,
    validation_fraction: float = 0.2,
    learning_rate: float = 0.01,
    epochs: int = 500,
) -> tuple[MLModel, MLValidationResult]:
    training_dataset, validation_dataset = (
        split_dataset(
            dataset,
            validation_fraction=validation_fraction,
        )
    )

    model = train_ml_model(
        training_dataset,
        learning_rate=learning_rate,
        epochs=epochs,
    )

    validation_result = validate_ml_model(
        model,
        validation_dataset,
    )

    return (
        model,
        MLValidationResult(
            training_rows=len(
                training_dataset.rows
            ),
            validation_rows=validation_result.validation_rows,
            correct_direction_predictions=(
                validation_result.correct_direction_predictions
            ),
            incorrect_direction_predictions=(
                validation_result.incorrect_direction_predictions
            ),
            direction_accuracy=(
                validation_result.direction_accuracy
            ),
            average_absolute_error_percent=(
                validation_result.average_absolute_error_percent
            ),
        ),
    )