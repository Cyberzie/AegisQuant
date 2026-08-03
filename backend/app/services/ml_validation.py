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


@dataclass(frozen=True)
class WalkForwardFold:
    fold_number: int
    training_rows: int
    validation_rows: int
    gap_rows: int
    correct_direction_predictions: int
    incorrect_direction_predictions: int
    direction_accuracy: float
    average_absolute_error_percent: float


@dataclass(frozen=True)
class WalkForwardValidationResult:
    total_training_rows: int
    total_validation_rows: int
    total_correct_predictions: int
    total_incorrect_predictions: int
    direction_accuracy: float
    average_absolute_error_percent: float
    folds: tuple[WalkForwardFold, ...]


def split_dataset(
    dataset: MLDataset,
    *,
    validation_fraction: float = 0.2,
) -> tuple[MLDataset, MLDataset]:
    """
    Split a chronological dataset into training and validation sets.

    No shuffling is performed.

    The earlier portion is always used for training and the later
    portion for validation, preserving time ordering.
    """

    if not dataset.rows:
        raise ValueError(
            "Dataset cannot be empty."
        )

    if not 0 < validation_fraction < 1:
        raise ValueError(
            "Validation fraction must be between 0 and 1."
        )

    if len(dataset.rows) < 2:
        raise ValueError(
            "Dataset must contain at least two rows."
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
    """
    Evaluate a trained model against a chronological validation dataset.

    The model is never retrained inside this function.
    """

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
        direction_accuracy=(
            correct / total
        ),
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
    """
    Train on the earlier portion of the dataset and validate on the
    later portion.

    This function preserves the original chronological holdout
    behaviour used by the existing ML evaluation endpoint.
    """

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
            validation_rows=(
                validation_result.validation_rows
            ),
            correct_direction_predictions=(
                validation_result
                .correct_direction_predictions
            ),
            incorrect_direction_predictions=(
                validation_result
                .incorrect_direction_predictions
            ),
            direction_accuracy=(
                validation_result.direction_accuracy
            ),
            average_absolute_error_percent=(
                validation_result
                .average_absolute_error_percent
            ),
        ),
    )


def _build_dataset_slice(
    dataset: MLDataset,
    start: int,
    end: int,
) -> MLDataset:
    return MLDataset(
        rows=tuple(
            dataset.rows[start:end]
        ),
        feature_names=dataset.feature_names,
    )


def walk_forward_validate(
    dataset: MLDataset,
    *,
    initial_training_fraction: float = 0.60,
    folds: int = 4,
    validation_window: int | None = None,
    gap_rows: int = 0,
    learning_rate: float = 0.01,
    epochs: int = 500,
) -> WalkForwardValidationResult:
    """
    Perform expanding-window chronological walk-forward validation.

    Structure:

        TRAIN TRAIN TRAIN | GAP | VALIDATE
        TRAIN TRAIN TRAIN TRAIN | GAP | VALIDATE
        TRAIN TRAIN TRAIN TRAIN TRAIN | GAP | VALIDATE

    The training window expands after every validation fold.

    No shuffling is performed.

    ``gap_rows`` is important when the target represents a future
    horizon. For example, when predicting a 5-row-ahead return,
    callers should normally use:

        gap_rows=5

    This prevents the validation observations immediately following
    the training boundary from sharing future target information with
    the final training observations.

    The function itself does not assume a particular prediction
    horizon because the MLDataset does not expose that metadata.
    The caller therefore controls the appropriate purge gap.
    """

    if not dataset.rows:
        raise ValueError(
            "Dataset cannot be empty."
        )

    if not 0 < initial_training_fraction < 1:
        raise ValueError(
            "Initial training fraction must be between 0 and 1."
        )

    if folds <= 0:
        raise ValueError(
            "Number of folds must be greater than zero."
        )

    if gap_rows < 0:
        raise ValueError(
            "Gap rows cannot be negative."
        )

    if learning_rate <= 0:
        raise ValueError(
            "Learning rate must be greater than zero."
        )

    if epochs <= 0:
        raise ValueError(
            "Epochs must be greater than zero."
        )

    total_rows = len(dataset.rows)

    if total_rows < 10:
        raise ValueError(
            "Dataset must contain at least 10 rows "
            "for walk-forward validation."
        )

    initial_training_rows = int(
        total_rows
        * initial_training_fraction
    )

    initial_training_rows = max(
        5,
        initial_training_rows,
    )

    if initial_training_rows >= total_rows:
        raise ValueError(
            "Initial training window leaves no "
            "validation data."
        )

    remaining_rows = (
        total_rows
        - initial_training_rows
        - gap_rows
    )

    if remaining_rows <= 0:
        raise ValueError(
            "Not enough rows remain after the "
            "training window and gap."
        )

    if validation_window is None:
        validation_window = max(
            1,
            remaining_rows // folds,
        )

    if validation_window <= 0:
        raise ValueError(
            "Validation window must be greater than zero."
        )

    fold_results: list[WalkForwardFold] = []

    training_end = initial_training_rows
    fold_number = 1

    while (
        fold_number <= folds
        and training_end + gap_rows < total_rows
    ):
        validation_start = (
            training_end + gap_rows
        )

        validation_end = min(
            validation_start
            + validation_window,
            total_rows,
        )

        training_dataset = _build_dataset_slice(
            dataset,
            0,
            training_end,
        )

        validation_dataset = _build_dataset_slice(
            dataset,
            validation_start,
            validation_end,
        )

        if len(training_dataset.rows) < 5:
            break

        if not validation_dataset.rows:
            break

        model = train_ml_model(
            training_dataset,
            learning_rate=learning_rate,
            epochs=epochs,
        )

        validation_result = validate_ml_model(
            model,
            validation_dataset,
        )

        fold_results.append(
            WalkForwardFold(
                fold_number=fold_number,
                training_rows=len(
                    training_dataset.rows
                ),
                validation_rows=(
                    validation_result.validation_rows
                ),
                gap_rows=(
                    validation_start
                    - training_end
                ),
                correct_direction_predictions=(
                    validation_result
                    .correct_direction_predictions
                ),
                incorrect_direction_predictions=(
                    validation_result
                    .incorrect_direction_predictions
                ),
                direction_accuracy=(
                    validation_result.direction_accuracy
                ),
                average_absolute_error_percent=(
                    validation_result
                    .average_absolute_error_percent
                ),
            )
        )

        training_end = validation_end
        fold_number += 1

    if not fold_results:
        raise ValueError(
            "Unable to construct walk-forward "
            "validation folds."
        )

    total_validation_rows = sum(
        fold.validation_rows
        for fold in fold_results
    )

    total_correct_predictions = sum(
        fold.correct_direction_predictions
        for fold in fold_results
    )

    total_incorrect_predictions = sum(
        fold.incorrect_direction_predictions
        for fold in fold_results
    )

    total_training_rows = sum(
        fold.training_rows
        for fold in fold_results
    )

    weighted_absolute_error = sum(
        fold.average_absolute_error_percent
        * fold.validation_rows
        for fold in fold_results
    )

    direction_accuracy = (
        total_correct_predictions
        / total_validation_rows
    )

    average_absolute_error_percent = (
        weighted_absolute_error
        / total_validation_rows
    )

    return WalkForwardValidationResult(
        total_training_rows=total_training_rows,
        total_validation_rows=(
            total_validation_rows
        ),
        total_correct_predictions=(
            total_correct_predictions
        ),
        total_incorrect_predictions=(
            total_incorrect_predictions
        ),
        direction_accuracy=direction_accuracy,
        average_absolute_error_percent=(
            average_absolute_error_percent
        ),
        folds=tuple(fold_results),
    )