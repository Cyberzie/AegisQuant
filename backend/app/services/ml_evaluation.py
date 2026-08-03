from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from app.models.market_data import MarketData
from app.services.ml_dataset import build_ml_dataset
from app.services.ml_validation import (
    WalkForwardValidationResult,
    train_and_validate,
    walk_forward_validate,
)


@dataclass(frozen=True)
class MLEvaluationResult:
    symbol: str
    dataset_rows: int
    training_rows: int
    validation_rows: int
    direction_accuracy: float
    average_absolute_error_percent: float


@dataclass(frozen=True)
class MLWalkForwardEvaluationResult:
    symbol: str
    horizon: int
    dataset_rows: int
    total_training_rows: int
    total_validation_rows: int
    total_correct_predictions: int
    total_incorrect_predictions: int
    direction_accuracy: float
    average_absolute_error_percent: float
    folds: tuple


def evaluate_symbol(
    rows: Sequence[MarketData],
    *,
    symbol: str,
    horizon: int = 5,
    validation_fraction: float = 0.2,
) -> MLEvaluationResult:
    dataset = build_ml_dataset(
        rows,
        horizon=horizon,
    )

    if len(dataset.rows) < 10:
        raise ValueError(
            "Not enough usable rows for ML evaluation."
        )

    _, validation = train_and_validate(
        dataset,
        validation_fraction=validation_fraction,
    )

    return MLEvaluationResult(
        symbol=symbol.upper(),
        dataset_rows=len(dataset.rows),
        training_rows=validation.training_rows,
        validation_rows=validation.validation_rows,
        direction_accuracy=validation.direction_accuracy,
        average_absolute_error_percent=(
            validation.average_absolute_error_percent
        ),
    )


def evaluate_symbol_walk_forward(
    rows: Sequence[MarketData],
    *,
    symbol: str,
    horizon: int = 5,
    initial_training_fraction: float = 0.60,
    folds: int = 4,
    validation_window: int | None = None,
    learning_rate: float = 0.01,
    epochs: int = 500,
) -> MLWalkForwardEvaluationResult:
    """
    Evaluate an ML model using expanding-window walk-forward
    validation.

    The purge gap is automatically set to the prediction horizon.

    For example, with horizon=5:

        TRAIN | 5-ROW GAP | VALIDATE

    This prevents observations immediately after the training
    boundary from sharing overlapping future targets with the
    final training observations.
    """

    if horizon <= 0:
        raise ValueError(
            "Horizon must be greater than zero."
        )

    dataset = build_ml_dataset(
        rows,
        horizon=horizon,
    )

    if len(dataset.rows) < 10:
        raise ValueError(
            "Not enough usable rows for walk-forward ML evaluation."
        )

    validation: WalkForwardValidationResult = (
        walk_forward_validate(
            dataset,
            initial_training_fraction=(
                initial_training_fraction
            ),
            folds=folds,
            validation_window=validation_window,
            gap_rows=horizon,
            learning_rate=learning_rate,
            epochs=epochs,
        )
    )

    return MLWalkForwardEvaluationResult(
        symbol=symbol.upper(),
        horizon=horizon,
        dataset_rows=len(dataset.rows),
        total_training_rows=(
            validation.total_training_rows
        ),
        total_validation_rows=(
            validation.total_validation_rows
        ),
        total_correct_predictions=(
            validation.total_correct_predictions
        ),
        total_incorrect_predictions=(
            validation.total_incorrect_predictions
        ),
        direction_accuracy=(
            validation.direction_accuracy
        ),
        average_absolute_error_percent=(
            validation.average_absolute_error_percent
        ),
        folds=validation.folds,
    )