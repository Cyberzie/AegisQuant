from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from app.models.market_data import MarketData
from app.services.ml_dataset import build_ml_dataset
from app.services.ml_validation import train_and_validate


@dataclass(frozen=True)
class MLEvaluationResult:
    symbol: str
    dataset_rows: int
    training_rows: int
    validation_rows: int
    direction_accuracy: float
    average_absolute_error_percent: float


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