from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from app.models.market_data import MarketData
from app.services.feature_engineering import (
    MarketFeature,
    build_market_features,
)


@dataclass(frozen=True)
class MLDatasetRow:
    timestamp: object
    features: dict[str, float]
    target_return: float
    target_direction: str


@dataclass(frozen=True)
class MLDataset:
    rows: tuple[MLDatasetRow, ...]
    feature_names: tuple[str, ...]


FEATURE_NAMES = (
    "close",
    "volume",
    "return_1",
    "return_5",
    "volatility_10",
    "sma_20",
    "ema_20",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_histogram",
    "bollinger_middle",
    "bollinger_upper",
    "bollinger_lower",
    "atr_14",
)


def _feature_values(
    feature: MarketFeature,
) -> dict[str, float] | None:
    values = {
        "close": feature.close,
        "volume": feature.volume,
        "return_1": feature.return_1,
        "return_5": feature.return_5,
        "volatility_10": feature.volatility_10,
        "sma_20": feature.sma_20,
        "ema_20": feature.ema_20,
        "rsi_14": feature.rsi_14,
        "macd": feature.macd,
        "macd_signal": feature.macd_signal,
        "macd_histogram": feature.macd_histogram,
        "bollinger_middle": feature.bollinger_middle,
        "bollinger_upper": feature.bollinger_upper,
        "bollinger_lower": feature.bollinger_lower,
        "atr_14": feature.atr_14,
    }

    if any(value is None for value in values.values()):
        return None

    return {
        name: float(value)
        for name, value in values.items()
    }


def _direction_from_return(
    target_return: float,
    threshold_percent: float,
) -> str:
    if target_return > threshold_percent:
        return "BUY"

    if target_return < -threshold_percent:
        return "SELL"

    return "HOLD"


def build_ml_dataset(
    rows: Sequence[MarketData],
    *,
    horizon: int = 5,
    direction_threshold_percent: float = 0.0,
) -> MLDataset:
    if horizon <= 0:
        raise ValueError(
            "Horizon must be greater than zero."
        )

    if direction_threshold_percent < 0:
        raise ValueError(
            "Direction threshold cannot be negative."
        )

    ordered_rows = sorted(
        rows,
        key=lambda row: row.timestamp,
    )

    features = build_market_features(
        ordered_rows
    )

    dataset_rows: list[MLDatasetRow] = []

    maximum_index = len(ordered_rows) - horizon

    for index in range(maximum_index):
        feature_values = _feature_values(
            features[index]
        )

        if feature_values is None:
            continue

        current_close = float(
            ordered_rows[index].close
        )

        future_close = float(
            ordered_rows[index + horizon].close
        )

        if current_close <= 0:
            continue

        target_return = (
            (future_close - current_close)
            / current_close
        ) * 100

        target_direction = _direction_from_return(
            target_return,
            direction_threshold_percent,
        )

        dataset_rows.append(
            MLDatasetRow(
                timestamp=features[index].timestamp,
                features=feature_values,
                target_return=target_return,
                target_direction=target_direction,
            )
        )

    return MLDataset(
        rows=tuple(dataset_rows),
        feature_names=FEATURE_NAMES,
    )