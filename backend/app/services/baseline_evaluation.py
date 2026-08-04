from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from app.models.market_data import MarketData
from app.services.adaptive_ensemble import (
    AdaptiveEnsembleState,
)
from app.services.ensemble_signal import (
    combine_signals,
)
from app.services.ml_dataset import (
    MLDataset,
    MLDatasetRow,
    build_ml_dataset,
)
from app.services.ml_model import (
    MLModel,
    predict_ml_model,
    train_ml_model,
)
from app.services.signal_engine import (
    SignalResult,
    generate_signal,
)


@dataclass(frozen=True)
class StrategyEvaluation:
    name: str
    prediction_count: int
    actionable_predictions: int
    correct_direction_predictions: int
    incorrect_direction_predictions: int
    direction_accuracy: float
    average_return_percent: float
    total_return_percent: float
    average_net_return_percent: float
    total_net_return_percent: float
    winning_predictions: int
    losing_predictions: int
    win_rate: float
    profit_factor: float


@dataclass(frozen=True)
class BaselineComparisonResult:
    symbol: str
    horizon: int
    dataset_rows: int
    total_training_rows: int
    total_validation_rows: int
    folds: int
    gap_rows: int
    strategies: tuple[StrategyEvaluation, ...]


def _rule_signal_from_features(
    row: MLDatasetRow,
) -> SignalResult:
    values = row.features

    return generate_signal(
        rsi_14=values["rsi_14"],
        macd=values["macd"],
        macd_signal=values["macd_signal"],
        sma_20=values["sma_20"],
        ema_20=values["ema_20"],
        close=values["close"],
    )


def _target_direction(
    target_return: float,
) -> str:
    if target_return > 0.0:
        return "BUY"

    if target_return < 0.0:
        return "SELL"

    return "HOLD"


def _evaluate_prediction(
    signal: str,
    target_return: float,
    *,
    transaction_cost_percent: float,
    slippage_percent: float,
) -> tuple[bool, float, float, bool]:

    if signal == "HOLD":
        return (
            target_return == 0.0,
            0.0,
            0.0,
            False,
        )

    if signal == "BUY":
        predicted_return = target_return
    elif signal == "SELL":
        predicted_return = -target_return
    else:
        raise ValueError(
            f"Invalid prediction signal: {signal}"
        )

    total_cost = (
        2.0 * transaction_cost_percent
        + 2.0 * slippage_percent
    )

    total_cost_fraction = (
        total_cost / 100.0
    )

    net_return = (
        predicted_return
        - total_cost_fraction
    )

    profitable = net_return > 0.0

    correct_direction = (
        (
            signal == "BUY"
            and target_return > 0.0
        )
        or (
            signal == "SELL"
            and target_return < 0.0
        )
    )

    return (
        correct_direction,
        predicted_return,
        net_return,
        profitable,
    )


def _calculate_strategy_evaluation(
    name: str,
    predictions: Sequence[tuple[str, float]],
    *,
    transaction_cost_percent: float,
    slippage_percent: float,
) -> StrategyEvaluation:

    prediction_count = len(predictions)

    actionable = [
        item
        for item in predictions
        if item[0] != "HOLD"
    ]

    correct = 0
    incorrect = 0

    raw_returns: list[float] = []
    net_returns: list[float] = []

    winning = 0
    losing = 0

    for signal, target_return in predictions:
        (
            is_correct,
            raw_return,
            net_return,
            profitable,
        ) = _evaluate_prediction(
            signal,
            target_return,
            transaction_cost_percent=(
                transaction_cost_percent
            ),
            slippage_percent=slippage_percent,
        )

        if signal == "HOLD":
            continue

        if is_correct:
            correct += 1
        else:
            incorrect += 1

        raw_returns.append(
            raw_return
        )

        net_returns.append(
            net_return
        )

        if profitable:
            winning += 1
        else:
            losing += 1

    actionable_count = len(actionable)

    if actionable_count:
        direction_accuracy = (
            correct / actionable_count
        )

        average_return = (
            sum(raw_returns)
            / actionable_count
        )

        total_return = sum(
            raw_returns
        )

        average_net_return = (
            sum(net_returns)
            / actionable_count
        )

        total_net_return = sum(
            net_returns
        )

        win_rate = (
            winning / actionable_count
        )

    else:
        direction_accuracy = 0.0
        average_return = 0.0
        total_return = 0.0
        average_net_return = 0.0
        total_net_return = 0.0
        win_rate = 0.0

    gross_profit = sum(
        value
        for value in net_returns
        if value > 0.0
    )

    gross_loss = abs(
        sum(
            value
            for value in net_returns
            if value < 0.0
        )
    )

    if gross_loss > 0.0:
        profit_factor = (
            gross_profit
            / gross_loss
        )
    elif gross_profit > 0.0:
        profit_factor = float("inf")
    else:
        profit_factor = 0.0

    return StrategyEvaluation(
        name=name,
        prediction_count=prediction_count,
        actionable_predictions=(
            actionable_count
        ),
        correct_direction_predictions=correct,
        incorrect_direction_predictions=incorrect,
        direction_accuracy=direction_accuracy,
        average_return_percent=(
            average_return * 100.0
        ),
        total_return_percent=(
            total_return * 100.0
        ),
        average_net_return_percent=(
            average_net_return * 100.0
        ),
        total_net_return_percent=(
            total_net_return * 100.0
        ),
        winning_predictions=winning,
        losing_predictions=losing,
        win_rate=win_rate,
        profit_factor=profit_factor,
    )


def _feature_row_for_prediction(
    row: MLDatasetRow,
) -> MLDatasetRow:

    return MLDatasetRow(
        timestamp=row.timestamp,
        features=dict(row.features),
        target_return=0.0,
        target_direction="HOLD",
    )


def _train_fold_model(
    dataset: MLDataset,
) -> MLModel:

    return train_ml_model(
        dataset,
        learning_rate=0.01,
        epochs=500,
    )


def _build_adaptive_ensemble_state() -> (
    AdaptiveEnsembleState
):
    return AdaptiveEnsembleState(
        window=50,
        base_rule_weight=0.4,
        base_ml_weight=0.6,
        minimum_weight=0.20,
        maximum_weight=0.80,
        minimum_observations=5,
        recency_decay=0.97,
    )

def build_latest_ensemble_signal(
    rows: Sequence[MarketData],
    *,
    horizon: int = 5,
):
    """
    Build the latest trading ensemble from historical market data.

    The model is trained only on historical rows preceding the
    latest prediction row, preventing the latest observation from
    leaking into model training.
    """
    ordered_rows = sorted(
        rows,
        key=lambda row: row.timestamp,
    )

    dataset = build_ml_dataset(
        ordered_rows,
        horizon=horizon,
    )

    if len(dataset.rows) < 20:
        raise ValueError(
            "Not enough market data to build "
            "the latest trading decision."
        )

    prediction_row = dataset.rows[-1]

    training_dataset = MLDataset(
        rows=tuple(dataset.rows[:-1]),
        feature_names=dataset.feature_names,
    )

    if len(training_dataset.rows) < 20:
        raise ValueError(
            "Not enough historical data to train "
            "the trading model."
        )

    model = _train_fold_model(
        training_dataset
    )

    rule_result = _rule_signal_from_features(
        prediction_row
    )

    ml_prediction = predict_ml_model(
        model,
        _feature_row_for_prediction(
            prediction_row
        ),
    )

    adaptive_state = (
        _build_adaptive_ensemble_state()
    )

    return combine_signals(
        rule_result,
        ml_prediction,
        adaptive_weights=(
            adaptive_state.weights()
        ),
    )

def compare_walk_forward_baselines(
    rows: Sequence[MarketData],
    *,
    symbol: str,
    horizon: int = 5,
    initial_training_fraction: float = 0.60,
    folds: int = 4,
    gap_rows: int | None = None,
    validation_window: int | None = None,
    transaction_cost_percent: float = 0.10,
    slippage_percent: float = 0.05,
) -> BaselineComparisonResult:

    if horizon <= 0:
        raise ValueError(
            "Horizon must be greater than zero."
        )

    if not 0 < initial_training_fraction < 1:
        raise ValueError(
            "Initial training fraction "
            "must be between 0 and 1."
        )

    if folds <= 0:
        raise ValueError(
            "Number of folds must be greater than zero."
        )

    if transaction_cost_percent < 0:
        raise ValueError(
            "Transaction cost cannot be negative."
        )

    if slippage_percent < 0:
        raise ValueError(
            "Slippage cannot be negative."
        )

    ordered_rows = sorted(
        rows,
        key=lambda row: row.timestamp,
    )

    dataset = build_ml_dataset(
        ordered_rows,
        horizon=horizon,
    )

    if len(dataset.rows) < 20:
        raise ValueError(
            "Dataset must contain at least 20 rows "
            "for baseline comparison."
        )

    if gap_rows is None:
        gap_rows = horizon

    if gap_rows < 0:
        raise ValueError(
            "Gap rows cannot be negative."
        )

    total_rows = len(dataset.rows)

    initial_training_rows = max(
        5,
        int(
            total_rows
            * initial_training_fraction
        ),
    )

    if initial_training_rows >= total_rows:
        raise ValueError(
            "Initial training window leaves "
            "no validation data."
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

    strategy_names = (
        "always_buy",
        "always_sell",
        "rule",
        "ml",
        "ensemble",
        "adaptive_ensemble",
    )

    predictions: dict[
        str,
        list[tuple[str, float]],
    ] = {
        name: []
        for name in strategy_names
    }

    training_end = initial_training_rows
    fold_number = 0

    total_training_rows = 0
    total_validation_rows = 0

    while (
        fold_number < folds
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

        training_dataset = MLDataset(
            rows=tuple(
                dataset.rows[:training_end]
            ),
            feature_names=(
                dataset.feature_names
            ),
        )

        validation_rows = dataset.rows[
            validation_start:validation_end
        ]

        if (
            not training_dataset.rows
            or not validation_rows
        ):
            break

        model = _train_fold_model(
            training_dataset
        )

        adaptive_state = (
            _build_adaptive_ensemble_state()
        )

        for row in validation_rows:
            rule_result = (
                _rule_signal_from_features(row)
            )

            ml_prediction = predict_ml_model(
                model,
                _feature_row_for_prediction(row),
            )

            fixed_ensemble_result = (
                combine_signals(
                    rule_result,
                    ml_prediction,
                    rule_weight=0.4,
                    ml_weight=0.6,
                )
            )

            adaptive_weights = (
                adaptive_state.weights()
            )

            adaptive_ensemble_result = (
                combine_signals(
                    rule_result,
                    ml_prediction,
                    adaptive_weights=(
                        adaptive_weights
                    ),
                )
            )

            target_return = row.target_return

            predictions["always_buy"].append(
                (
                    "BUY",
                    target_return,
                )
            )

            predictions["always_sell"].append(
                (
                    "SELL",
                    target_return,
                )
            )

            predictions["rule"].append(
                (
                    rule_result.signal,
                    target_return,
                )
            )

            predictions["ml"].append(
                (
                    ml_prediction.direction,
                    target_return,
                )
            )

            predictions["ensemble"].append(
                (
                    fixed_ensemble_result.signal,
                    target_return,
                )
            )

            predictions[
                "adaptive_ensemble"
            ].append(
                (
                    adaptive_ensemble_result.signal,
                    target_return,
                )
            )

            actual_direction = (
                _target_direction(
                    target_return
                )
            )

            adaptive_state.observe(
                rule_signal=rule_result.signal,
                ml_signal=ml_prediction.direction,
                actual_direction=(
                    actual_direction
                ),
                realized_return_percent=(
                    target_return * 100.0
                ),
            )

        total_training_rows += (
            len(training_dataset.rows)
        )

        total_validation_rows += (
            len(validation_rows)
        )

        training_end = validation_end
        fold_number += 1

    if total_validation_rows == 0:
        raise ValueError(
            "Unable to construct baseline "
            "comparison folds."
        )

    evaluations = tuple(
        _calculate_strategy_evaluation(
            name,
            predictions[name],
            transaction_cost_percent=(
                transaction_cost_percent
            ),
            slippage_percent=slippage_percent,
        )
        for name in strategy_names
    )

    return BaselineComparisonResult(
        symbol=symbol.upper(),
        horizon=horizon,
        dataset_rows=total_rows,
        total_training_rows=(
            total_training_rows
        ),
        total_validation_rows=(
            total_validation_rows
        ),
        folds=fold_number,
        gap_rows=gap_rows,
        strategies=evaluations,
    )