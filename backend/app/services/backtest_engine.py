from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from app.models.market_data import MarketData
from app.services.adaptive_ensemble import AdaptiveEnsembleState
from app.services.ensemble_signal import combine_signals
from app.services.feature_engineering import (
    MarketFeature,
    build_market_features,
)
from app.services.ml_dataset import (
    FEATURE_NAMES,
    MLDataset,
    MLDatasetRow,
)
from app.services.ml_model import (
    MLPrediction,
    predict_ml_model,
    train_ml_model,
)
from app.services.signal_engine import (
    SignalResult,
    generate_signal,
)


# ============================================================================
# Result models
# ============================================================================


@dataclass(frozen=True)
class BacktestTrade:
    timestamp: object
    signal: str
    confidence: float

    entry_price: float
    exit_price: float

    return_percent: float
    profitable: bool
    position_return_percent: float

    equity_after: float


@dataclass(frozen=True)
class BacktestResult:
    symbol: str

    total_rows: int
    evaluated_rows: int

    actionable_trades: int
    winning_trades: int
    losing_trades: int

    win_rate: float
    average_return_percent: float
    total_return_percent: float

    starting_capital: float
    ending_capital: float
    net_profit: float
    net_return_percent: float

    gross_profit: float
    gross_loss: float
    profit_factor: float

    average_winning_trade_percent: float
    average_losing_trade_percent: float

    maximum_drawdown_percent: float

    buy_and_hold_return_percent: float
    strategy_outperformance_percent: float

    trades: tuple[BacktestTrade, ...]


# ============================================================================
# Backtest configuration
# ============================================================================


# The minimum amount of historical information required before the
# backtester starts making predictions.
_MINIMUM_HISTORY = 60

# Do not retrain the neural network on every bar.
#
# Training is considerably more expensive than prediction.  Retraining
# periodically gives us a genuine walk-forward model while keeping the
# backtest fast.
_RETRAIN_INTERVAL = 25

# Keep the ML training reasonably lightweight for repeated backtesting.
#
# The ML model itself performs deterministic training, so reducing epochs
# here does not introduce uncontrolled randomness.
_TRAINING_EPOCHS = 120

# Five-bar prediction horizon is the existing AegisQuant default.
_DEFAULT_HORIZON = 5


# ============================================================================
# Validation / helpers
# ============================================================================


def _empty_result(
    *,
    symbol: str,
    starting_capital: float,
) -> BacktestResult:
    return BacktestResult(
        symbol=symbol.upper(),
        total_rows=0,
        evaluated_rows=0,
        actionable_trades=0,
        winning_trades=0,
        losing_trades=0,
        win_rate=0.0,
        average_return_percent=0.0,
        total_return_percent=0.0,
        starting_capital=starting_capital,
        ending_capital=starting_capital,
        net_profit=0.0,
        net_return_percent=0.0,
        gross_profit=0.0,
        gross_loss=0.0,
        profit_factor=0.0,
        average_winning_trade_percent=0.0,
        average_losing_trade_percent=0.0,
        maximum_drawdown_percent=0.0,
        buy_and_hold_return_percent=0.0,
        strategy_outperformance_percent=0.0,
        trades=(),
    )


def _feature_to_values(
    feature: MarketFeature,
) -> dict[str, float] | None:
    """
    Convert a precomputed MarketFeature into the exact feature dictionary
    expected by MLDatasetRow.

    Returning None means that the technical indicators for this bar are not
    sufficiently populated yet.
    """

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


def _feature_signal(
    feature: MarketFeature,
) -> SignalResult:
    """
    Generate the rule-based signal directly from an already computed
    MarketFeature.

    This avoids recalculating SMA, EMA, RSI and MACD for every backtest bar.
    """

    return generate_signal(
        rsi_14=feature.rsi_14,
        macd=feature.macd,
        macd_signal=feature.macd_signal,
        sma_20=feature.sma_20,
        ema_20=feature.ema_20,
        close=feature.close,
    )


def _build_training_dataset(
    rows: Sequence[MarketData],
    features: Sequence[MarketFeature],
    *,
    end_index: int,
    horizon: int,
) -> MLDataset:
    """
    Build a strictly historical training dataset.

    IMPORTANT:

    For a prediction made at `end_index`, only observations whose COMPLETE
    future target is already known are included.

    Therefore:

        target_index <= end_index

    This prevents future-data leakage.
    """

    dataset_rows: list[MLDatasetRow] = []

    maximum_target_index = end_index

    maximum_feature_index = (
        maximum_target_index - horizon
    )

    if maximum_feature_index < 0:
        return MLDataset(
            rows=(),
            feature_names=FEATURE_NAMES,
        )

    upper_index = min(
        maximum_feature_index,
        len(rows) - horizon - 1,
        len(features) - 1,
    )

    for index in range(upper_index + 1):
        feature_values = _feature_to_values(
            features[index]
        )

        if feature_values is None:
            continue

        current_close = float(
            rows[index].close
        )

        future_close = float(
            rows[index + horizon].close
        )

        if current_close <= 0:
            continue

        target_return = (
            (future_close - current_close)
            / current_close
        ) * 100.0

        if target_return > 0:
            target_direction = "BUY"
        elif target_return < 0:
            target_direction = "SELL"
        else:
            target_direction = "HOLD"

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


def _train_walk_forward_model(
    rows: Sequence[MarketData],
    features: Sequence[MarketFeature],
    *,
    end_index: int,
    horizon: int,
):
    """
    Train an ML model using ONLY information available at end_index.

    Returns None when there is insufficient usable historical data.
    """

    dataset = _build_training_dataset(
        rows,
        features,
        end_index=end_index,
        horizon=horizon,
    )

    # Require a meaningful training sample.
    if len(dataset.rows) < 40:
        return None

    return train_ml_model(
        dataset,
        learning_rate=0.01,
        epochs=_TRAINING_EPOCHS,
        l2_regularization=1e-4,
    )


def _actual_direction(
    return_percent: float,
) -> str:
    """
    Convert a realized trade return into the direction that actually won.

    This is used only to update the adaptive ensemble AFTER the trade has
    completed.
    """

    if return_percent > 0:
        return "BUY"

    if return_percent < 0:
        return "SELL"

    return "HOLD"


def _calculate_trade_return(
    *,
    signal: str,
    raw_entry: float,
    raw_exit: float,
    slippage_percent: float,
    transaction_cost_percent: float,
) -> tuple[float, float, float]:
    """
    Return:

        entry_price
        exit_price
        net_return_percent
    """

    slippage = (
        float(slippage_percent)
        / 100.0
    )

    transaction_cost = (
        2.0
        * float(transaction_cost_percent)
        / 100.0
    )

    if signal == "BUY":
        entry_price = (
            raw_entry
            * (1.0 + slippage)
        )

        exit_price = (
            raw_exit
            * (1.0 - slippage)
        )

        gross_return = (
            exit_price - entry_price
        ) / entry_price

    elif signal == "SELL":
        entry_price = (
            raw_entry
            * (1.0 - slippage)
        )

        exit_price = (
            raw_exit
            * (1.0 + slippage)
        )

        gross_return = (
            entry_price - exit_price
        ) / entry_price

    else:
        raise ValueError(
            "Trade signal must be BUY or SELL."
        )

    net_return = (
        gross_return
        - transaction_cost
    )

    return (
        entry_price,
        exit_price,
        net_return * 100.0,
    )


# ============================================================================
# Main backtest
# ============================================================================


def backtest_market_data(
    rows: Sequence[MarketData],
    *,
    symbol: str,
    horizon: int = _DEFAULT_HORIZON,
    starting_capital: float = 1_000_000.0,
    transaction_cost_percent: float = 0.10,
    slippage_percent: float = 0.05,
) -> BacktestResult:
    """
    Walk-forward backtest of the AegisQuant rule + ML ensemble.

    Architecture:

        historical prices
              |
              v
        precompute features ONCE
              |
              v
        walk-forward evaluation
              |
              +--> rule engine
              |
              +--> ML model
              |
              +--> adaptive ensemble
              |
              v
        next-bar execution
              |
              v
        realized result
              |
              v
        adaptive ensemble update

    No future observations are used to generate a prediction.

    The neural model is retrained periodically rather than on every bar.
    """

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    if horizon <= 0:
        raise ValueError(
            "Horizon must be greater than zero."
        )

    if starting_capital <= 0:
        raise ValueError(
            "Starting capital must be greater than zero."
        )

    if transaction_cost_percent < 0:
        raise ValueError(
            "Transaction cost cannot be negative."
        )

    if slippage_percent < 0:
        raise ValueError(
            "Slippage cannot be negative."
        )

    # ------------------------------------------------------------------
    # Sort once
    # ------------------------------------------------------------------

    ordered_rows = sorted(
        rows,
        key=lambda row: row.timestamp,
    )

    total_rows = len(ordered_rows)

    if total_rows == 0:
        return _empty_result(
            symbol=symbol,
            starting_capital=starting_capital,
        )

    # ------------------------------------------------------------------
    # PRECOMPUTE FEATURES ONCE
    #
    # This is the major performance correction.
    #
    # The previous implementation repeatedly reconstructed the complete
    # close-price history and recalculated SMA/EMA/RSI/MACD for every bar.
    #
    # That is unnecessary.
    # ------------------------------------------------------------------

    features = build_market_features(
        ordered_rows
    )

    # ------------------------------------------------------------------
    # Determine evaluation range
    # ------------------------------------------------------------------

    first_signal_index = _MINIMUM_HISTORY - 1

    last_signal_index = (
        total_rows
        - horizon
        - 2
    )

    if last_signal_index < first_signal_index:
        return BacktestResult(
            symbol=symbol.upper(),
            total_rows=total_rows,
            evaluated_rows=0,
            actionable_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            average_return_percent=0.0,
            total_return_percent=0.0,
            starting_capital=starting_capital,
            ending_capital=starting_capital,
            net_profit=0.0,
            net_return_percent=0.0,
            gross_profit=0.0,
            gross_loss=0.0,
            profit_factor=0.0,
            average_winning_trade_percent=0.0,
            average_losing_trade_percent=0.0,
            maximum_drawdown_percent=0.0,
            buy_and_hold_return_percent=0.0,
            strategy_outperformance_percent=0.0,
            trades=(),
        )

    evaluated_rows = (
        last_signal_index
        - first_signal_index
        + 1
    )

    # ------------------------------------------------------------------
    # Adaptive ensemble
    # ------------------------------------------------------------------

    adaptive_state = AdaptiveEnsembleState(
        window=50,
        base_rule_weight=0.40,
        base_ml_weight=0.60,
        minimum_weight=0.20,
        maximum_weight=0.80,
        minimum_observations=5,
        recency_decay=0.97,
    )

    # ------------------------------------------------------------------
    # ML model state
    # ------------------------------------------------------------------

    ml_model = None
    last_training_index = -10_000

    # ------------------------------------------------------------------
    # Equity state
    # ------------------------------------------------------------------

    equity = float(starting_capital)

    peak_equity = equity
    maximum_drawdown_percent = 0.0

    trades: list[BacktestTrade] = []

    # ------------------------------------------------------------------
    # Walk forward
    # ------------------------------------------------------------------

    signal_index = first_signal_index

    while signal_index <= last_signal_index:

        # --------------------------------------------------------------
        # Feature available at this exact historical point.
        # --------------------------------------------------------------

        feature = features[signal_index]

        # --------------------------------------------------------------
        # Rule-based signal
        # --------------------------------------------------------------

        rule_signal = _feature_signal(
            feature
        )

        # --------------------------------------------------------------
        # Train / retrain ML model periodically.
        #
        # The training dataset ends at signal_index, and only observations
        # whose future target is already known are included.
        # --------------------------------------------------------------

        should_train = (
            ml_model is None
            or (
                signal_index
                - last_training_index
                >= _RETRAIN_INTERVAL
            )
        )

        if should_train:
            ml_model = _train_walk_forward_model(
                ordered_rows,
                features,
                end_index=signal_index,
                horizon=horizon,
            )

            last_training_index = signal_index

        # --------------------------------------------------------------
        # If there is not enough historical data for ML yet, use a
        # neutral ML prediction. This keeps the ensemble operational
        # during the early walk-forward period.
        # --------------------------------------------------------------

        if ml_model is not None:
            feature_values = _feature_to_values(
                feature
            )

            if feature_values is None:
                ml_prediction = MLPrediction(
                    direction="HOLD",
                    confidence=0.50,
                    expected_return_percent=0.0,
                )

            else:
                ml_row = MLDatasetRow(
                    timestamp=feature.timestamp,
                    features=feature_values,
                    target_return=0.0,
                    target_direction="HOLD",
                )

                ml_prediction = predict_ml_model(
                    ml_model,
                    ml_row,
                )

        else:
            ml_prediction = MLPrediction(
                direction="HOLD",
                confidence=0.50,
                expected_return_percent=0.0,
            )

        # --------------------------------------------------------------
        # Adaptive ensemble weights.
        # --------------------------------------------------------------

        adaptive_weights = (
            adaptive_state.weights()
        )

        ensemble = combine_signals(
            rule_signal,
            ml_prediction,
            adaptive_weights=adaptive_weights,
        )

        signal = ensemble.signal

        # --------------------------------------------------------------
        # HOLD means no position.
        #
        # We advance exactly one bar so that future opportunities are
        # still evaluated.
        # --------------------------------------------------------------

        if signal == "HOLD":
            signal_index += 1
            continue

        # --------------------------------------------------------------
        # Execution occurs on the NEXT bar.
        # --------------------------------------------------------------

        entry_index = signal_index + 1
        exit_index = entry_index + horizon

        if exit_index >= total_rows:
            break

        entry_row = ordered_rows[
            entry_index
        ]

        exit_row = ordered_rows[
            exit_index
        ]

        raw_entry = float(
            entry_row.open
        )

        raw_exit = float(
            exit_row.open
        )

        if raw_entry <= 0 or raw_exit <= 0:
            signal_index += 1
            continue

        # --------------------------------------------------------------
        # Calculate realistic trade return.
        # --------------------------------------------------------------

        (
            entry_price,
            exit_price,
            return_percent,
        ) = _calculate_trade_return(
            signal=signal,
            raw_entry=raw_entry,
            raw_exit=raw_exit,
            slippage_percent=slippage_percent,
            transaction_cost_percent=transaction_cost_percent,
        )

        net_position_return = (
            return_percent / 100.0
        )

        profitable = (
            return_percent > 0
        )

        # --------------------------------------------------------------
        # Update equity.
        # --------------------------------------------------------------

        equity += (
            equity
            * net_position_return
        )

        peak_equity = max(
            peak_equity,
            equity,
        )

        if peak_equity > 0:
            drawdown = (
                (peak_equity - equity)
                / peak_equity
            ) * 100.0

            maximum_drawdown_percent = max(
                maximum_drawdown_percent,
                drawdown,
            )

        # --------------------------------------------------------------
        # Record trade.
        # --------------------------------------------------------------

        trades.append(
            BacktestTrade(
                timestamp=entry_row.timestamp,
                signal=signal,
                confidence=ensemble.confidence,
                entry_price=entry_price,
                exit_price=exit_price,
                return_percent=return_percent,
                profitable=profitable,
                position_return_percent=return_percent,
                equity_after=equity,
            )
        )

        # --------------------------------------------------------------
        # ONLY NOW do we give the adaptive ensemble the outcome.
        #
        # This is important:
        #
        # The ensemble cannot know the outcome when making the decision.
        # It receives the result only after the trade has completed.
        # --------------------------------------------------------------

        actual_direction = _actual_direction(
            return_percent
        )

        adaptive_state.observe(
            rule_signal=rule_signal.signal,
            ml_signal=ml_prediction.direction,
            actual_direction=actual_direction,
            realized_return_percent=return_percent,
        )

        # --------------------------------------------------------------
        # No overlapping positions.
        #
        # The next decision occurs after the exit bar.
        # --------------------------------------------------------------

        signal_index = (
            exit_index + 1
        )

    # =========================================================================
    # Performance calculations
    # =========================================================================

    actionable_trades = len(
        trades
    )

    winning_trades = sum(
        trade.profitable
        for trade in trades
    )

    losing_trades = (
        actionable_trades
        - winning_trades
    )

    if actionable_trades > 0:

        win_rate = (
            winning_trades
            / actionable_trades
        )

        average_return_percent = (
            sum(
                trade.return_percent
                for trade in trades
            )
            / actionable_trades
        )

        # This field represents the sum of individual trade returns.
        # The actual compounded result is net_return_percent below.
        total_return_percent = sum(
            trade.return_percent
            for trade in trades
        )

    else:
        win_rate = 0.0
        average_return_percent = 0.0
        total_return_percent = 0.0

    # =========================================================================
    # Gross profit / gross loss
    # =========================================================================

    gross_profit = 0.0
    gross_loss = 0.0

    previous_equity = (
        starting_capital
    )

    for trade in trades:

        change = (
            trade.equity_after
            - previous_equity
        )

        if change > 0:
            gross_profit += change

        elif change < 0:
            gross_loss += abs(
                change
            )

        previous_equity = (
            trade.equity_after
        )

    # =========================================================================
    # Profit factor
    # =========================================================================

    if gross_loss > 0:
        profit_factor = (
            gross_profit
            / gross_loss
        )

    elif gross_profit > 0:
        profit_factor = float("inf")

    else:
        profit_factor = 0.0

    # =========================================================================
    # Average winning / losing trades
    # =========================================================================

    winning_returns = [
        trade.return_percent
        for trade in trades
        if trade.profitable
    ]

    losing_returns = [
        trade.return_percent
        for trade in trades
        if not trade.profitable
    ]

    if winning_returns:
        average_winning_trade_percent = (
            sum(winning_returns)
            / len(winning_returns)
        )
    else:
        average_winning_trade_percent = 0.0

    if losing_returns:
        average_losing_trade_percent = (
            sum(losing_returns)
            / len(losing_returns)
        )
    else:
        average_losing_trade_percent = 0.0

    # =========================================================================
    # Final equity
    # =========================================================================

    ending_capital = equity

    net_profit = (
        ending_capital
        - starting_capital
    )

    net_return_percent = (
        net_profit
        / starting_capital
    ) * 100.0

    # =========================================================================
    # Buy and hold benchmark
    # =========================================================================

    first_close = float(
        ordered_rows[0].close
    )

    last_close = float(
        ordered_rows[-1].close
    )

    if first_close > 0:

        buy_and_hold_return_percent = (
            (
                last_close
                - first_close
            )
            / first_close
        ) * 100.0

    else:
        buy_and_hold_return_percent = 0.0

    strategy_outperformance_percent = (
        net_return_percent
        - buy_and_hold_return_percent
    )

    # =========================================================================
    # Final result
    # =========================================================================

    return BacktestResult(
        symbol=symbol.upper(),
        total_rows=total_rows,
        evaluated_rows=evaluated_rows,
        actionable_trades=actionable_trades,
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        win_rate=win_rate,
        average_return_percent=average_return_percent,
        total_return_percent=total_return_percent,
        starting_capital=starting_capital,
        ending_capital=ending_capital,
        net_profit=net_profit,
        net_return_percent=net_return_percent,
        gross_profit=gross_profit,
        gross_loss=gross_loss,
        profit_factor=profit_factor,
        average_winning_trade_percent=(
            average_winning_trade_percent
        ),
        average_losing_trade_percent=(
            average_losing_trade_percent
        ),
        maximum_drawdown_percent=(
            maximum_drawdown_percent
        ),
        buy_and_hold_return_percent=(
            buy_and_hold_return_percent
        ),
        strategy_outperformance_percent=(
            strategy_outperformance_percent
        ),
        trades=tuple(trades),
    )