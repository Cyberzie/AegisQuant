from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from app.models.market_data import MarketData
from app.services.adaptive_ensemble import AdaptiveEnsembleState
from app.services.ensemble_signal import combine_signals
from app.services.feature_engineering import build_market_features
from app.services.ml_dataset import (
    MLDatasetRow,
    build_ml_dataset,
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
from app.services.technical_indicators import (
    ema,
    macd,
    rsi,
    sma,
)


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


MINIMUM_HISTORY = 26
MINIMUM_ML_ROWS = 20

DEFAULT_LEARNING_RATE = 0.01
DEFAULT_EPOCHS = 500

RULE_WEIGHT = 0.45
ML_WEIGHT = 0.55

MINIMUM_ACTIONABLE_CONFIDENCE = 0.55


def _calculate_rule_result(
    closes: list[float],
) -> SignalResult:
    sma_values = sma(closes, 20)
    ema_values = ema(closes, 20)
    rsi_values = rsi(closes, 14)
    macd_values = macd(closes)

    index = len(closes) - 1

    return generate_signal(
        rsi_14=rsi_values[index],
        macd=macd_values["macd"][index],
        macd_signal=macd_values["signal"][index],
        sma_20=sma_values[index],
        ema_20=ema_values[index],
        close=closes[index],
    )


def _train_ml_model_for_history(
    rows: Sequence[MarketData],
    *,
    horizon: int,
):
    dataset = build_ml_dataset(
        rows,
        horizon=horizon,
    )

    if len(dataset.rows) < MINIMUM_ML_ROWS:
        return None

    return train_ml_model(
        dataset,
        learning_rate=DEFAULT_LEARNING_RATE,
        epochs=DEFAULT_EPOCHS,
    )


def _build_prediction_row(
    rows: Sequence[MarketData],
) -> MLDatasetRow | None:
    features = build_market_features(rows)

    if not features:
        return None

    latest_feature = features[-1]

    feature_values = {
        "close": latest_feature.close,
        "volume": latest_feature.volume,
        "return_1": latest_feature.return_1,
        "return_5": latest_feature.return_5,
        "volatility_10": latest_feature.volatility_10,
        "sma_20": latest_feature.sma_20,
        "ema_20": latest_feature.ema_20,
        "rsi_14": latest_feature.rsi_14,
        "macd": latest_feature.macd,
        "macd_signal": latest_feature.macd_signal,
        "macd_histogram": latest_feature.macd_histogram,
        "bollinger_middle": latest_feature.bollinger_middle,
        "bollinger_upper": latest_feature.bollinger_upper,
        "bollinger_lower": latest_feature.bollinger_lower,
        "atr_14": latest_feature.atr_14,
    }

    if any(
        value is None
        for value in feature_values.values()
    ):
        return None

    return MLDatasetRow(
        timestamp=latest_feature.timestamp,
        features={
            name: float(value)
            for name, value in feature_values.items()
        },
        target_return=0.0,
        target_direction="HOLD",
    )


def _calculate_ml_prediction(
    rows: Sequence[MarketData],
    *,
    horizon: int,
) -> MLPrediction | None:
    model = _train_ml_model_for_history(
        rows,
        horizon=horizon,
    )

    if model is None:
        return None

    prediction_row = _build_prediction_row(rows)

    if prediction_row is None:
        return None

    return predict_ml_model(
        model,
        prediction_row,
    )


def _calculate_ensemble_signal(
    rows: Sequence[MarketData],
    *,
    horizon: int,
    adaptive_state: AdaptiveEnsembleState,
) -> tuple[str, float, str, str]:

    closes = [
        float(row.close)
        for row in rows
    ]

    rule_result = _calculate_rule_result(
        closes
    )

    ml_prediction = _calculate_ml_prediction(
        rows,
        horizon=horizon,
    )

    if ml_prediction is None:
        return (
            rule_result.signal,
            rule_result.confidence,
            rule_result.signal,
            "HOLD",
        )

    adaptive_weights = adaptive_state.weights()

    ensemble = combine_signals(
        rule_result,
        ml_prediction,
        adaptive_weights=adaptive_weights,
    )

    return (
        ensemble.signal,
        ensemble.confidence,
        ensemble.rule_signal,
        ensemble.ml_signal,
    )


def _calculate_position_return(
    signal: str,
    entry_price: float,
    exit_price: float,
    *,
    transaction_cost_percent: float,
    slippage_percent: float,
) -> float:

    if signal == "BUY":
        raw_return = (
            exit_price - entry_price
        ) / entry_price

    elif signal == "SELL":
        raw_return = (
            entry_price - exit_price
        ) / entry_price

    else:
        return 0.0

    transaction_cost = (
        2.0
        * transaction_cost_percent
        / 100.0
    )

    slippage_cost = (
        2.0
        * slippage_percent
        / 100.0
    )

    total_cost = (
        transaction_cost
        + slippage_cost
    )

    return raw_return - total_cost


def _calculate_drawdown(
    equity: float,
    peak_equity: float,
) -> float:

    if peak_equity <= 0:
        return 0.0

    return (
        (peak_equity - equity)
        / peak_equity
    ) * 100.0


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


def backtest_market_data(
    rows: Sequence[MarketData],
    *,
    symbol: str,
    horizon: int = 5,
    starting_capital: float = 1_000_000.0,
    transaction_cost_percent: float = 0.10,
    slippage_percent: float = 0.05,
) -> BacktestResult:

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

    if total_rows < MINIMUM_HISTORY:
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

    adaptive_state = AdaptiveEnsembleState(
        window=50,
        base_rule_weight=RULE_WEIGHT,
        base_ml_weight=ML_WEIGHT,
        minimum_weight=0.20,
        maximum_weight=0.80,
        minimum_observations=5,
        recency_decay=0.97,
    )

    trades: list[BacktestTrade] = []

    equity = float(starting_capital)
    peak_equity = equity
    maximum_drawdown_percent = 0.0

    evaluated_rows = 0

    index = MINIMUM_HISTORY - 1

    while index < total_rows - 1:

        signal_index = index

        historical_rows = ordered_rows[
            : signal_index + 1
        ]

        (
            signal,
            confidence,
            rule_signal,
            ml_signal,
        ) = _calculate_ensemble_signal(
            historical_rows,
            horizon=horizon,
            adaptive_state=adaptive_state,
        )

        evaluated_rows += 1

        # Update adaptive performance only after the future outcome
        # becomes available. This keeps the weighting walk-forward safe.
        outcome_index = (
            signal_index + horizon
        )

        if outcome_index < total_rows:

            signal_price = float(
                ordered_rows[
                    signal_index
                ].close
            )

            outcome_price = float(
                ordered_rows[
                    outcome_index
                ].close
            )

            if (
                signal_price > 0
                and outcome_price > 0
            ):

                future_return_percent = (
                    (
                        outcome_price
                        - signal_price
                    )
                    / signal_price
                ) * 100.0

                if future_return_percent > 0:
                    actual_direction = "BUY"

                elif future_return_percent < 0:
                    actual_direction = "SELL"

                else:
                    actual_direction = "HOLD"

                adaptive_state.observe(
                    rule_signal=rule_signal,
                    ml_signal=ml_signal,
                    actual_direction=(
                        actual_direction
                    ),
                    realized_return_percent=(
                        future_return_percent
                    ),
                )

        # Do not enter weak or ambiguous signals.
        if (
            signal not in {"BUY", "SELL"}
            or confidence
            < MINIMUM_ACTIONABLE_CONFIDENCE
        ):
            index += 1
            continue

        # Enter on the bar immediately after the
        # signal bar. This prevents look-ahead bias.
        entry_index = signal_index + 1

        # Position remains open for exactly `horizon`
        # bars after entry.
        exit_index = (
            entry_index + horizon
        )

        if exit_index >= total_rows:
            break

        entry_price = float(
            ordered_rows[
                entry_index
            ].close
        )

        exit_price = float(
            ordered_rows[
                exit_index
            ].close
        )

        if (
            entry_price <= 0
            or exit_price <= 0
        ):
            index += 1
            continue

        net_position_return = (
            _calculate_position_return(
                signal,
                entry_price,
                exit_price,
                transaction_cost_percent=(
                    transaction_cost_percent
                ),
                slippage_percent=(
                    slippage_percent
                ),
            )
        )

        return_percent = (
            net_position_return * 100.0
        )

        profitable = return_percent > 0

        trade_profit = (
            equity
            * net_position_return
        )

        equity += trade_profit

        if equity > peak_equity:
            peak_equity = equity

        drawdown_percent = (
            _calculate_drawdown(
                equity,
                peak_equity,
            )
        )

        maximum_drawdown_percent = max(
            maximum_drawdown_percent,
            drawdown_percent,
        )

        trades.append(
            BacktestTrade(
                timestamp=ordered_rows[
                    entry_index
                ].timestamp,
                signal=signal,
                confidence=confidence,
                entry_price=entry_price,
                exit_price=exit_price,
                return_percent=return_percent,
                profitable=profitable,
                position_return_percent=(
                    net_position_return * 100.0
                ),
                equity_after=equity,
            )
        )

        # Never permit overlapping positions.
        index = exit_index + 1

    actionable_trades = len(trades)

    winning_trades = sum(
        1
        for trade in trades
        if trade.profitable
    )

    losing_trades = (
        actionable_trades
        - winning_trades
    )

    if actionable_trades:

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

        total_return_percent = (
            (
                equity
                / starting_capital
            )
            - 1.0
        ) * 100.0

    else:

        win_rate = 0.0
        average_return_percent = 0.0
        total_return_percent = 0.0

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
            gross_loss += abs(change)

        previous_equity = (
            trade.equity_after
        )

    if gross_loss > 0:

        profit_factor = (
            gross_profit
            / gross_loss
        )

    elif gross_profit > 0:

        profit_factor = float("inf")

    else:

        profit_factor = 0.0

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

    average_winning_trade_percent = (
        sum(winning_returns)
        / len(winning_returns)
        if winning_returns
        else 0.0
    )

    average_losing_trade_percent = (
        sum(losing_returns)
        / len(losing_returns)
        if losing_returns
        else 0.0
    )

    ending_capital = equity

    net_profit = (
        ending_capital
        - starting_capital
    )

    net_return_percent = (
        net_profit
        / starting_capital
    ) * 100.0

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

    return BacktestResult(
        symbol=symbol.upper(),
        total_rows=total_rows,
        evaluated_rows=evaluated_rows,
        actionable_trades=actionable_trades,
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        win_rate=win_rate,
        average_return_percent=(
            average_return_percent
        ),
        total_return_percent=(
            total_return_percent
        ),
        starting_capital=starting_capital,
        ending_capital=ending_capital,
        net_profit=net_profit,
        net_return_percent=(
            net_return_percent
        ),
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