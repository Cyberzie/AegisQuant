from __future__ import annotations

from dataclasses import dataclass
from math import sqrt


@dataclass(frozen=True)
class TradeRecord:
    symbol: str
    signal: str
    quantity: float
    execution_price: float
    realized_pnl: float


@dataclass(frozen=True)
class PerformanceSnapshot:
    starting_capital: float
    current_equity: float
    realized_pnl: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate_percent: float
    profit_factor: float
    cumulative_return_percent: float
    maximum_drawdown_percent: float
    sharpe_ratio: float


class PerformanceTracker:
    """Track realized trading performance and equity history."""

    def __init__(self, starting_capital: float) -> None:
        if starting_capital <= 0:
            raise ValueError(
                "Starting capital must be greater than zero."
            )

        self.starting_capital = starting_capital
        self._trades: list[TradeRecord] = []
        self._equity_history: list[float] = [
            starting_capital
        ]

    @property
    def trades(self) -> tuple[TradeRecord, ...]:
        return tuple(self._trades)

    @property
    def equity_history(self) -> tuple[float, ...]:
        return tuple(self._equity_history)

    def record_trade(
        self,
        *,
        symbol: str,
        signal: str,
        quantity: float,
        execution_price: float,
        realized_pnl: float,
        current_equity: float,
    ) -> TradeRecord:
        if not symbol.strip():
            raise ValueError(
                "Symbol cannot be empty."
            )

        if signal not in {"BUY", "SELL"}:
            raise ValueError(
                "Performance records must be BUY or SELL."
            )

        if quantity <= 0:
            raise ValueError(
                "Quantity must be greater than zero."
            )

        if execution_price <= 0:
            raise ValueError(
                "Execution price must be greater than zero."
            )

        if current_equity <= 0:
            raise ValueError(
                "Current equity must be greater than zero."
            )

        record = TradeRecord(
            symbol=symbol,
            signal=signal,
            quantity=quantity,
            execution_price=execution_price,
            realized_pnl=realized_pnl,
        )

        self._trades.append(record)
        self._equity_history.append(current_equity)

        return record

    def update_equity(
        self,
        current_equity: float,
    ) -> None:
        if current_equity <= 0:
            raise ValueError(
                "Current equity must be greater than zero."
            )

        self._equity_history.append(current_equity)

    def realized_pnl(self) -> float:
        return sum(
            trade.realized_pnl
            for trade in self._trades
        )

    def total_trades(self) -> int:
        return len(self._trades)

    def winning_trades(self) -> int:
        return sum(
            trade.realized_pnl > 0
            for trade in self._trades
        )

    def losing_trades(self) -> int:
        return sum(
            trade.realized_pnl < 0
            for trade in self._trades
        )

    def win_rate_percent(self) -> float:
        total = self.total_trades()

        if total == 0:
            return 0.0

        return (
            self.winning_trades()
            / total
            * 100.0
        )

    def profit_factor(self) -> float:
        gross_profit = sum(
            trade.realized_pnl
            for trade in self._trades
            if trade.realized_pnl > 0
        )

        gross_loss = sum(
            -trade.realized_pnl
            for trade in self._trades
            if trade.realized_pnl < 0
        )

        if gross_loss == 0:
            if gross_profit > 0:
                return float("inf")

            return 0.0

        return gross_profit / gross_loss

    def cumulative_return_percent(
        self,
        current_equity: float | None = None,
    ) -> float:
        equity = (
            current_equity
            if current_equity is not None
            else self._equity_history[-1]
        )

        return (
            (equity - self.starting_capital)
            / self.starting_capital
            * 100.0
        )

    def maximum_drawdown_percent(self) -> float:
        if not self._equity_history:
            return 0.0

        peak = self._equity_history[0]
        maximum_drawdown = 0.0

        for equity in self._equity_history:
            peak = max(peak, equity)

            if peak <= 0:
                continue

            drawdown = (
                (peak - equity)
                / peak
                * 100.0
            )

            maximum_drawdown = max(
                maximum_drawdown,
                drawdown,
            )

        return maximum_drawdown

    def sharpe_ratio(self) -> float:
        if len(self._equity_history) < 3:
            return 0.0

        returns: list[float] = []

        for previous, current in zip(
            self._equity_history,
            self._equity_history[1:],
        ):
            if previous <= 0:
                continue

            returns.append(
                (current - previous)
                / previous
            )

        if len(returns) < 2:
            return 0.0

        mean_return = sum(returns) / len(returns)

        variance = sum(
            (value - mean_return) ** 2
            for value in returns
        ) / (len(returns) - 1)

        standard_deviation = sqrt(variance)

        if standard_deviation == 0:
            return 0.0

        return (
            mean_return
            / standard_deviation
            * sqrt(len(returns))
        )

    def snapshot(
        self,
        current_equity: float | None = None,
    ) -> PerformanceSnapshot:
        equity = (
            current_equity
            if current_equity is not None
            else self._equity_history[-1]
        )

        return PerformanceSnapshot(
            starting_capital=self.starting_capital,
            current_equity=equity,
            realized_pnl=self.realized_pnl(),
            total_trades=self.total_trades(),
            winning_trades=self.winning_trades(),
            losing_trades=self.losing_trades(),
            win_rate_percent=self.win_rate_percent(),
            profit_factor=self.profit_factor(),
            cumulative_return_percent=(
                self.cumulative_return_percent(equity)
            ),
            maximum_drawdown_percent=(
                self.maximum_drawdown_percent()
            ),
            sharpe_ratio=self.sharpe_ratio(),
        )