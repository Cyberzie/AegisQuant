from __future__ import annotations

from dataclasses import dataclass

from app.services.portfolio import Portfolio
from app.services.risk_management import assess_trade_risk


@dataclass(frozen=True)
class ExecutionResult:
    symbol: str
    signal: str
    executed: bool
    reason: str
    quantity: float
    execution_price: float
    realized_pnl: float
    cash_after: float


class PaperExecutionEngine:
    def __init__(
        self,
        portfolio: Portfolio,
        *,
        transaction_cost_percent: float = 0.10,
        slippage_percent: float = 0.05,
    ) -> None:
        if transaction_cost_percent < 0:
            raise ValueError(
                "Transaction cost cannot be negative."
            )

        if slippage_percent < 0:
            raise ValueError(
                "Slippage cannot be negative."
            )

        self.portfolio = portfolio
        self.transaction_cost_percent = (
            transaction_cost_percent
        )
        self.slippage_percent = slippage_percent

    def execute(
        self,
        *,
        symbol: str,
        signal: str,
        confidence: float,
        entry_price: float,
    ) -> ExecutionResult:
        if signal not in {"BUY", "SELL", "HOLD"}:
            raise ValueError(
                "Signal must be BUY, SELL, or HOLD."
            )

        if entry_price <= 0:
            raise ValueError(
                "Entry price must be greater than zero."
            )

        if signal == "HOLD":
            return ExecutionResult(
                symbol=symbol,
                signal=signal,
                executed=False,
                reason="HOLD signal produces no order.",
                quantity=0.0,
                execution_price=entry_price,
                realized_pnl=0.0,
                cash_after=self.portfolio.cash,
            )

        # SELL closes the existing long position.
        #
        # A SELL is not risk-sized like a new BUY order.
        # The position already exists, so the correct quantity
        # is the full quantity currently held.
        if signal == "SELL":
            position = self.portfolio.positions.get(
                symbol
            )

            if position is None:
                return ExecutionResult(
                    symbol=symbol,
                    signal="SELL",
                    executed=False,
                    reason="No open position to sell.",
                    quantity=0.0,
                    execution_price=entry_price,
                    realized_pnl=0.0,
                    cash_after=self.portfolio.cash,
                )

            execution_price = self._execution_price(
                signal=signal,
                market_price=entry_price,
            )

            return self._execute_sell(
                symbol=symbol,
                quantity=position.quantity,
                execution_price=execution_price,
            )

        # BUY orders are sized through risk management.
        risk = assess_trade_risk(
            signal=signal,
            confidence=confidence,
            entry_price=entry_price,
            capital=self.portfolio.equity(
                {symbol: entry_price}
            ),
        )

        if not risk.approved:
            return ExecutionResult(
                symbol=symbol,
                signal=signal,
                executed=False,
                reason=risk.reason,
                quantity=0.0,
                execution_price=entry_price,
                realized_pnl=0.0,
                cash_after=self.portfolio.cash,
            )

        quantity = risk.position_size

        if quantity <= 0:
            return ExecutionResult(
                symbol=symbol,
                signal=signal,
                executed=False,
                reason=(
                    "Risk management produced "
                    "zero position size."
                ),
                quantity=0.0,
                execution_price=entry_price,
                realized_pnl=0.0,
                cash_after=self.portfolio.cash,
            )

        execution_price = self._execution_price(
            signal=signal,
            market_price=entry_price,
        )

        return self._execute_buy(
            symbol=symbol,
            quantity=quantity,
            execution_price=execution_price,
        )

    def _execute_buy(
        self,
        *,
        symbol: str,
        quantity: float,
        execution_price: float,
    ) -> ExecutionResult:
        cost = (
            quantity
            * execution_price
        )

        transaction_cost = (
            cost
            * self.transaction_cost_percent
            / 100.0
        )

        total_cost = cost + transaction_cost

        if total_cost > self.portfolio.cash:
            return ExecutionResult(
                symbol=symbol,
                signal="BUY",
                executed=False,
                reason=(
                    "Insufficient cash after "
                    "execution costs."
                ),
                quantity=0.0,
                execution_price=execution_price,
                realized_pnl=0.0,
                cash_after=self.portfolio.cash,
            )

        self.portfolio.buy(
            symbol=symbol,
            quantity=quantity,
            price=execution_price,
        )

        self.portfolio.cash -= transaction_cost

        return ExecutionResult(
            symbol=symbol,
            signal="BUY",
            executed=True,
            reason="BUY order executed.",
            quantity=quantity,
            execution_price=execution_price,
            realized_pnl=0.0,
            cash_after=self.portfolio.cash,
        )

    def _execute_sell(
        self,
        *,
        symbol: str,
        quantity: float,
        execution_price: float,
    ) -> ExecutionResult:
        position = self.portfolio.positions.get(
            symbol
        )

        if position is None:
            return ExecutionResult(
                symbol=symbol,
                signal="SELL",
                executed=False,
                reason="No open position to sell.",
                quantity=0.0,
                execution_price=execution_price,
                realized_pnl=0.0,
                cash_after=self.portfolio.cash,
            )

        quantity = min(
            quantity,
            position.quantity,
        )

        proceeds = (
            quantity
            * execution_price
        )

        transaction_cost = (
            proceeds
            * self.transaction_cost_percent
            / 100.0
        )

        realized_pnl = self.portfolio.sell(
            symbol=symbol,
            quantity=quantity,
            price=execution_price,
        )

        self.portfolio.cash -= transaction_cost
        realized_pnl -= transaction_cost
        self.portfolio.realized_pnl -= transaction_cost

        return ExecutionResult(
            symbol=symbol,
            signal="SELL",
            executed=True,
            reason="SELL order executed.",
            quantity=quantity,
            execution_price=execution_price,
            realized_pnl=realized_pnl,
            cash_after=self.portfolio.cash,
        )

    def _execution_price(
        self,
        *,
        signal: str,
        market_price: float,
    ) -> float:
        slippage = (
            self.slippage_percent / 100.0
        )

        if signal == "BUY":
            return market_price * (
                1.0 + slippage
            )

        return market_price * (
            1.0 - slippage
        )