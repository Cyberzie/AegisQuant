from __future__ import annotations

from dataclasses import dataclass

from app.services.ensemble_signal import EnsembleSignal
from app.services.paper_execution import (
    ExecutionResult,
    PaperExecutionEngine,
)
from app.services.portfolio import Portfolio
from app.services.risk_management import RiskParameters
from app.services.trading_decision import (
    TradingDecision,
    build_trading_decision,
)


@dataclass(frozen=True)
class PaperTradingResult:
    symbol: str
    entry_price: float
    capital_before: float
    decision: TradingDecision
    execution: ExecutionResult
    equity_after: float


class PaperTradingEngine:
    """
    Orchestrates the AegisQuant decision-to-execution pipeline.

    Pipeline:

        EnsembleSignal
            ↓
        TradingDecision
            ↓
        RiskAssessment
            ↓
        PaperExecutionEngine
            ↓
        Portfolio
    """

    def __init__(
        self,
        portfolio: Portfolio,
        *,
        transaction_cost_percent: float = 0.10,
        slippage_percent: float = 0.05,
        risk_parameters: RiskParameters | None = None,
    ) -> None:
        self.portfolio = portfolio
        self.risk_parameters = risk_parameters

        self.execution_engine = PaperExecutionEngine(
            portfolio,
            transaction_cost_percent=(
                transaction_cost_percent
            ),
            slippage_percent=(
                slippage_percent
            ),
        )

    def process(
        self,
        *,
        symbol: str,
        ensemble: EnsembleSignal,
        entry_price: float,
    ) -> PaperTradingResult:
        if not symbol.strip():
            raise ValueError(
                "Symbol cannot be empty."
            )

        if entry_price <= 0:
            raise ValueError(
                "Entry price must be greater than zero."
            )

        capital_before = self.portfolio.equity(
            {symbol: entry_price}
        )

        decision = build_trading_decision(
            ensemble,
            entry_price=entry_price,
            capital=capital_before,
            risk_parameters=self.risk_parameters,
        )

        execution = self.execution_engine.execute(
            symbol=symbol,
            signal=decision.signal,
            confidence=decision.confidence,
            entry_price=entry_price,
        )

        equity_after = self.portfolio.equity(
            {symbol: entry_price}
        )

        return PaperTradingResult(
            symbol=symbol,
            entry_price=entry_price,
            capital_before=capital_before,
            decision=decision,
            execution=execution,
            equity_after=equity_after,
        )