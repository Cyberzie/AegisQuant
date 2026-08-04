from __future__ import annotations

from dataclasses import dataclass

from app.services.ensemble_signal import EnsembleSignal
from app.services.risk_management import (
    RiskAssessment,
    RiskParameters,
    assess_trade_risk,
)


@dataclass(frozen=True)
class TradingDecision:
    signal: str
    confidence: float
    expected_return_percent: float
    rule_weight: float
    ml_weight: float
    risk: RiskAssessment


def build_trading_decision(
    ensemble: EnsembleSignal,
    *,
    entry_price: float,
    capital: float,
    risk_parameters: RiskParameters | None = None,
) -> TradingDecision:
    """Convert ensemble evidence into a risk-aware trading decision."""
    risk = assess_trade_risk(
        signal=ensemble.signal,
        confidence=ensemble.confidence,
        entry_price=entry_price,
        capital=capital,
        parameters=risk_parameters,
    )

    return TradingDecision(
        signal=ensemble.signal,
        confidence=ensemble.confidence,
        expected_return_percent=(
            ensemble.ml_expected_return_percent
        ),
        rule_weight=ensemble.rule_weight,
        ml_weight=ensemble.ml_weight,
        risk=risk,
    )
