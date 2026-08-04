from __future__ import annotations

from dataclasses import dataclass
from math import floor, isfinite


@dataclass(frozen=True)
class RiskParameters:
    risk_per_trade_percent: float = 1.0
    stop_loss_percent: float = 2.0
    take_profit_percent: float = 4.0
    maximum_position_percent: float = 25.0
    minimum_confidence: float = 0.55


@dataclass(frozen=True)
class RiskAssessment:
    approved: bool
    reason: str
    risk_amount: float
    position_size: float
    position_value: float
    position_percent: float
    stop_loss_price: float | None
    take_profit_price: float | None


def assess_trade_risk(
    *,
    signal: str,
    confidence: float,
    entry_price: float,
    capital: float,
    parameters: RiskParameters | None = None,
) -> RiskAssessment:
    """Apply deterministic position and loss limits to a trading signal."""
    params = parameters or RiskParameters()

    if signal not in {"BUY", "SELL", "HOLD"}:
        raise ValueError("Signal must be BUY, SELL, or HOLD.")
    if not isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("Confidence must be between 0 and 1.")
    if not isfinite(entry_price) or entry_price <= 0:
        raise ValueError("Entry price must be greater than zero.")
    if not isfinite(capital) or capital <= 0:
        raise ValueError("Capital must be greater than zero.")
    if params.risk_per_trade_percent <= 0:
        raise ValueError("Risk per trade must be greater than zero.")
    if params.stop_loss_percent <= 0:
        raise ValueError("Stop loss percent must be greater than zero.")
    if params.take_profit_percent < 0:
        raise ValueError("Take profit percent cannot be negative.")
    if not 0 < params.maximum_position_percent <= 100:
        raise ValueError("Maximum position percent must be in (0, 100].")
    if not 0 <= params.minimum_confidence <= 1:
        raise ValueError("Minimum confidence must be between 0 and 1.")

    if signal == "HOLD":
        return RiskAssessment(
            approved=False,
            reason="HOLD signal does not open a position.",
            risk_amount=0.0,
            position_size=0.0,
            position_value=0.0,
            position_percent=0.0,
            stop_loss_price=None,
            take_profit_price=None,
        )

    if confidence < params.minimum_confidence:
        return RiskAssessment(
            approved=False,
            reason="Signal confidence is below the minimum threshold.",
            risk_amount=0.0,
            position_size=0.0,
            position_value=0.0,
            position_percent=0.0,
            stop_loss_price=None,
            take_profit_price=None,
        )

    risk_amount = capital * params.risk_per_trade_percent / 100.0
    stop_distance = entry_price * params.stop_loss_percent / 100.0
    maximum_value = capital * params.maximum_position_percent / 100.0

    position_size = min(
        risk_amount / stop_distance,
        maximum_value / entry_price,
    )
    position_size = floor(position_size * 1_000_000) / 1_000_000
    position_value = position_size * entry_price
    position_percent = position_value / capital * 100.0

    if position_size <= 0:
        return RiskAssessment(
            approved=False,
            reason="Risk limits produce a zero-sized position.",
            risk_amount=risk_amount,
            position_size=0.0,
            position_value=0.0,
            position_percent=0.0,
            stop_loss_price=None,
            take_profit_price=None,
        )

    if signal == "BUY":
        stop_loss_price = entry_price - stop_distance
        take_profit_price = (
            entry_price
            * (1.0 + params.take_profit_percent / 100.0)
        )
    else:
        stop_loss_price = entry_price + stop_distance
        take_profit_price = (
            entry_price
            * (1.0 - params.take_profit_percent / 100.0)
        )

    return RiskAssessment(
        approved=True,
        reason="Trade satisfies confidence and risk limits.",
        risk_amount=risk_amount,
        position_size=position_size,
        position_value=position_value,
        position_percent=position_percent,
        stop_loss_price=stop_loss_price,
        take_profit_price=take_profit_price,
    )
