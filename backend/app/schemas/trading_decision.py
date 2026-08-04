from datetime import datetime

from pydantic import BaseModel


class TradingRiskResponse(BaseModel):
    approved: bool
    reason: str

    risk_amount: float
    position_size: float
    position_value: float
    position_percent: float

    stop_loss_price: float | None
    take_profit_price: float | None


class TradingDecisionResponse(BaseModel):
    symbol: str
    timestamp: datetime
    close: float

    signal: str
    confidence: float
    expected_return_percent: float

    rule_weight: float
    ml_weight: float

    risk: TradingRiskResponse