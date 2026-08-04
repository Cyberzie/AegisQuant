from datetime import datetime

from pydantic import BaseModel, Field


class PaperTradingRequest(BaseModel):
    symbol: str = Field(min_length=1)

    entry_price: float = Field(gt=0)

    signal: str
    confidence: float = Field(ge=0, le=1)

    rule_signal: str
    rule_confidence: float = Field(ge=0, le=1)

    ml_signal: str
    ml_confidence: float = Field(ge=0, le=1)

    ml_expected_return_percent: float

    rule_weight: float = Field(ge=0, le=1)
    ml_weight: float = Field(ge=0, le=1)


class PaperTradingExecutionResponse(BaseModel):
    symbol: str
    signal: str
    executed: bool
    reason: str
    quantity: float
    execution_price: float
    realized_pnl: float
    cash_after: float


class PaperTradingRiskResponse(BaseModel):
    approved: bool
    reason: str
    risk_amount: float
    position_size: float
    position_value: float
    position_percent: float
    stop_loss_price: float | None
    take_profit_price: float | None


class PaperTradingResponse(BaseModel):
    symbol: str
    timestamp: datetime
    entry_price: float
    capital_before: float
    equity_after: float

    signal: str
    confidence: float
    expected_return_percent: float

    rule_weight: float
    ml_weight: float

    risk: PaperTradingRiskResponse
    execution: PaperTradingExecutionResponse