from datetime import datetime

from pydantic import BaseModel


class BacktestTradeResponse(BaseModel):
    timestamp: datetime
    signal: str
    confidence: float
    entry_price: float
    exit_price: float
    return_percent: float
    profitable: bool
    position_return_percent: float
    equity_after: float


class BacktestResponse(BaseModel):
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

    trades: list[BacktestTradeResponse]


class ConfidenceBucketResponse(BaseModel):
    label: str
    minimum_confidence: float
    maximum_confidence: float

    trade_count: int
    winning_trades: int
    losing_trades: int

    win_rate: float
    average_return_percent: float
    total_return_percent: float

    gross_profit_percent: float
    gross_loss_percent: float
    profit_factor: float


class BacktestAnalysisResponse(BaseModel):
    symbol: str
    horizon: int

    total_rows: int
    evaluated_rows: int
    total_trades: int

    confidence_buckets: list[ConfidenceBucketResponse]

    high_confidence_win_rate: float
    low_confidence_win_rate: float

    high_confidence_average_return_percent: float
    low_confidence_average_return_percent: float