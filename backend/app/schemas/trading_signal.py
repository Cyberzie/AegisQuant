from datetime import datetime

from pydantic import BaseModel


class TradingSignalResponse(BaseModel):
    symbol: str
    timestamp: datetime

    signal: str
    confidence: float

    close: float

    rsi_14: float | None
    macd: float | None
    macd_signal: float | None

    sma_20: float | None
    ema_20: float | None

    bollinger_middle: float | None
    bollinger_upper: float | None
    bollinger_lower: float | None

    atr_14: float | None