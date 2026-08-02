from datetime import datetime

from pydantic import BaseModel


class TechnicalIndicatorsResponse(BaseModel):
    symbol: str
    timestamp: datetime

    close: float

    sma_20: float | None
    ema_20: float | None
    rsi_14: float | None

    macd: float | None
    macd_signal: float | None
    macd_histogram: float | None

    bollinger_middle: float | None
    bollinger_upper: float | None
    bollinger_lower: float | None

    atr_14: float | None