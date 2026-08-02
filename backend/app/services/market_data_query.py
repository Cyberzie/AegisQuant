from datetime import datetime

from sqlalchemy.orm import Session

from app.models.instrument import Instrument
from app.models.market_data import MarketData


def get_market_data_by_symbol(
    db: Session,
    symbol: str,
    start: datetime | None = None,
    end: datetime | None = None,
    skip: int = 0,
    limit: int = 100,
) -> list[MarketData]:
    instrument = (
        db.query(Instrument)
        .filter(Instrument.symbol == symbol.upper())
        .first()
    )

    if instrument is None:
        return []

    query = (
        db.query(MarketData)
        .filter(MarketData.instrument_id == instrument.id)
    )

    if start is not None:
        query = query.filter(MarketData.timestamp >= start)

    if end is not None:
        query = query.filter(MarketData.timestamp <= end)

    return (
        query
        .order_by(MarketData.timestamp)
        .offset(skip)
        .limit(limit)
        .all()
    )