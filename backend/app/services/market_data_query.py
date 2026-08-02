from datetime import datetime

from sqlalchemy import func
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


def get_latest_market_data(
    db: Session,
    symbol: str,
    limit: int = 100,
) -> list[MarketData]:
    instrument = (
        db.query(Instrument)
        .filter(Instrument.symbol == symbol.upper())
        .first()
    )

    if instrument is None:
        raise ValueError(
            f"Instrument with symbol '{symbol}' not found."
        )

    return (
        db.query(MarketData)
        .filter(MarketData.instrument_id == instrument.id)
        .order_by(MarketData.timestamp.desc())
        .limit(limit)
        .all()
    )

def get_market_data_summary(
    db: Session,
    symbol: str,
    start: datetime | None = None,
    end: datetime | None = None,
) -> dict:
    instrument = (
        db.query(Instrument)
        .filter(Instrument.symbol == symbol.upper())
        .first()
    )

    if instrument is None:
        raise ValueError(
            f"Instrument with symbol '{symbol}' not found."
        )

    query = db.query(MarketData).filter(
        MarketData.instrument_id == instrument.id
    )

    if start is not None:
        query = query.filter(MarketData.timestamp >= start)

    if end is not None:
        query = query.filter(MarketData.timestamp <= end)

    first_row = query.order_by(MarketData.timestamp.asc()).first()
    latest_row = query.order_by(MarketData.timestamp.desc()).first()

    if first_row is None or latest_row is None:
        raise ValueError(
            f"No market data found for instrument '{symbol}'."
        )

    data_points = query.count()
    high = query.with_entities(
        func.max(MarketData.high)
    ).scalar()
    low = query.with_entities(
        func.min(MarketData.low)
    ).scalar()

    change = latest_row.close - first_row.open
    change_percent = (
        change / first_row.open * 100
        if first_row.open
        else 0.0
    )

    return {
        "symbol": instrument.symbol,
        "data_points": data_points,
        "first_timestamp": first_row.timestamp,
        "last_timestamp": latest_row.timestamp,
        "first_open": first_row.open,
        "latest_close": latest_row.close,
        "high": high,
        "low": low,
        "change": change,
        "change_percent": change_percent,
    }
