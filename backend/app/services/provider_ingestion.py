from datetime import datetime

from sqlalchemy.orm import Session

from app.models.instrument import Instrument
from app.providers.base import MarketDataProvider
from app.services.market_data_ingestion import (
    IngestionResult,
    ingest_market_data,
)


def ingest_from_provider(
    db: Session,
    provider: MarketDataProvider,
    symbol: str,
    start: datetime | None = None,
    end: datetime | None = None,
) -> IngestionResult:
    instrument = (
        db.query(Instrument)
        .filter(Instrument.symbol == symbol)
        .first()
    )

    if instrument is None:
        raise ValueError(
            f"Instrument with symbol '{symbol}' not found."
        )

    provider_rows = provider.get_market_data(
        symbol=symbol,
        start=start,
        end=end,
    )

    normalized_rows = []

    for row in provider_rows:
        normalized_rows.append(
            {
                "instrument_id": instrument.id,
                "timestamp": row["timestamp"],
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "volume": row.get("volume"),
            }
        )

    return ingest_market_data(
        db=db,
        rows=normalized_rows,
    )