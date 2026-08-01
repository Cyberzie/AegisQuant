from datetime import datetime

from app.database.session import SessionLocal
from app.models.instrument import Instrument
from app.models.market_data import MarketData
from app.providers.mock import MockMarketDataProvider
from app.services.provider_ingestion import ingest_from_provider


def test_provider_ingestion():
    db = SessionLocal()

    provider = MockMarketDataProvider()

    start = datetime(2026, 8, 4, 10, 0, 0)
    end = datetime(2026, 8, 4, 12, 0, 0)

    try:
        instrument = (
            db.query(Instrument)
            .filter(Instrument.symbol == "AAPL")
            .first()
        )

        assert instrument is not None

        result = ingest_from_provider(
            db=db,
            provider=provider,
            symbol="AAPL",
            start=start,
            end=end,
        )

        assert result.received == 3
        assert result.inserted == 3
        assert result.duplicates == 0
        assert result.invalid == 0

    finally:
        if "instrument" in locals() and instrument is not None:
            db.query(MarketData).filter(
                MarketData.instrument_id == instrument.id,
                MarketData.timestamp >= start,
                MarketData.timestamp <= end,
            ).delete(synchronize_session=False)

            db.commit()

        db.close()