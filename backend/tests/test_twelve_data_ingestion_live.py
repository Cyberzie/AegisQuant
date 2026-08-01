from datetime import datetime
import pytest

from app.database.session import SessionLocal
from app.models.instrument import Instrument
from app.models.market_data import MarketData
from app.providers.factory import get_market_data_provider
from app.services.provider_ingestion import ingest_from_provider

@pytest.mark.live
def test_twelve_data_live_ingestion():
    db = SessionLocal()

    try:
        instrument = (
            db.query(Instrument)
            .filter(Instrument.symbol == "AAPL")
            .first()
        )

        assert instrument is not None

        provider = get_market_data_provider("twelve_data")

        rows = provider.get_market_data(
            symbol="AAPL",
        )

        assert rows

        timestamps = {
            row["timestamp"]
            for row in rows
        }

        existing_timestamps = {
            timestamp
            for (timestamp,) in (
                db.query(MarketData.timestamp)
                .filter(
                    MarketData.instrument_id == instrument.id,
                    MarketData.timestamp.in_(timestamps),
                )
                .all()
            )
        }

        result = ingest_from_provider(
            db=db,
            provider=provider,
            symbol="AAPL",
        )

        assert result.received == len(rows)
        assert result.invalid == 0
        assert (
            result.inserted + result.duplicates
            == result.received
        )

        saved_count = (
            db.query(MarketData)
            .filter(
                MarketData.instrument_id == instrument.id,
                MarketData.timestamp.in_(timestamps),
            )
            .count()
        )

        assert saved_count == len(timestamps)

        new_timestamps = (
            timestamps - existing_timestamps
        )

        if new_timestamps:
            db.query(MarketData).filter(
                MarketData.instrument_id == instrument.id,
                MarketData.timestamp.in_(new_timestamps),
            ).delete(
                synchronize_session=False
            )

            db.commit()

        print(
            f"\nReceived: {result.received}"
        )
        print(
            f"Inserted: {result.inserted}"
        )
        print(
            f"Duplicates: {result.duplicates}"
        )
        print(
            f"Invalid: {result.invalid}"
        )

    finally:
        db.close()
