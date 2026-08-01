from datetime import datetime

from app.database.session import SessionLocal
from app.models.market_data import MarketData
from app.services.market_data_ingestion import ingest_market_data


def test_market_data_ingestion():
    db = SessionLocal()

    test_timestamps = [
        datetime(2026, 8, 3, 10, 0, 0),
        datetime(2026, 8, 3, 11, 0, 0),
        datetime(2026, 8, 3, 12, 0, 0),
    ]

    rows = [
        {
            "instrument_id": 1,
            "timestamp": test_timestamps[0],
            "open": 240.0,
            "high": 245.0,
            "low": 238.0,
            "close": 243.0,
            "volume": 1500000,
        },
        {
            "instrument_id": 1,
            "timestamp": test_timestamps[1],
            "open": 243.0,
            "high": 248.0,
            "low": 241.0,
            "close": 246.0,
            "volume": 1600000,
        },
        {
            "instrument_id": 1,
            "timestamp": test_timestamps[2],
            "open": 250.0,
            "high": 245.0,
            "low": 240.0,
            "close": 243.0,
            "volume": 1600000,
        },
    ]

    try:
        db.query(MarketData).filter(
            MarketData.instrument_id == 1,
            MarketData.timestamp.in_(test_timestamps),
        ).delete(synchronize_session=False)

        db.commit()

        result = ingest_market_data(db, rows)

        assert result.received == 3
        assert result.inserted == 2
        assert result.invalid == 1
        assert result.duplicates == 0

    finally:
        db.query(MarketData).filter(
            MarketData.instrument_id == 1,
            MarketData.timestamp.in_(test_timestamps),
        ).delete(synchronize_session=False)

        db.commit()
        db.close()