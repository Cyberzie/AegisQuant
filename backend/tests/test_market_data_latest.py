from datetime import datetime

import pytest

from app.database.session import SessionLocal
from app.models.instrument import Instrument
from app.models.market_data import MarketData
from app.services.market_data_query import get_latest_market_data


def get_aapl(db):
    return (
        db.query(Instrument)
        .filter(Instrument.symbol == "AAPL")
        .first()
    )


def create_test_rows(db, instrument_id):
    rows = [
        MarketData(
            instrument_id=instrument_id,
            timestamp=datetime(2026, 8, 7, 10, 0, 0),
            open=110.0,
            high=112.0,
            low=109.0,
            close=111.0,
            volume=1000.0,
        ),
        MarketData(
            instrument_id=instrument_id,
            timestamp=datetime(2026, 8, 7, 11, 0, 0),
            open=111.0,
            high=113.0,
            low=110.0,
            close=112.0,
            volume=1100.0,
        ),
        MarketData(
            instrument_id=instrument_id,
            timestamp=datetime(2026, 8, 7, 12, 0, 0),
            open=112.0,
            high=114.0,
            low=111.0,
            close=113.0,
            volume=1200.0,
        ),
    ]

    db.add_all(rows)
    db.commit()


def cleanup_test_rows(db, instrument_id):
    db.query(MarketData).filter(
        MarketData.instrument_id == instrument_id,
        MarketData.timestamp >= datetime(2026, 8, 7, 10, 0, 0),
        MarketData.timestamp <= datetime(2026, 8, 7, 12, 0, 0),
    ).delete(
        synchronize_session=False
    )

    db.commit()


def test_get_latest_market_data():
    db = SessionLocal()
    instrument = None

    try:
        instrument = get_aapl(db)

        assert instrument is not None

        create_test_rows(db, instrument.id)

        data = get_latest_market_data(
            db=db,
            symbol="AAPL",
            limit=3,
        )

        assert len(data) == 3
        assert data[0].timestamp == datetime(2026, 8, 7, 12, 0, 0)
        assert data[1].timestamp == datetime(2026, 8, 7, 11, 0, 0)
        assert data[2].timestamp == datetime(2026, 8, 7, 10, 0, 0)

    finally:
        if instrument is not None:
            cleanup_test_rows(db, instrument.id)

        db.close()


def test_get_latest_market_data_case_insensitive():
    db = SessionLocal()
    instrument = None

    try:
        instrument = get_aapl(db)

        assert instrument is not None

        create_test_rows(db, instrument.id)

        data = get_latest_market_data(
            db=db,
            symbol="aapl",
            limit=2,
        )

        assert len(data) == 2
        assert data[0].timestamp == datetime(2026, 8, 7, 12, 0, 0)

    finally:
        if instrument is not None:
            cleanup_test_rows(db, instrument.id)

        db.close()


def test_get_latest_market_data_limit():
    db = SessionLocal()
    instrument = None

    try:
        instrument = get_aapl(db)

        assert instrument is not None

        create_test_rows(db, instrument.id)

        data = get_latest_market_data(
            db=db,
            symbol="AAPL",
            limit=2,
        )

        assert len(data) == 2

    finally:
        if instrument is not None:
            cleanup_test_rows(db, instrument.id)

        db.close()


def test_get_latest_market_data_not_found():
    db = SessionLocal()

    try:
        with pytest.raises(ValueError, match="Instrument with symbol"):
            get_latest_market_data(
                db=db,
                symbol="UNKNOWN",
                limit=10,
            )

    finally:
        db.close()