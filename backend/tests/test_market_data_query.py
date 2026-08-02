from datetime import datetime

from app.database.session import SessionLocal
from app.models.instrument import Instrument
from app.models.market_data import MarketData
from app.services.market_data_query import get_market_data_by_symbol


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
            open=100.0,
            high=105.0,
            low=99.0,
            close=103.0,
            volume=1000.0,
        ),
        MarketData(
            instrument_id=instrument_id,
            timestamp=datetime(2026, 8, 7, 11, 0, 0),
            open=103.0,
            high=108.0,
            low=102.0,
            close=107.0,
            volume=1200.0,
        ),
        MarketData(
            instrument_id=instrument_id,
            timestamp=datetime(2026, 8, 7, 12, 0, 0),
            open=107.0,
            high=110.0,
            low=106.0,
            close=109.0,
            volume=1400.0,
        ),
    ]

    db.add_all(rows)
    db.commit()

    return rows


def cleanup_test_rows(db, instrument_id):
    db.query(MarketData).filter(
        MarketData.instrument_id == instrument_id,
        MarketData.timestamp >= datetime(2026, 8, 7, 10, 0, 0),
        MarketData.timestamp <= datetime(2026, 8, 7, 12, 0, 0),
    ).delete(
        synchronize_session=False
    )

    db.commit()


def test_get_market_data_by_symbol():
    db = SessionLocal()
    instrument = None

    try:
        instrument = get_aapl(db)

        assert instrument is not None

        create_test_rows(db, instrument.id)

        rows = get_market_data_by_symbol(
            db=db,
            symbol="AAPL",
            start=datetime(2026, 8, 7, 10, 0, 0),
            end=datetime(2026, 8, 7, 12, 0, 0),
        )

        assert len(rows) == 3
        assert rows[0].close == 103.0
        assert rows[1].close == 107.0
        assert rows[2].close == 109.0

    finally:
        if instrument is not None:
            cleanup_test_rows(db, instrument.id)

        db.close()


def test_get_market_data_by_symbol_case_insensitive():
    db = SessionLocal()
    instrument = None

    try:
        instrument = get_aapl(db)

        assert instrument is not None

        create_test_rows(db, instrument.id)

        rows = get_market_data_by_symbol(
            db=db,
            symbol="aapl",
            start=datetime(2026, 8, 7, 10, 0, 0),
            end=datetime(2026, 8, 7, 12, 0, 0),
        )

        assert len(rows) == 3

    finally:
        if instrument is not None:
            cleanup_test_rows(db, instrument.id)

        db.close()


def test_get_market_data_by_symbol_not_found():
    db = SessionLocal()

    try:
        rows = get_market_data_by_symbol(
            db=db,
            symbol="DOES_NOT_EXIST",
        )

        assert rows == []

    finally:
        db.close()


def test_get_market_data_with_pagination():
    db = SessionLocal()
    instrument = None

    try:
        instrument = get_aapl(db)

        assert instrument is not None

        create_test_rows(db, instrument.id)

        rows = get_market_data_by_symbol(
            db=db,
            symbol="AAPL",
            start=datetime(2026, 8, 7, 10, 0, 0),
            end=datetime(2026, 8, 7, 12, 0, 0),
            skip=1,
            limit=1,
        )

        assert len(rows) == 1
        assert rows[0].close == 107.0

    finally:
        if instrument is not None:
            cleanup_test_rows(db, instrument.id)

        db.close()


def test_get_market_data_with_date_filter():
    db = SessionLocal()
    instrument = None

    try:
        instrument = get_aapl(db)

        assert instrument is not None

        create_test_rows(db, instrument.id)

        rows = get_market_data_by_symbol(
            db=db,
            symbol="AAPL",
            start=datetime(2026, 8, 7, 11, 0, 0),
            end=datetime(2026, 8, 7, 12, 0, 0),
        )

        assert len(rows) == 2
        assert rows[0].close == 107.0
        assert rows[1].close == 109.0

    finally:
        if instrument is not None:
            cleanup_test_rows(db, instrument.id)

        db.close()