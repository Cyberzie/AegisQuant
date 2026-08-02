from datetime import datetime

from fastapi.testclient import TestClient

from app.database.session import SessionLocal
from app.main import app
from app.models.instrument import Instrument
from app.models.market_data import MarketData


client = TestClient(app)


def create_test_rows(db, instrument_id):
    rows = [
        MarketData(
            instrument_id=instrument_id,
            timestamp=datetime(2026, 8, 6, 10, 0, 0),
            open=100.0,
            high=105.0,
            low=99.0,
            close=103.0,
            volume=1000.0,
        ),
        MarketData(
            instrument_id=instrument_id,
            timestamp=datetime(2026, 8, 6, 11, 0, 0),
            open=103.0,
            high=108.0,
            low=102.0,
            close=107.0,
            volume=1200.0,
        ),
        MarketData(
            instrument_id=instrument_id,
            timestamp=datetime(2026, 8, 6, 12, 0, 0),
            open=107.0,
            high=110.0,
            low=106.0,
            close=109.0,
            volume=1400.0,
        ),
        MarketData(
            instrument_id=instrument_id,
            timestamp=datetime(2026, 8, 6, 13, 0, 0),
            open=109.0,
            high=112.0,
            low=108.0,
            close=111.0,
            volume=1600.0,
        ),
        MarketData(
            instrument_id=instrument_id,
            timestamp=datetime(2026, 8, 6, 14, 0, 0),
            open=111.0,
            high=115.0,
            low=110.0,
            close=114.0,
            volume=1800.0,
        ),
    ]

    db.add_all(rows)
    db.commit()

    return rows


def cleanup_test_rows(db, instrument_id):
    db.query(MarketData).filter(
        MarketData.instrument_id == instrument_id,
        MarketData.timestamp >= datetime(2026, 8, 6, 10, 0, 0),
        MarketData.timestamp <= datetime(2026, 8, 6, 14, 0, 0),
    ).delete(
        synchronize_session=False
    )

    db.commit()


def get_aapl(db):
    return (
        db.query(Instrument)
        .filter(Instrument.symbol == "AAPL")
        .first()
    )


def test_get_market_data_by_symbol():
    db = SessionLocal()
    instrument = None

    try:
        instrument = get_aapl(db)

        assert instrument is not None

        create_test_rows(db, instrument.id)

        response = client.get(
            "/market-data/symbol/AAPL",
            params={
                "start": "2026-08-06T10:00:00",
                "end": "2026-08-06T14:00:00",
            },
        )

        assert response.status_code == 200

        data = response.json()

        assert len(data) == 5

        assert data[0]["close"] == 103.0
        assert data[1]["close"] == 107.0
        assert data[2]["close"] == 109.0
        assert data[3]["close"] == 111.0
        assert data[4]["close"] == 114.0

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

        response = client.get(
            "/market-data/symbol/aapl",
            params={
                "start": "2026-08-06T10:00:00",
                "end": "2026-08-06T14:00:00",
            },
        )

        assert response.status_code == 200

        data = response.json()

        assert len(data) == 5

    finally:
        if instrument is not None:
            cleanup_test_rows(db, instrument.id)

        db.close()


def test_get_market_data_by_symbol_not_found():
    response = client.get(
        "/market-data/symbol/UNKNOWN",
    )

    assert response.status_code == 404

    data = response.json()

    assert "not found" in data["detail"].lower()


def test_market_data_limit():
    db = SessionLocal()
    instrument = None

    try:
        instrument = get_aapl(db)

        assert instrument is not None

        create_test_rows(db, instrument.id)

        response = client.get(
            "/market-data/symbol/AAPL",
            params={
                "start": "2026-08-06T10:00:00",
                "end": "2026-08-06T14:00:00",
                "limit": 2,
            },
        )

        assert response.status_code == 200

        data = response.json()

        assert len(data) == 2
        assert data[0]["close"] == 103.0
        assert data[1]["close"] == 107.0

    finally:
        if instrument is not None:
            cleanup_test_rows(db, instrument.id)

        db.close()


def test_market_data_skip():
    db = SessionLocal()
    instrument = None

    try:
        instrument = get_aapl(db)

        assert instrument is not None

        create_test_rows(db, instrument.id)

        response = client.get(
            "/market-data/symbol/AAPL",
            params={
                "start": "2026-08-06T10:00:00",
                "end": "2026-08-06T14:00:00",
                "skip": 2,
                "limit": 2,
            },
        )

        assert response.status_code == 200

        data = response.json()

        assert len(data) == 2
        assert data[0]["close"] == 109.0
        assert data[1]["close"] == 111.0

    finally:
        if instrument is not None:
            cleanup_test_rows(db, instrument.id)

        db.close()


def test_market_data_start_filter():
    db = SessionLocal()
    instrument = None

    try:
        instrument = get_aapl(db)

        assert instrument is not None

        create_test_rows(db, instrument.id)

        response = client.get(
            "/market-data/symbol/AAPL",
            params={
                "start": "2026-08-06T12:00:00",
                "end": "2026-08-06T14:00:00",
            },
        )

        assert response.status_code == 200

        data = response.json()

        assert len(data) == 3
        assert data[0]["close"] == 109.0

    finally:
        if instrument is not None:
            cleanup_test_rows(db, instrument.id)

        db.close()


def test_market_data_end_filter():
    db = SessionLocal()
    instrument = None

    try:
        instrument = get_aapl(db)

        assert instrument is not None

        create_test_rows(db, instrument.id)

        response = client.get(
            "/market-data/symbol/AAPL",
            params={
                "start": "2026-08-06T10:00:00",
                "end": "2026-08-06T12:00:00",
            },
        )

        assert response.status_code == 200

        data = response.json()

        assert len(data) == 3
        assert data[-1]["close"] == 109.0

    finally:
        if instrument is not None:
            cleanup_test_rows(db, instrument.id)

        db.close()