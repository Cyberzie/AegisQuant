from datetime import datetime

from fastapi.testclient import TestClient

from app.database.session import SessionLocal
from app.main import app
from app.models.instrument import Instrument
from app.models.market_data import MarketData
from app.services.market_data_query import get_market_data_summary


client = TestClient(app)

SYMBOL = "SUMMARYTEST"


def create_test_instrument(db):
    instrument = Instrument(
        symbol=SYMBOL,
        name="Summary Test Instrument",
        asset_type="stock",
        exchange="TEST",
        currency="USD",
    )
    db.add(instrument)
    db.commit()
    db.refresh(instrument)
    return instrument


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
            high=115.0,
            low=110.0,
            close=114.0,
            volume=1100.0,
        ),
        MarketData(
            instrument_id=instrument_id,
            timestamp=datetime(2026, 8, 7, 12, 0, 0),
            open=114.0,
            high=116.0,
            low=113.0,
            close=115.0,
            volume=1200.0,
        ),
    ]
    db.add_all(rows)
    db.commit()


def cleanup_test_data(db):
    instrument = (
        db.query(Instrument)
        .filter(Instrument.symbol == SYMBOL)
        .first()
    )

    if instrument is not None:
        db.query(MarketData).filter(
            MarketData.instrument_id == instrument.id
        ).delete(synchronize_session=False)

        db.delete(instrument)
        db.commit()


def test_get_market_data_summary():
    db = SessionLocal()

    try:
        cleanup_test_data(db)
        instrument = create_test_instrument(db)
        create_test_rows(db, instrument.id)

        result = get_market_data_summary(db, SYMBOL)

        assert result["symbol"] == SYMBOL
        assert result["data_points"] == 3
        assert result["first_timestamp"] == datetime(2026, 8, 7, 10, 0, 0)
        assert result["last_timestamp"] == datetime(2026, 8, 7, 12, 0, 0)
        assert result["first_open"] == 110.0
        assert result["latest_close"] == 115.0
        assert result["high"] == 116.0
        assert result["low"] == 109.0
        assert result["change"] == 5.0
        assert result["change_percent"] == (5.0 / 110.0) * 100

    finally:
        cleanup_test_data(db)
        db.close()


def test_get_market_data_summary_date_filter():
    db = SessionLocal()

    try:
        cleanup_test_data(db)
        instrument = create_test_instrument(db)
        create_test_rows(db, instrument.id)

        result = get_market_data_summary(
            db,
            SYMBOL,
            start=datetime(2026, 8, 7, 11, 0, 0),
            end=datetime(2026, 8, 7, 12, 0, 0),
        )

        assert result["data_points"] == 2
        assert result["first_open"] == 111.0
        assert result["latest_close"] == 115.0
        assert result["high"] == 116.0
        assert result["low"] == 110.0

    finally:
        cleanup_test_data(db)
        db.close()


def test_get_market_data_summary_not_found():
    db = SessionLocal()

    try:
        cleanup_test_data(db)

        try:
            get_market_data_summary(db, "UNKNOWN")
            assert False
        except ValueError as exc:
            assert "Instrument with symbol" in str(exc)

    finally:
        db.close()


def test_get_market_data_summary_api():
    db = SessionLocal()

    try:
        cleanup_test_data(db)
        instrument = create_test_instrument(db)
        create_test_rows(db, instrument.id)

        response = client.get(f"/market-data/summary/{SYMBOL}")

        assert response.status_code == 200

        data = response.json()

        assert data["symbol"] == SYMBOL
        assert data["data_points"] == 3
        assert data["first_open"] == 110.0
        assert data["latest_close"] == 115.0
        assert data["high"] == 116.0
        assert data["low"] == 109.0
        assert data["change"] == 5.0

    finally:
        cleanup_test_data(db)
        db.close()


def test_get_market_data_summary_api_not_found():
    response = client.get("/market-data/summary/UNKNOWN")

    assert response.status_code == 404
    assert "Instrument with symbol" in response.json()["detail"]
