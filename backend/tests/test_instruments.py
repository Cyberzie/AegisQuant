from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.database.session import SessionLocal
from app.main import app
from app.models.instrument import Instrument


client = TestClient(app)


def unique_symbol():
    return f"TST{uuid4().hex[:8].upper()}"


def cleanup_instrument(symbol):
    db = SessionLocal()

    try:
        instrument = (
            db.query(Instrument)
            .filter(Instrument.symbol == symbol)
            .first()
        )

        if instrument is not None:
            db.delete(instrument)
            db.commit()

    finally:
        db.close()


def test_create_instrument():
    symbol = unique_symbol()

    try:
        response = client.post(
            "/instruments/",
            json={
                "symbol": symbol,
                "name": "Test Instrument",
                "asset_type": "stock",
                "exchange": "TEST",
                "currency": "USD",
            },
        )

        assert response.status_code == 201

        data = response.json()

        assert data["symbol"] == symbol
        assert data["name"] == "Test Instrument"
        assert data["asset_type"] == "stock"
        assert data["exchange"] == "TEST"
        assert data["currency"] == "USD"
        assert data["is_active"] is True
        assert data["id"] > 0
        assert data["created_at"] is not None

    finally:
        cleanup_instrument(symbol)


def test_create_instrument_with_optional_fields_omitted():
    symbol = unique_symbol()

    try:
        response = client.post(
            "/instruments/",
            json={
                "symbol": symbol,
                "name": "Minimal Test Instrument",
                "asset_type": "stock",
            },
        )

        assert response.status_code == 201

        data = response.json()

        assert data["symbol"] == symbol
        assert data["name"] == "Minimal Test Instrument"
        assert data["asset_type"] == "stock"
        assert data["exchange"] is None
        assert data["currency"] is None
        assert data["is_active"] is True

    finally:
        cleanup_instrument(symbol)


def test_create_duplicate_instrument_symbol():
    symbol = unique_symbol()

    try:
        first_response = client.post(
            "/instruments/",
            json={
                "symbol": symbol,
                "name": "First Instrument",
                "asset_type": "stock",
            },
        )

        assert first_response.status_code == 201

        duplicate_response = client.post(
            "/instruments/",
            json={
                "symbol": symbol,
                "name": "Duplicate Instrument",
                "asset_type": "stock",
            },
        )

        assert duplicate_response.status_code == 409
        assert (
            duplicate_response.json()["detail"]
            == "Instrument with this symbol already exists."
        )

    finally:
        cleanup_instrument(symbol)


def test_list_instruments():
    symbol_one = unique_symbol()
    symbol_two = unique_symbol()

    try:
        first_response = client.post(
            "/instruments/",
            json={
                "symbol": symbol_one,
                "name": "Test Instrument One",
                "asset_type": "stock",
            },
        )

        second_response = client.post(
            "/instruments/",
            json={
                "symbol": symbol_two,
                "name": "Test Instrument Two",
                "asset_type": "forex",
            },
        )

        assert first_response.status_code == 201
        assert second_response.status_code == 201

        response = client.get("/instruments/")

        assert response.status_code == 200

        data = response.json()

        symbols = [instrument["symbol"] for instrument in data]

        assert symbol_one in symbols
        assert symbol_two in symbols

        ids = [
            instrument["id"]
            for instrument in data
            if instrument["symbol"] in {symbol_one, symbol_two}
        ]

        assert ids == sorted(ids)

    finally:
        cleanup_instrument(symbol_one)
        cleanup_instrument(symbol_two)


def test_get_instrument_by_id():
    symbol = unique_symbol()

    try:
        create_response = client.post(
            "/instruments/",
            json={
                "symbol": symbol,
                "name": "Retrievable Instrument",
                "asset_type": "stock",
                "exchange": "TEST",
                "currency": "USD",
            },
        )

        assert create_response.status_code == 201

        created = create_response.json()
        instrument_id = created["id"]

        response = client.get(
            f"/instruments/{instrument_id}"
        )

        assert response.status_code == 200

        data = response.json()

        assert data["id"] == instrument_id
        assert data["symbol"] == symbol
        assert data["name"] == "Retrievable Instrument"
        assert data["asset_type"] == "stock"

    finally:
        cleanup_instrument(symbol)


def test_get_instrument_not_found():
    response = client.get("/instruments/999999999")

    assert response.status_code == 404
    assert response.json()["detail"] == "Instrument not found."


def test_create_instrument_missing_required_field():
    symbol = unique_symbol()

    response = client.post(
        "/instruments/",
        json={
            "symbol": symbol,
            "name": "Incomplete Instrument",
        },
    )

    assert response.status_code == 422

    cleanup_instrument(symbol)


def test_create_instrument_empty_body():
    response = client.post(
        "/instruments/",
        json={},
    )

    assert response.status_code == 422


def test_get_instrument_invalid_id_type():
    response = client.get("/instruments/not-an-integer")

    assert response.status_code == 422