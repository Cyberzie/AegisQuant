import pytest
import requests
from datetime import datetime
from app.providers.twelve_data import TwelveDataProvider

def test_twelve_data_provider_parses_response(monkeypatch):
    payload = {
        "meta": {
            "symbol": "AAPL",
            "interval": "1day",
        },
        "values": [
            {
                "datetime": "2026-08-04",
                "open": "240.00",
                "high": "245.00",
                "low": "238.00",
                "close": "243.00",
                "volume": "1500000",
            },
            {
                "datetime": "2026-08-05",
                "open": "243.00",
                "high": "248.00",
                "low": "241.00",
                "close": "246.00",
                "volume": "1600000",
            },
        ],
    }

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return payload

    def fake_get(*args, **kwargs):
        return FakeResponse()

    monkeypatch.setattr(
        "app.providers.twelve_data.requests.get",
        fake_get,
    )

    provider = TwelveDataProvider(
        api_key="test-key",
        base_url="https://example.com",
        interval="1day",
    )

    rows = provider.get_market_data(
        symbol="AAPL",
        start=datetime(2026, 8, 4),
        end=datetime(2026, 8, 5),
    )

    assert len(rows) == 2

    assert rows[0]["symbol"] == "AAPL"
    assert rows[0]["open"] == 240.0
    assert rows[0]["high"] == 245.0
    assert rows[0]["low"] == 238.0
    assert rows[0]["close"] == 243.0
    assert rows[0]["volume"] == 1500000.0

    assert rows[0]["timestamp"] == datetime(
        2026,
        8,
        4,
    )

def test_twelve_data_provider_api_error(monkeypatch):
    payload = {
        "status": "error",
        "message": "Invalid API key.",
    }

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return payload

    def fake_get(*args, **kwargs):
        return FakeResponse()

    monkeypatch.setattr(
        "app.providers.twelve_data.requests.get",
        fake_get,
    )

    provider = TwelveDataProvider(
        api_key="bad-key",
        base_url="https://example.com",
        interval="1day",
    )

    try:
        provider.get_market_data("AAPL")
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "Invalid API key." in str(exc)

def test_twelve_data_provider_empty_response(monkeypatch):
    payload = {
        "meta": {
            "symbol": "AAPL",
            "interval": "1day",
        },
        "values": [],
    }

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return payload

    def fake_get(*args, **kwargs):
        return FakeResponse()

    monkeypatch.setattr(
        "app.providers.twelve_data.requests.get",
        fake_get,
    )

    provider = TwelveDataProvider(
        api_key="test-key",
        base_url="https://example.com",
        interval="1day",
    )

    rows = provider.get_market_data("AAPL")

    assert rows == []

def test_twelve_data_provider_http_error(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            raise requests.HTTPError("429 Too Many Requests")

        def json(self):
            return {}

    def fake_get(*args, **kwargs):
        return FakeResponse()

    monkeypatch.setattr(
        "app.providers.twelve_data.requests.get",
        fake_get,
    )

    provider = TwelveDataProvider(
        api_key="test-key",
        base_url="https://example.com",
        interval="1day",
    )

    with pytest.raises(requests.HTTPError):
        provider.get_market_data("AAPL")
