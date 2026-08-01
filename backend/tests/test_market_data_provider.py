from datetime import datetime

from app.providers.mock import MockMarketDataProvider


def test_mock_market_data_provider():

    provider = MockMarketDataProvider()

    rows = provider.get_market_data(
        symbol="TEST",
        start=datetime(2026, 8, 1, 10, 0, 0),
    )

    assert len(rows) == 3

    assert rows[0]["symbol"] == "TEST"

    assert rows[0]["open"] == 230.0
    assert rows[0]["high"] == 235.0
    assert rows[0]["low"] == 228.0
    assert rows[0]["close"] == 233.0

    assert rows[0]["low"] <= rows[0]["open"] <= rows[0]["high"]
    assert rows[0]["low"] <= rows[0]["close"] <= rows[0]["high"]

    assert rows[0]["volume"] >= 0