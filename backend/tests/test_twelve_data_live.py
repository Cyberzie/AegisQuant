import pytest

from app.providers.twelve_data import TwelveDataProvider


@pytest.mark.live
def test_twelve_data_live_connection():
    provider = TwelveDataProvider()

    rows = provider.get_market_data(
        symbol="AAPL",
    )

    assert rows, "Twelve Data returned no market data."

    print(f"\nReceived {len(rows)} rows from Twelve Data.")
    print("First row:")
    print(rows[0])