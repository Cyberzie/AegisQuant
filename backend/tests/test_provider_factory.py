import pytest
from app.providers.twelve_data import TwelveDataProvider
from app.providers.factory import get_market_data_provider
from app.providers.mock import MockMarketDataProvider

def test_get_mock_provider():
    provider = get_market_data_provider("mock")

    assert isinstance(provider, MockMarketDataProvider)

def test_get_mock_provider_from_settings(monkeypatch):
    monkeypatch.setattr(
        "app.providers.factory.settings.MARKET_DATA_PROVIDER",
        "mock",
    )

    provider = get_market_data_provider()

    assert isinstance(provider, MockMarketDataProvider)

def test_unknown_provider():
    with pytest.raises(ValueError):
        get_market_data_provider("unknown")

def test_get_twelve_data_provider():
    provider = get_market_data_provider("twelve_data")
    assert isinstance(provider, TwelveDataProvider)

def test_get_provider_from_twelve_data_settings(monkeypatch):
    monkeypatch.setattr(
        "app.providers.factory.settings.MARKET_DATA_PROVIDER",
        "twelve_data",
    )

    provider = get_market_data_provider()

    assert isinstance(provider, TwelveDataProvider)