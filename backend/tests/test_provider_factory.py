import pytest

from app.providers.factory import get_market_data_provider
from app.providers.mock import MockMarketDataProvider


def test_get_mock_provider():
    provider = get_market_data_provider("mock")

    assert isinstance(provider, MockMarketDataProvider)


def test_unknown_provider():
    with pytest.raises(ValueError):
        get_market_data_provider("unknown")