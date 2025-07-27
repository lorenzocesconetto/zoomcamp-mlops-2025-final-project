from typing import Generator
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_ohlcv_data() -> Generator[pd.DataFrame, None, None]:
    """Sample OHLCV DataFrame for testing"""
    dates = pd.date_range("2024-01-01", periods=100, freq="1min")
    np.random.seed(42)  # For reproducible tests

    data = {
        "open": 50000 + np.random.randn(100) * 100,
        "high": 50100 + np.random.randn(100) * 100,
        "low": 49900 + np.random.randn(100) * 100,
        "close": 50000 + np.random.randn(100) * 100,
        "volume": 1000 + np.random.randn(100) * 100,
    }

    df = pd.DataFrame(data, index=dates)
    # Ensure high >= max(open, close) and low <= min(open, close)
    df["high"] = np.maximum(df["high"], np.maximum(df["open"], df["close"]))
    df["low"] = np.minimum(df["low"], np.minimum(df["open"], df["close"]))

    yield df


@pytest.fixture
def sample_enriched_data() -> Generator[pd.DataFrame, None, None]:
    """Sample DataFrame with technical indicators for testing"""
    dates = pd.date_range("2024-01-01", periods=100, freq="1min")
    np.random.seed(42)

    data = {
        "open": 50000 + np.random.randn(100) * 100,
        "high": 50100 + np.random.randn(100) * 100,
        "low": 49900 + np.random.randn(100) * 100,
        "close": 50000 + np.random.randn(100) * 100,
        "volume": 1000 + np.random.randn(100) * 100,
        "target": np.random.choice([0, 1, 2], size=100),
    }

    df = pd.DataFrame(data, index=dates)
    df["high"] = np.maximum(df["high"], np.maximum(df["open"], df["close"]))
    df["low"] = np.minimum(df["low"], np.minimum(df["open"], df["close"]))

    yield df


@pytest.fixture
def mock_exchange() -> Generator[Mock, None, None]:
    """Mock CCXT exchange for testing"""
    exchange = Mock()
    exchange.parse8601.side_effect = lambda x: pd.Timestamp(x).timestamp() * 1000
    exchange.milliseconds.return_value = pd.Timestamp("2024-01-02").timestamp() * 1000
    yield exchange


@pytest.fixture
def mock_logger() -> Generator[Mock, None, None]:
    """Mock logger for testing"""
    yield Mock()
