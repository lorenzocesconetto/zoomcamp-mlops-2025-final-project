from typing import Generator
from unittest.mock import Mock

import pandas as pd
import pytest

from app.adapters.target_builders.tp_sl import TPSLTargetBuilder


class TestTPSLTargetBuilder:
    """Test suite for TPSLTargetBuilder"""

    @pytest.fixture
    def target_builder(
        self, mock_exchange: Mock, mock_logger: Mock
    ) -> Generator[TPSLTargetBuilder, None, None]:
        """Create TPSLTargetBuilder instance for testing"""
        yield TPSLTargetBuilder(
            exchange=mock_exchange,
            threshold_percent=1.0,
            time_window_minutes=10,
            logger=mock_logger,
        )

    @pytest.fixture
    def sample_price_data(self) -> Generator[pd.DataFrame, None, None]:
        """Sample price data with predictable patterns"""
        dates = pd.date_range("2024-01-01", periods=20, freq="1min")

        # Create data where price goes up 2% then down 2%
        base_price = 50000.0
        prices: list[float] = [base_price]

        for i in range(1, 20):
            if i < 5:
                # Price goes up
                prices.append(base_price * (1 + 0.005 * i))  # Gradual increase
            elif i == 5:
                prices.append(base_price * 1.025)  # 2.5% up (hits threshold)
            elif i < 10:
                # Price stays elevated
                prices.append(base_price * 1.02)
            elif i < 15:
                # Price goes down
                prices.append(base_price * (1.02 - 0.003 * (i - 10)))
            else:
                # Price goes down further
                prices.append(base_price * 0.97)  # 3% down

        yield pd.DataFrame(
            {
                "open": prices,
                "high": [p * 1.001 for p in prices],
                "low": [p * 0.999 for p in prices],
                "close": prices,
                "volume": [1000] * 20,
            },
            index=dates,
        )

    def test_init(self, mock_exchange: Mock, mock_logger: Mock) -> None:
        """Test TPSLTargetBuilder initialization"""
        builder = TPSLTargetBuilder(
            exchange=mock_exchange,
            threshold_percent=2.0,
            time_window_minutes=15,
            logger=mock_logger,
        )

        assert builder.exchange == mock_exchange
        assert builder.threshold_percent == 2.0
        assert builder.time_window_minutes == 15
        assert builder.logger == mock_logger

    def test_execute_basic_functionality(
        self, target_builder: TPSLTargetBuilder, sample_price_data: pd.DataFrame
    ) -> None:
        """Test basic target building functionality"""
        targets = target_builder.execute(sample_price_data)

        assert isinstance(targets, list)
        assert len(targets) == len(sample_price_data)

        # All targets should be 0, 1, or 2
        assert all(t in [0, 1, 2] for t in targets)

    def test_execute_upward_threshold_hit(self, mock_exchange: Mock, mock_logger: Mock) -> None:
        """Test when upward threshold is hit first"""
        # Create data where price goes up by threshold first
        dates = pd.date_range("2024-01-01", periods=10, freq="1min")
        base_price = 50000

        prices = [
            base_price,  # Starting price
            base_price * 1.005,  # +0.5%
            base_price * 1.015,  # +1.5% (hits 1% threshold)
            base_price * 1.02,  # +2%
            base_price * 0.99,  # -1% (down threshold hit later)
        ] + [base_price] * 5

        df = pd.DataFrame({"close": prices}, index=dates)

        builder = TPSLTargetBuilder(mock_exchange, 1.0, 10, mock_logger)
        targets = builder.execute(df)

        # First candle should predict upward movement (target = 1)
        assert targets[0] == 1

    def test_execute_empty_dataframe(self, target_builder: TPSLTargetBuilder) -> None:
        """Test with empty DataFrame"""
        empty_df = pd.DataFrame()

        targets = target_builder.execute(empty_df)

        assert targets == []
