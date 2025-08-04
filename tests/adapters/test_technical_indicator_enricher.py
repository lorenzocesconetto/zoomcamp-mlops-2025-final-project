import numpy as np
import pandas as pd
import pytest

from app.adapters.preprocessors.techinical_indicator_enricher import (
    TechnicalIndicatorEnricherPreProcessor,
)


class TestTechnicalIndicatorEnricherPreProcessor:
    """Test suite for TechnicalIndicatorEnricherPreProcessor"""

    def test_calculate_rsi_basic(self, sample_ohlcv_data: pd.DataFrame) -> None:
        """Test basic RSI calculation"""
        prices = sample_ohlcv_data["close"]
        rsi = TechnicalIndicatorEnricherPreProcessor.calculate_rsi(prices, period=14)

        assert isinstance(rsi, pd.Series)
        assert len(rsi) == len(prices)
        # RSI should be between 0 and 100
        assert (rsi.dropna() >= 0).all()
        assert (rsi.dropna() <= 100).all()
        # First 13 values should be NaN due to rolling window
        assert rsi.iloc[:13].isna().all()

    def test_calculate_rsi_different_periods(self, sample_ohlcv_data: pd.DataFrame) -> None:
        """Test RSI calculation with different periods"""
        prices = sample_ohlcv_data["close"]

        rsi_7 = TechnicalIndicatorEnricherPreProcessor.calculate_rsi(prices, period=7)
        rsi_21 = TechnicalIndicatorEnricherPreProcessor.calculate_rsi(prices, period=21)

        # Different periods should have different NaN counts
        assert rsi_7.iloc[:6].isna().all()
        assert not rsi_7.iloc[7:].isna().all()
        assert rsi_21.iloc[:20].isna().all()

    def test_calculate_macd_basic(self, sample_ohlcv_data: pd.DataFrame) -> None:
        """Test basic MACD calculation"""
        prices = sample_ohlcv_data["close"]
        macd, signal, histogram = TechnicalIndicatorEnricherPreProcessor.calculate_macd(prices)

        # All should be Series of same length
        assert all(isinstance(x, pd.Series) for x in [macd, signal, histogram])
        assert len(macd) == len(signal) == len(histogram) == len(prices)

        # Histogram should equal macd - signal
        pd.testing.assert_series_equal(histogram, macd - signal)

    def test_calculate_macd_custom_parameters(self, sample_ohlcv_data: pd.DataFrame) -> None:
        """Test MACD calculation with custom parameters"""
        prices = sample_ohlcv_data["close"]
        macd, signal, histogram = TechnicalIndicatorEnricherPreProcessor.calculate_macd(
            prices, fast=5, slow=10, signal=3
        )

        assert isinstance(macd, pd.Series)
        assert isinstance(signal, pd.Series)
        assert isinstance(histogram, pd.Series)

    def test_calculate_bollinger_bands(self, sample_ohlcv_data: pd.DataFrame) -> None:
        """Test Bollinger Bands calculation"""
        prices = sample_ohlcv_data["close"]
        upper, lower, sma = TechnicalIndicatorEnricherPreProcessor.calculate_bollinger_bands(
            prices
        )

        # All should be Series
        assert all(isinstance(x, pd.Series) for x in [upper, lower, sma])
        assert len(upper) == len(lower) == len(sma) == len(prices)

        # Upper should be >= SMA >= Lower (where not NaN)
        valid_idx = ~(upper.isna() | lower.isna() | sma.isna())
        assert (upper[valid_idx] >= sma[valid_idx]).all()
        assert (sma[valid_idx] >= lower[valid_idx]).all()

    def test_calculate_bollinger_bands_custom_params(
        self, sample_ohlcv_data: pd.DataFrame
    ) -> None:
        """Test Bollinger Bands with custom parameters"""
        prices = sample_ohlcv_data["close"]
        upper, lower, sma = TechnicalIndicatorEnricherPreProcessor.calculate_bollinger_bands(
            prices, period=10, std_dev=1
        )

        # First 9 values should be NaN due to rolling window
        assert upper.iloc[:9].isna().all()
        assert lower.iloc[:9].isna().all()
        assert sma.iloc[:9].isna().all()

    def test_calculate_stochastic(self, sample_ohlcv_data: pd.DataFrame) -> None:
        """Test Stochastic Oscillator calculation"""
        high, low, close = (
            sample_ohlcv_data["high"],
            sample_ohlcv_data["low"],
            sample_ohlcv_data["close"],
        )
        k_percent, d_percent = TechnicalIndicatorEnricherPreProcessor.calculate_stochastic(
            high, low, close
        )

        assert isinstance(k_percent, pd.Series)
        assert isinstance(d_percent, pd.Series)
        assert len(k_percent) == len(d_percent) == len(close)

        # K% should be between 0 and 100
        valid_k = k_percent.dropna()
        assert (valid_k >= 0).all()
        assert (valid_k <= 100).all()

    def test_calculate_williams_r(self, sample_ohlcv_data: pd.DataFrame) -> None:
        """Test Williams %R calculation"""
        high, low, close = (
            sample_ohlcv_data["high"],
            sample_ohlcv_data["low"],
            sample_ohlcv_data["close"],
        )
        williams_r = TechnicalIndicatorEnricherPreProcessor.calculate_williams_r(high, low, close)

        assert isinstance(williams_r, pd.Series)
        assert len(williams_r) == len(close)

        # Williams %R should be between -100 and 0
        valid_values = williams_r.dropna()
        assert (valid_values >= -100).all()
        assert (valid_values <= 0).all()

    def test_calculate_atr(self, sample_ohlcv_data: pd.DataFrame) -> None:
        """Test Average True Range calculation"""
        high, low, close = (
            sample_ohlcv_data["high"],
            sample_ohlcv_data["low"],
            sample_ohlcv_data["close"],
        )
        atr = TechnicalIndicatorEnricherPreProcessor.calculate_atr(high, low, close)

        assert isinstance(atr, pd.Series)
        assert len(atr) == len(close)

        # ATR should be positive
        valid_atr = atr.dropna()
        assert (valid_atr >= 0).all()

    def test_calculate_mfi(self, sample_ohlcv_data: pd.DataFrame) -> None:
        """Test Money Flow Index calculation"""
        high, low, close, volume = (
            sample_ohlcv_data["high"],
            sample_ohlcv_data["low"],
            sample_ohlcv_data["close"],
            sample_ohlcv_data["volume"],
        )
        mfi = TechnicalIndicatorEnricherPreProcessor.calculate_mfi(high, low, close, volume)

        assert isinstance(mfi, pd.Series)
        assert len(mfi) == len(close)

        # MFI should be between 0 and 100
        valid_mfi = mfi.dropna()
        assert (valid_mfi >= 0).all()
        assert (valid_mfi <= 100).all()

    def test_add_technical_indicators_comprehensive(self, sample_ohlcv_data: pd.DataFrame) -> None:
        """Test comprehensive technical indicators addition"""
        original_cols = set(sample_ohlcv_data.columns)
        enriched_df = TechnicalIndicatorEnricherPreProcessor.execute(sample_ohlcv_data.copy())

        # Check that original columns are preserved
        assert original_cols.issubset(set(enriched_df.columns))

        # Check that new indicators are added
        expected_indicators = [
            "price_change_1",
            "price_change_5",
            "price_change_15",
            "price_change_30",
            "sma_5",
            "sma_10",
            "sma_20",
            "sma_50",
            "ema_5",
            "ema_10",
            "ema_20",
            "price_sma_5_ratio",
            "price_sma_20_ratio",
            "price_ema_10_ratio",
            "volatility_10",
            "volatility_20",
            "price_range",
            "volume_sma_10",
            "volume_ratio",
            "volume_change",
            "vwap",
            "price_vwap_ratio",
            "rsi_14",
            "rsi_7",
            "macd",
            "macd_signal",
            "macd_histogram",
            "bb_upper",
            "bb_lower",
            "bb_width",
            "bb_position",
            "stoch_k",
            "stoch_d",
            "williams_r",
            "atr",
            "mfi",
            "hour",
            "day_of_week",
            "is_weekend",
            "hour_sin",
            "hour_cos",
            "dow_sin",
            "dow_cos",
            "support_level",
            "resistance_level",
            "support_distance",
            "resistance_distance",
            "bid_ask_spread",
            "spread_ratio",
        ]

        for indicator in expected_indicators:
            assert indicator in enriched_df.columns, f"Missing indicator: {indicator}"

    def test_add_technical_indicators_data_types(self, sample_ohlcv_data: pd.DataFrame) -> None:
        """Test that technical indicators have correct data types"""
        enriched_df = TechnicalIndicatorEnricherPreProcessor.execute(sample_ohlcv_data.copy())

        # Time-based features should be integers
        assert enriched_df["hour"].dtype in [np.int64, np.int32]
        assert enriched_df["day_of_week"].dtype in [np.int64, np.int32]
        assert enriched_df["is_weekend"].dtype in [np.int64, np.int32]

        # Cyclical features should be float
        assert enriched_df["hour_sin"].dtype == np.float64
        assert enriched_df["hour_cos"].dtype == np.float64

    def test_add_technical_indicators_bollinger_position(
        self, sample_ohlcv_data: pd.DataFrame
    ) -> None:
        """Test Bollinger Band position calculation"""
        enriched_df = TechnicalIndicatorEnricherPreProcessor.execute(sample_ohlcv_data.copy())

        # BB position can go outside 0-1 range when price is outside the bands
        # This is expected behavior - just check that the calculation works
        bb_position = enriched_df["bb_position"].dropna()
        assert len(bb_position) > 0  # Should have some valid values
        assert isinstance(bb_position.iloc[0], (int, float))  # Should be numeric

    def test_add_technical_indicators_price_ratios(self, sample_ohlcv_data: pd.DataFrame) -> None:
        """Test price ratio calculations"""
        enriched_df = TechnicalIndicatorEnricherPreProcessor.execute(sample_ohlcv_data.copy())

        # Price ratios should be positive
        price_ratios = [
            "price_sma_5_ratio",
            "price_sma_20_ratio",
            "price_ema_10_ratio",
            "price_vwap_ratio",
        ]
        for ratio in price_ratios:
            valid_values = enriched_df[ratio].dropna()
            assert (valid_values > 0).all(), f"{ratio} should be positive"

    def test_add_technical_indicators_cyclical_encoding(
        self, sample_ohlcv_data: pd.DataFrame
    ) -> None:
        """Test cyclical encoding of time features"""
        enriched_df = TechnicalIndicatorEnricherPreProcessor.execute(sample_ohlcv_data.copy())

        # Cyclical features should be between -1 and 1
        cyclical_features = ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]
        for feature in cyclical_features:
            values = enriched_df[feature]
            assert (values >= -1).all()
            assert (values <= 1).all()

        # Sin^2 + Cos^2 should equal 1 (approximately)
        hour_circle = enriched_df["hour_sin"] ** 2 + enriched_df["hour_cos"] ** 2
        dow_circle = enriched_df["dow_sin"] ** 2 + enriched_df["dow_cos"] ** 2

        np.testing.assert_array_almost_equal(hour_circle, 1.0, decimal=10)
        np.testing.assert_array_almost_equal(dow_circle, 1.0, decimal=10)

    def test_add_technical_indicators_empty_dataframe(self) -> None:
        """Test handling of empty DataFrame"""
        empty_df = pd.DataFrame()

        # Should handle empty DataFrame gracefully
        with pytest.raises((KeyError, AttributeError)):
            TechnicalIndicatorEnricherPreProcessor.execute(empty_df)

    def test_add_technical_indicators_preserves_index(
        self, sample_ohlcv_data: pd.DataFrame
    ) -> None:
        """Test that the original index is preserved"""
        original_index = sample_ohlcv_data.index
        enriched_df = TechnicalIndicatorEnricherPreProcessor.execute(sample_ohlcv_data.copy())

        pd.testing.assert_index_equal(enriched_df.index, original_index)

    def test_support_resistance_levels(self, sample_ohlcv_data: pd.DataFrame) -> None:
        """Test support and resistance level calculations"""
        enriched_df = TechnicalIndicatorEnricherPreProcessor.execute(sample_ohlcv_data.copy())

        # Support should be <= close, resistance should be >= close (where not NaN)
        valid_idx = ~(enriched_df["support_level"].isna() | enriched_df["resistance_level"].isna())

        support_valid = (
            enriched_df.loc[valid_idx, "support_level"] <= enriched_df.loc[valid_idx, "close"]
        )
        resistance_valid = (
            enriched_df.loc[valid_idx, "resistance_level"] >= enriched_df.loc[valid_idx, "close"]
        )

        assert support_valid.all()
        assert resistance_valid.all()
