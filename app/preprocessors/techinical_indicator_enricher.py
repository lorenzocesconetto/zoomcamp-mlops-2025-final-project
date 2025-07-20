import numpy as np
import pandas as pd


class TechnicalIndicatorEnricherPreProcessor:
    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def calculate_macd(
        prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD indicator"""
        exp1 = prices.ewm(span=fast).mean()
        exp2 = prices.ewm(span=slow).mean()
        macd = exp1 - exp2
        macd_signal = macd.ewm(span=signal).mean()
        macd_histogram = macd - macd_signal
        return macd, macd_signal, macd_histogram

    @staticmethod
    def calculate_bollinger_bands(
        prices: pd.Series, period: int = 20, std_dev: int = 2
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, lower_band, sma

    @staticmethod
    def calculate_stochastic(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        k_period: int = 14,
        d_period: int = 3,
    ) -> tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator"""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        return k_percent, d_percent

    @staticmethod
    def calculate_williams_r(
        high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
    ) -> pd.Series:
        """Calculate Williams %R"""
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
        return williams_r

    @staticmethod
    def calculate_atr(
        high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
    ) -> pd.Series:
        """Calculate Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr

    @staticmethod
    def calculate_mfi(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        period: int = 14,
    ) -> pd.Series:
        """Calculate Money Flow Index"""
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume

        positive_flow = (
            money_flow.where(typical_price > typical_price.shift(1), 0)
            .rolling(window=period)
            .sum()
        )
        negative_flow = (
            money_flow.where(typical_price < typical_price.shift(1), 0)
            .rolling(window=period)
            .sum()
        )

        money_ratio = positive_flow / negative_flow
        mfi = 100 - (100 / (1 + money_ratio))
        return mfi

    @classmethod
    def add_technical_indicators(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive technical indicators to the dataset"""
        # Price-based features
        df["price_change_1"] = df["close"].pct_change(1)
        df["price_change_5"] = df["close"].pct_change(5)
        df["price_change_15"] = df["close"].pct_change(15)
        df["price_change_30"] = df["close"].pct_change(30)

        # Moving averages
        df["sma_5"] = df["close"].rolling(window=5).mean()
        df["sma_10"] = df["close"].rolling(window=10).mean()
        df["sma_20"] = df["close"].rolling(window=20).mean()
        df["sma_50"] = df["close"].rolling(window=50).mean()

        df["ema_5"] = df["close"].ewm(span=5).mean()
        df["ema_10"] = df["close"].ewm(span=10).mean()
        df["ema_20"] = df["close"].ewm(span=20).mean()

        # Price relative to moving averages
        df["price_sma_5_ratio"] = df["close"] / df["sma_5"]
        df["price_sma_20_ratio"] = df["close"] / df["sma_20"]
        df["price_ema_10_ratio"] = df["close"] / df["ema_10"]

        # Volatility measures
        df["volatility_10"] = df["close"].rolling(window=10).std()
        df["volatility_20"] = df["close"].rolling(window=20).std()
        df["price_range"] = (df["high"] - df["low"]) / df["close"]

        # Volume features
        df["volume_sma_10"] = df["volume"].rolling(window=10).mean()
        df["volume_ratio"] = df["volume"] / df["volume_sma_10"]
        df["volume_change"] = df["volume"].pct_change(1)

        # Volume weighted average price (VWAP)
        df["vwap"] = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()
        df["price_vwap_ratio"] = df["close"] / df["vwap"]

        # Technical indicators
        df["rsi_14"] = cls.calculate_rsi(df["close"], 14)
        df["rsi_7"] = cls.calculate_rsi(df["close"], 7)

        macd, macd_signal, macd_hist = cls.calculate_macd(df["close"])
        df["macd"] = macd
        df["macd_signal"] = macd_signal
        df["macd_histogram"] = macd_hist

        bb_upper, bb_lower, bb_sma = cls.calculate_bollinger_bands(df["close"])
        df["bb_upper"] = bb_upper
        df["bb_lower"] = bb_lower
        df["bb_width"] = (bb_upper - bb_lower) / bb_sma
        df["bb_position"] = (df["close"] - bb_lower) / (bb_upper - bb_lower)

        stoch_k, stoch_d = cls.calculate_stochastic(df["high"], df["low"], df["close"])
        df["stoch_k"] = stoch_k
        df["stoch_d"] = stoch_d

        df["williams_r"] = cls.calculate_williams_r(df["high"], df["low"], df["close"])
        df["atr"] = cls.calculate_atr(df["high"], df["low"], df["close"])
        df["mfi"] = cls.calculate_mfi(df["high"], df["low"], df["close"], df["volume"])

        # Time-based features
        df["hour"] = df.index.hour
        df["day_of_week"] = df.index.dayofweek
        df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)

        # Cyclical encoding for time features
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

        # TODO: Improve this... Support and resistance levels (simplified)
        df["support_level"] = df["low"].rolling(window=20).min()
        df["resistance_level"] = df["high"].rolling(window=20).max()
        df["support_distance"] = (df["close"] - df["support_level"]) / df["close"]
        df["resistance_distance"] = (df["resistance_level"] - df["close"]) / df["close"]

        # TODO: Fix this... Order book simulation (since we don't have real order book data)
        df["bid_ask_spread"] = df["high"] - df["low"]  # Simplified spread
        df["spread_ratio"] = df["bid_ask_spread"] / df["close"]

        return df
