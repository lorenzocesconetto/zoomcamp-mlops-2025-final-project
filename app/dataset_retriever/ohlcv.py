import logging
import warnings

import pandas as pd
from ccxt import Exchange

from app.helpers.logger import build_logger

warnings.filterwarnings("ignore")

logger = build_logger(__name__)


class OHLCVDatasetRetriever:
    def __init__(
        self,
        exchange: Exchange,
        logger: logging.Logger = logger,
    ) -> None:
        """
        Initialize the Enhanced Bitcoin Dataset Creator

        Args:
            exchange (Exchange): CCXT exchange instance
            logger (logging.Logger): Logger instance for logging messages
        """
        self.logger = logger
        self.exchange = exchange

    def execute(
        self, symbol: str = "BTC/USDT", timeframe: str = "1m", limit: int = 1_000
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data from exchange or generate sample data

        Args:
            symbol (str): Trading symbol
            timeframe (str): Timeframe for data
            limit (int): Number of candles to fetch

        Returns:
            pd.DataFrame: OHLCV data with datetime index
        """
        ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        return df
