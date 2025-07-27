import logging
import time
import warnings
from typing import Optional, cast

import pandas as pd
from ccxt import Exchange

from app.domain.ports.dataset_retriever import DatasetRetriever
from app.helpers.logger import build_logger

warnings.filterwarnings("ignore")

logger = build_logger("OHLCVDatasetRetriever")


class OHLCVDatasetRetriever(DatasetRetriever):
    def __init__(
        self,
        exchange: Exchange,
        max_retries: int = 5,
        request_delay: int = 1,
        logger: logging.Logger = logger,
    ) -> None:
        """
        Initialize the Enhanced Bitcoin Dataset Creator

        Args:
            exchange (Exchange): CCXT exchange instance
            max_retries (int): Maximum number of retries for failed requests
            request_delay (int): Delay in seconds between requests to avoid rate limiting
            logger (logging.Logger): Logger instance for logging messages
        """
        self.logger = logger
        self.exchange = exchange
        self.max_retries = max_retries
        self.request_delay = request_delay

    def execute(
        self,
        *,
        symbol: str = "BTC/USDT",
        timeframe: str = "1m",
        start_date: str,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data with pagination support

        Args:
            symbol (str): Trading symbol
            timeframe (str): Timeframe for data
            limit (int): Number of candles per request (max 1000)
            start_date (str): Start date in format 'YYYY-MM-DD HH:MM:SS'
            end_date (str): End date in format 'YYYY-MM-DD HH:MM:SS'
        """
        # Parse start and end dates
        since_ts = self.exchange.parse8601(start_date)
        end_ts = self.exchange.parse8601(end_date) if end_date else self.exchange.milliseconds()
        return self._fetch_historical_data(symbol, timeframe, since_ts, end_ts)

    def _fetch_historical_data(
        self, symbol: str, timeframe: str, since_ts: int, end_ts: int
    ) -> pd.DataFrame:
        """
        Fetch historical data using pagination
        """
        batch_limit = 1000  # CCXT max limit for fetch_ohlcv
        all_ohlcv = []
        current_since = since_ts

        self.logger.info("Starting historical data fetch for %s", symbol)

        while current_since < end_ts:
            try:
                # Fetch batch with retry logic
                batch_data = self._fetch_batch_with_retry(
                    symbol, timeframe, current_since, batch_limit
                )

                if not batch_data:
                    self.logger.warning("No more data available")
                    break

                # Filter data that exceeds end_ts
                filtered_batch = [candle for candle in batch_data if candle[0] <= end_ts]
                all_ohlcv.extend(filtered_batch)

                # If we got less than requested, we've reached the end
                if len(batch_data) < batch_limit:
                    self.logger.info("Reached end of available data")
                    break

                # Update since timestamp to the timestamp after the last candle
                current_since = batch_data[-1][0] + 1

                self.logger.info(f"Fetched {len(batch_data)} candles. Total: {len(all_ohlcv)}")

                # Rate limiting to avoid getting banned
                time.sleep(self.request_delay)

            except Exception as e:
                self.logger.error(f"Error fetching batch: {e}")
                break

        self.logger.info(f"Completed fetch. Total candles: {len(all_ohlcv)}")
        return self._convert_to_dataframe(all_ohlcv)

    def _fetch_batch_with_retry(
        self, symbol: str, timeframe: str, since: int, batch_limit: int
    ) -> list[tuple[int, float, float, float, float, float]]:
        """Fetch a batch with retry logic"""

        for attempt in range(self.max_retries + 1):
            try:
                return cast(
                    list[tuple[int, float, float, float, float, float]],
                    self.exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=batch_limit),
                )
            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2**attempt)  # Exponential backoff
                else:
                    raise e
        return []

    def _convert_to_dataframe(self, ohlcv_data: list) -> pd.DataFrame:
        """Convert OHLCV data to pandas DataFrame"""
        if not ohlcv_data:
            return pd.DataFrame()

        df = pd.DataFrame(
            ohlcv_data, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        df = df.sort_index()  # Ensure chronological order

        # Remove duplicates that might occur at batch boundaries
        df = df[~df.index.duplicated(keep="first")]

        return df
