import logging
import warnings

import pandas as pd
from ccxt import Exchange

from app.helpers.logger import build_logger

warnings.filterwarnings("ignore")

logger = build_logger(__name__)


class TPSLTargetBuilder:
    def __init__(
        self,
        exchange: Exchange,
        threshold_percent: float = 1.0,
        time_window_minutes: int = 15,
        logger: logging.Logger = logger,
    ) -> None:
        """
        Initialize the Enhanced Bitcoin Dataset Creator

        Args:
            exchange (Exchange): CCXT exchange instance (default is Binance)
            threshold_percent (float): Percentage threshold for price movement (e.g., 1.0 for 1%)
            time_window_minutes (int): Time window to check for threshold breach in minutes
            logger (logging.Logger): Logger instance for logging messages
        """
        self.threshold_percent = threshold_percent
        self.time_window_minutes = time_window_minutes
        self.logger = logger
        self.exchange = exchange

    def execute(self, df: pd.DataFrame) -> list[int]:
        """
        Create target variable based on whether price goes up/down by threshold% first

        Returns:
            - 0: Price went down by threshold% first
            - 1: Price went up by threshold% first
            - 2: Neither threshold was reached within time window, or both hit at the same time
        """
        targets = []

        for i in range(len(df)):
            current_price = df.iloc[i]["close"]

            # Look ahead within time window
            end_idx = min(i + self.time_window_minutes, len(df))
            future_prices = df.iloc[i:end_idx]["close"]

            # Calculate percentage changes from current price
            pct_changes = (future_prices - current_price) / current_price * 100

            # Check which threshold is hit first
            up_threshold_hit = pct_changes >= self.threshold_percent
            down_threshold_hit = pct_changes <= -self.threshold_percent

            up_first_idx = up_threshold_hit.idxmax() if up_threshold_hit.any() else None
            down_first_idx = down_threshold_hit.idxmax() if down_threshold_hit.any() else None

            if up_first_idx is not None and down_first_idx is not None:
                # Both thresholds hit, see which came first
                up_first_pos = future_prices.index.get_loc(up_first_idx)
                down_first_pos = future_prices.index.get_loc(down_first_idx)

                if up_first_pos < down_first_pos:
                    targets.append(1)  # Up first
                elif down_first_pos < up_first_pos:
                    targets.append(0)  # Down first
                else:
                    targets.append(2)  # Both hit at the same time
            elif up_first_idx is not None:
                targets.append(1)  # Only up threshold hit
            elif down_first_idx is not None:
                targets.append(0)  # Only down threshold hit
            else:
                targets.append(2)  # Neither threshold hit

        return targets
