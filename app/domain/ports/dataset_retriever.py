from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd


class DatasetRetriever(ABC):
    """Port for retrieving dataset information"""

    @abstractmethod
    def execute(
        self,
        *,
        symbol: str = "BTC/USDT",
        timeframe: str = "1m",
        start_date: str,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Retrieve historical OHLCV data

        Args:
            symbol: Trading symbol
            timeframe: Timeframe for data
            start_date: Start date in format 'YYYY-MM-DD HH:MM:SS'
            end_date: End date in format 'YYYY-MM-DD HH:MM:SS'

        Returns:
            DataFrame with OHLCV data
        """
        pass
