from abc import ABC, abstractmethod

import pandas as pd


class Preprocessor(ABC):
    """Port for preprocessing data"""

    @classmethod
    @abstractmethod
    def execute(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to the dataset

        Args:
            df: Input DataFrame with OHLCV data

        Returns:
            DataFrame enriched with technical indicators
        """
