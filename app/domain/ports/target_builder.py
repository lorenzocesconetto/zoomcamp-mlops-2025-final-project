from abc import ABC, abstractmethod
from typing import List

import pandas as pd


class TargetBuilder(ABC):
    """Port for building target variables"""

    @abstractmethod
    def execute(self, df: pd.DataFrame) -> List[int]:
        """
        Create target variable from DataFrame

        Args:
            df: Input DataFrame with OHLCV data

        Returns:
            List of target values
        """
        pass
