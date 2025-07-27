from abc import ABC, abstractmethod
from typing import Tuple

import pandas as pd


class SplitStrategy(ABC):
    """Port for data splitting strategies"""

    @staticmethod
    @abstractmethod
    def execute(
        dataset: pd.DataFrame, test_size: float
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split dataset into training and testing sets

        Args:
            dataset: Input dataset
            test_size: Proportion of dataset to include in test split

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        pass
