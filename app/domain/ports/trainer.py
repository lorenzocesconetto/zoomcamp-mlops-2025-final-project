from abc import ABC, abstractmethod

import pandas as pd


class Trainer(ABC):
    """Port for model training"""

    @abstractmethod
    def execute(
        self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series
    ) -> None:
        """
        Train models and evaluate performance

        Args:
            X_train: Training features
            X_test: Testing features
            y_train: Training targets
            y_test: Testing targets
        """
        pass
