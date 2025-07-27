import pandas as pd

from app.domain.ports.split_strategy import SplitStrategy


class OutOfTimeSplitStrategy(SplitStrategy):
    @staticmethod
    def execute(
        dataset: pd.DataFrame, test_size: float
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare dataset for machine learning
        """
        # Separate features and target
        feature_cols = [col for col in dataset.columns if col != "target"]
        X = dataset[feature_cols]
        y = dataset["target"]

        size = len(dataset)
        train_size = int(size * (1 - test_size))
        # Split the data
        X_train, X_test, y_train, y_test = (
            X[:train_size],
            X[train_size:],
            y[:train_size],
            y[train_size:],
        )

        return X_train, X_test, y_train, y_test
