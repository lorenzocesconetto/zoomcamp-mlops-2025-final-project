import logging
from typing import TypedDict

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

from app.helpers.logger import build_logger

logger = build_logger(__name__)


class Model(TypedDict):
    name: str
    model: object


class SKLearnModelTrainer:
    # TODO: Perform hyperparameter search and tuning (grid search, random search, etc.)
    # TODO: Add MLFlow
    def __init__(self, models: list[Model], logger: logging.Logger = logger):
        self.models = models
        self.logger = logger

    def execute(
        self, X_train: np.ndarray, X_test: np.ndarray, y_train: pd.Series, y_test: pd.Series
    ) -> None:
        """
        Train baseline models and return results
        """

        for model in self.models:
            self.logger.info(f"\nTraining {model['name']}...")
            model["model"].fit(X_train, y_train)

            y_pred = model["model"].predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            self.logger.info(f"{model['name']} Accuracy: {accuracy:.4f}")
            self.logger.info(f"{model['name']} Classification Report:")
            self.logger.info(classification_report(y_test, y_pred))
