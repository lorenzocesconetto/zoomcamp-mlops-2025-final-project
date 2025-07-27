import logging
from typing import TypedDict

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score, classification_report

from app.domain.ports.trainer import Trainer
from app.helpers.logger import build_logger

logger = build_logger(__name__)


class Model(TypedDict):
    name: str
    model: ClassifierMixin


class SKLearnModelTrainer(Trainer):
    # TODO: Perform hyperparameter search and tuning (grid search, random search, etc.)
    def __init__(self, models: list[Model], logger: logging.Logger = logger):
        self.models = models
        self.logger = logger

    def execute(
        self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series
    ) -> None:
        """
        Train baseline models and return results
        """

        for model in self.models:
            self.logger.info("Training %s...", model["name"])
            model["model"].fit(X_train, y_train)

            y_pred = model["model"].predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            self.logger.info("%s Accuracy: %.4f", model["name"], accuracy)
            self.logger.info(
                "%s Classification Report:\n %s",
                model["name"],
                classification_report(y_test, y_pred, zero_division=np.nan),
            )
