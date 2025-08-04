#!/usr/bin/env python3
"""
SageMaker Model Evaluation Service for Crypto Price Prediction
Domain service that handles model evaluation for the ML pipeline.
"""

import argparse
import json
import logging
import os
import pickle
import sys
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EvaluationService:
    """Domain service for evaluating crypto prediction models."""

    def load_model(self, model_path: str) -> Tuple[BaseEstimator, Optional[List[str]]]:
        """Load the trained model and feature names."""
        logger.info("Loading model from %s", model_path)

        model_file = os.path.join(model_path, "model.pkl")
        feature_names_file = os.path.join(model_path, "feature_names.pkl")

        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file not found: {model_file}")

        with open(model_file, "rb") as f:
            model: BaseEstimator = pickle.load(f)

        feature_names: Optional[List[str]] = None
        if os.path.exists(feature_names_file):
            with open(feature_names_file, "rb") as f:
                feature_names = pickle.load(f)

        logger.info("Model loaded successfully")
        return model, feature_names

    def execute(self, model_path: str, test_path: str, output_path: str) -> None:
        """Evaluate the model performance."""
        logger.info("Starting model evaluation...")

        # Load model
        model, feature_names = self.load_model(model_path)

        # Load test data
        test_features_path = os.path.join(test_path, "test_features.csv")
        test_targets_path = os.path.join(test_path, "test_targets.csv")

        if not os.path.exists(test_features_path):
            raise FileNotFoundError(f"Test features not found: {test_features_path}")
        if not os.path.exists(test_targets_path):
            raise FileNotFoundError(f"Test targets not found: {test_targets_path}")

        X_test: pd.DataFrame = pd.read_csv(test_features_path, index_col=0, parse_dates=True)
        y_test: pd.Series = cast(
            pd.Series, pd.read_csv(test_targets_path, index_col=0, parse_dates=True).squeeze()
        )

        logger.info(
            "Loaded test data: X_test shape %s, y_test shape %s", X_test.shape, y_test.shape
        )

        # Make predictions
        y_pred: np.ndarray = model.predict(X_test)
        y_pred_proba: np.ndarray = model.predict_proba(X_test)

        # Calculate metrics
        accuracy: float = accuracy_score(y_test, y_pred)
        precision_macro: float = precision_score(y_test, y_pred, average="macro", zero_division=0)
        recall_macro: float = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1_macro: float = f1_score(y_test, y_pred, average="macro", zero_division=0)

        # Per-class metrics
        precision_per_class: np.ndarray = precision_score(
            y_test, y_pred, average=None, zero_division=0
        )
        recall_per_class: np.ndarray = recall_score(y_test, y_pred, average=None, zero_division=0)
        f1_per_class: np.ndarray = f1_score(y_test, y_pred, average=None, zero_division=0)

        # Classification report
        class_report: Dict[str, Any] = classification_report(
            y_test, y_pred, output_dict=True, zero_division=0
        )

        # Confusion matrix
        conf_matrix: np.ndarray = confusion_matrix(y_test, y_pred)

        # Model metrics for SageMaker
        evaluation_metrics: Dict[str, Any] = {
            "classification_metrics": {
                "accuracy": {"value": float(accuracy)},
                "precision_macro": {"value": float(precision_macro)},
                "recall_macro": {"value": float(recall_macro)},
                "f1_macro": {"value": float(f1_macro)},
            },
            "confusion_matrix": conf_matrix.tolist(),
            "classification_report": class_report,
        }

        # Log metrics
        logger.info("Model Evaluation Results:")
        logger.info("  Accuracy: %.4f", accuracy)
        logger.info("  Precision (macro): %.4f", precision_macro)
        logger.info("  Recall (macro): %.4f", recall_macro)
        logger.info("  F1-score (macro): %.4f", f1_macro)

        logger.info("Per-class metrics:")
        unique_classes: List[Any] = sorted(y_test.unique())
        for i, cls in enumerate(unique_classes):
            if i < len(precision_per_class):
                logger.info(
                    "  Class %s - P: %.4f, R: %.4f, F1: %.4f",
                    cls,
                    precision_per_class[i],
                    recall_per_class[i],
                    f1_per_class[i],
                )

        # Save evaluation results
        os.makedirs(output_path, exist_ok=True)

        # Save metrics in SageMaker format
        evaluation_file: str = os.path.join(output_path, "evaluation.json")
        with open(evaluation_file, "w") as f:
            json.dump(evaluation_metrics, f, indent=2)

        # Save detailed results
        detailed_results: Dict[str, Any] = {
            "test_accuracy": float(accuracy),
            "precision_macro": float(precision_macro),
            "recall_macro": float(recall_macro),
            "f1_macro": float(f1_macro),
            "per_class_precision": precision_per_class.tolist(),
            "per_class_recall": recall_per_class.tolist(),
            "per_class_f1": f1_per_class.tolist(),
            "confusion_matrix": conf_matrix.tolist(),
            "classification_report": class_report,
            "test_samples": len(X_test),
            "unique_classes": unique_classes,
        }

        detailed_file: str = os.path.join(output_path, "detailed_evaluation.json")
        with open(detailed_file, "w") as f:
            json.dump(detailed_results, f, indent=2)

        # Save predictions
        predictions_df: pd.DataFrame = pd.DataFrame(
            {
                "actual": y_test.values,
                "predicted": y_pred,
                "predicted_proba_class_0": y_pred_proba[:, 0] if y_pred_proba.shape[1] > 0 else 0,
                "predicted_proba_class_1": y_pred_proba[:, 1] if y_pred_proba.shape[1] > 1 else 0,
                "predicted_proba_class_2": y_pred_proba[:, 2] if y_pred_proba.shape[1] > 2 else 0,
            },
            index=X_test.index,
        )

        predictions_file: str = os.path.join(output_path, "predictions.csv")
        predictions_df.to_csv(predictions_file)

        # Feature importance analysis
        if hasattr(model, "feature_importances_"):
            feature_importance: pd.DataFrame = pd.DataFrame(
                {"feature": X_test.columns, "importance": model.feature_importances_}
            ).sort_values("importance", ascending=False)

            importance_file: str = os.path.join(output_path, "feature_importance.csv")
            feature_importance.to_csv(importance_file, index=False)

            logger.info("Top 10 most important features:")
            for _, row in feature_importance.head(10).iterrows():
                logger.info("  %s: %.4f", row["feature"], row["importance"])

        logger.info("Evaluation complete. Results saved to %s", output_path)
        logger.info("Evaluation file: %s", evaluation_file)
        logger.info("Detailed results: %s", detailed_file)
        logger.info("Predictions: %s", predictions_file)


def main() -> None:
    """Entry point for SageMaker evaluation job."""
    parser = argparse.ArgumentParser(description="Evaluate crypto prediction model")
    parser.add_argument(
        "--model-path",
        type=str,
        default="/opt/ml/processing/model",
        help="Path to the trained model",
    )
    parser.add_argument(
        "--test-path", type=str, default="/opt/ml/processing/test", help="Path to test data"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="/opt/ml/processing/evaluation",
        help="Path to save evaluation results",
    )

    args = parser.parse_args()

    try:
        service = EvaluationService()
        service.execute(args.model_path, args.test_path, args.output_path)
        logger.info("Evaluation job completed successfully!")
    except Exception as e:
        logger.error("Evaluation job failed: %s", str(e))
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
