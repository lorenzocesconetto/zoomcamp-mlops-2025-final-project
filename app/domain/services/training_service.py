#!/usr/bin/env python3
"""
SageMaker Training Service for Crypto Price Prediction
Domain service that handles model training for the ML pipeline.
"""

import argparse
import json
import logging
import os
import pickle
import sys
import tarfile
from typing import Any, Dict, List, Optional, cast

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainingService:
    """Domain service for training crypto prediction models."""

    def __init__(self, hyperparameters: Optional[Dict[str, Any]] = None) -> None:
        self.hyperparameters: Dict[str, Any] = (
            hyperparameters if hyperparameters is not None else self._get_default_hyperparameters()
        )

    def _get_default_hyperparameters(self) -> Dict[str, Any]:
        """Get default hyperparameters."""
        return {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "random_state": 42,
        }

    def log_metrics_to_sagemaker(self, metrics: Dict[str, Any]) -> None:
        """Log metrics for SageMaker Experiments tracking."""
        try:
            # SageMaker automatically captures metrics printed in this format
            for metric_name, metric_value in metrics.items():
                print(f"{metric_name}: {metric_value}")

            # Also save metrics to a file for the evaluation step
            metrics_path: str = "/opt/ml/output/metrics.json"
            os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)

            logger.info("Metrics logged to %s", metrics_path)
        except Exception as e:
            logger.warning("Failed to log metrics: %s", str(e))

    def execute(self, train_path: str, model_path: str) -> None:
        """Train the crypto prediction model."""
        logger.info("Starting model training...")
        logger.info("Hyperparameters: %s", self.hyperparameters)

        # Load training data
        train_features_path: str = os.path.join(train_path, "train_features.csv")
        train_targets_path: str = os.path.join(train_path, "train_targets.csv")

        if not os.path.exists(train_features_path) or not os.path.exists(train_targets_path):
            raise FileNotFoundError(f"Training data not found at {train_path}")

        X_train: pd.DataFrame = pd.read_csv(train_features_path, index_col=0, parse_dates=True)
        y_train: pd.Series = cast(
            pd.Series, pd.read_csv(train_targets_path, index_col=0, parse_dates=True).squeeze()
        )

        logger.info(
            "Loaded training data: X_train shape %s, y_train shape %s",
            X_train.shape,
            y_train.shape,
        )
        logger.info("Target distribution: %s", y_train.value_counts().to_dict())

        # Initialize model with hyperparameters
        model: RandomForestClassifier = RandomForestClassifier(
            n_estimators=self.hyperparameters.get("n_estimators", 100),
            max_depth=self.hyperparameters.get("max_depth", None),
            min_samples_split=self.hyperparameters.get("min_samples_split", 2),
            min_samples_leaf=self.hyperparameters.get("min_samples_leaf", 1),
            max_features=self.hyperparameters.get("max_features", "sqrt"),
            random_state=self.hyperparameters.get("random_state", 42),
            n_jobs=-1,
            class_weight="balanced",  # Handle class imbalance
        )

        # Train the model
        logger.info("Training RandomForest model...")
        model.fit(X_train, y_train)

        # Make predictions on training data for training metrics
        y_train_pred: np.ndarray = model.predict(X_train)
        train_accuracy: float = accuracy_score(y_train, y_train_pred)

        logger.info("Training accuracy: %.4f", train_accuracy)

        # Feature importance analysis
        feature_importance: pd.DataFrame = pd.DataFrame(
            {"feature": X_train.columns, "importance": model.feature_importances_}
        ).sort_values("importance", ascending=False)

        logger.info("Top 10 most important features:")
        for idx, row in feature_importance.head(10).iterrows():
            logger.info("  %s: %.4f", row["feature"], row["importance"])

        # Log metrics for SageMaker Experiments
        metrics: Dict[str, Any] = {
            "train_accuracy": float(train_accuracy),
            "n_estimators": self.hyperparameters.get("n_estimators", 100),
            "max_depth": self.hyperparameters.get("max_depth", "None"),
            "n_features": len(X_train.columns),
            "n_samples": len(X_train),
        }

        self.log_metrics_to_sagemaker(metrics)

        # Save the model
        os.makedirs(model_path, exist_ok=True)

        # Save model
        model_file: str = os.path.join(model_path, "model.pkl")
        with open(model_file, "wb") as f:
            pickle.dump(model, f)

        # Save feature names
        feature_names_file: str = os.path.join(model_path, "feature_names.pkl")
        with open(feature_names_file, "wb") as f:
            pickle.dump(list(X_train.columns), f)

        # Save feature importance
        feature_importance_file: str = os.path.join(model_path, "feature_importance.csv")
        feature_importance.to_csv(feature_importance_file, index=False)

        # Save model metadata
        metadata: Dict[str, Any] = {
            "model_type": "RandomForestClassifier",
            "hyperparameters": self.hyperparameters,
            "feature_count": len(X_train.columns),
            "training_samples": len(X_train),
            "target_classes": list(y_train.unique()),
            "target_distribution": y_train.value_counts().to_dict(),
            "train_accuracy": float(train_accuracy),
        }

        metadata_file: str = os.path.join(model_path, "metadata.json")
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info("Model saved to %s", model_path)

        # Create model tar.gz for SageMaker
        model_tar_path: str = os.path.join(model_path, "model.tar.gz")
        with tarfile.open(model_tar_path, "w:gz") as tar:
            tar.add(model_file, arcname="model.pkl")
            tar.add(feature_names_file, arcname="feature_names.pkl")
            tar.add(metadata_file, arcname="metadata.json")

        logger.info("Model artifacts packaged in %s", model_tar_path)
        logger.info("Training completed successfully!")


def parse_hyperparameters() -> Dict[str, Any]:
    """Parse hyperparameters from environment variables."""
    hyperparameters: Dict[str, Any] = {}

    # Get hyperparameters from SageMaker environment
    hp_keys: List[str] = [
        "n_estimators",
        "max_depth",
        "min_samples_split",
        "min_samples_leaf",
        "max_features",
        "random_state",
    ]

    for key in hp_keys:
        value: Optional[str] = os.environ.get(f"SM_HP_{key.upper()}", os.environ.get(key))
        if value is not None:
            try:
                # Try to convert to int first
                if key in [
                    "n_estimators",
                    "max_depth",
                    "min_samples_split",
                    "min_samples_leaf",
                    "random_state",
                ]:
                    if value.lower() != "none":
                        hyperparameters[key] = int(value)
                    else:
                        hyperparameters[key] = None
                else:
                    hyperparameters[key] = value
            except ValueError:
                hyperparameters[key] = value

    # Set defaults
    hyperparameters.setdefault("n_estimators", 100)
    hyperparameters.setdefault("max_depth", 10)
    hyperparameters.setdefault("random_state", 42)

    return hyperparameters


def main() -> None:
    """Entry point for SageMaker training job."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Train crypto prediction model"
    )
    parser.add_argument(
        "--train",
        type=str,
        default=os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training"),
        help="Path to training data",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"),
        help="Path to save the model",
    )

    args = parser.parse_args()

    try:
        # Parse hyperparameters
        hyperparameters: Dict[str, Any] = parse_hyperparameters()

        # Create training service and execute
        service: TrainingService = TrainingService(hyperparameters)
        service.execute(args.train, args.model_dir)

        logger.info("Training job completed successfully!")

    except Exception as e:
        logger.error("Training job failed: %s", str(e))
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
