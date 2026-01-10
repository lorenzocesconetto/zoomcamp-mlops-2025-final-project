#!/usr/bin/env python3
"""
SageMaker Inference Script Entry Point for Crypto Price Prediction
This script serves as the entry point for SageMaker inference endpoints.

This is a self-contained script that includes all inference logic to avoid
import issues when deployed to SageMaker.
"""

import json
import logging
import os
import pickle
from io import StringIO
from typing import Any, Dict, List, Optional, TypedDict

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelArtifact(TypedDict):
    model: Any
    feature_names: List[str]
    metadata: Dict[str, Any]


class InferenceService:
    """Domain service for crypto prediction model inference."""

    def __init__(self) -> None:
        self.model_artifact: Optional[ModelArtifact] = None

    def load_model(self, model_dir: str) -> ModelArtifact:
        """Load the model from the model directory."""
        logger.info("Loading model from %s", model_dir)

        model_path = os.path.join(model_dir, "model.pkl")
        feature_names_path = os.path.join(model_dir, "feature_names.pkl")
        metadata_path = os.path.join(model_dir, "metadata.json")

        # Load the trained model
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        # Load feature names
        feature_names = None
        if os.path.exists(feature_names_path):
            with open(feature_names_path, "rb") as f:
                feature_names = pickle.load(f)

        # Load metadata
        metadata = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

        # Return model along with additional info
        model_artifact: ModelArtifact = {
            "model": model,
            "feature_names": feature_names or [],
            "metadata": metadata,
        }

        logger.info("Model loaded successfully")
        logger.info("Model type: %s", metadata.get("model_type", "Unknown"))
        logger.info("Feature count: %s", metadata.get("feature_count", "Unknown"))
        logger.info("Loaded feature_names (%d): %s", len(feature_names or []), feature_names)

        self.model_artifact = model_artifact
        return model_artifact

    def parse_input(
        self, request_body: str, request_content_type: str = "application/json"
    ) -> pd.DataFrame:
        """Parse input data for prediction."""
        logger.info("Parsing input with content type: %s", request_content_type)

        if request_content_type == "application/json":
            try:
                input_data = json.loads(request_body)

                # Handle different input formats
                if "instances" in input_data:
                    # Format: {"instances": [...]}
                    instances = input_data["instances"]
                elif "data" in input_data:
                    # Format: {"data": [...]}
                    instances = input_data["data"]
                elif isinstance(input_data, list):
                    # Format: [...]
                    instances = input_data
                else:
                    # Assume the entire payload is a single instance
                    instances = [input_data]

                # Convert to DataFrame
                df = pd.DataFrame(instances)
                logger.info("Parsed %d instances with %d features", len(df), len(df.columns))

                return df

            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in request body: {str(e)}")

        elif request_content_type == "text/csv":
            try:
                # Parse CSV data
                df = pd.read_csv(StringIO(request_body))
                logger.info(
                    "Parsed CSV with %d instances and %d features", len(df), len(df.columns)
                )
                return df
            except Exception as e:
                raise ValueError(f"Invalid CSV in request body: {str(e)}")

        else:
            raise ValueError(f"Unsupported content type: {request_content_type}")

    def predict(self, input_data: pd.DataFrame) -> Dict[str, Any]:
        """Make predictions using the loaded model."""
        if self.model_artifact is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        logger.info("Making predictions for %d instances", len(input_data))

        model = self.model_artifact["model"]
        feature_names = self.model_artifact["feature_names"]
        metadata = self.model_artifact["metadata"]

        try:
            # Ensure feature order matches training
            if feature_names:
                logger.info("Expected features: %s", feature_names)
                logger.info("Received features: %s", list(input_data.columns))

                # Check if all required features are present
                missing_features = set(feature_names) - set(input_data.columns)
                if missing_features:
                    logger.warning("Missing features: %s", missing_features)
                    raise ValueError("Missing features")

                # Reorder columns to match training
                input_data = input_data[feature_names]
            else:
                logger.warning("No feature_names available, using input features as-is")

            # Ensure all data is numeric
            logger.info("Input data shape before prediction: %s", input_data.shape)
            logger.info("Input data dtypes: %s", input_data.dtypes.to_dict())

            # Convert to numeric, coercing errors to NaN, then fill NaN with 0
            input_data = input_data.apply(pd.to_numeric, errors="coerce").fillna(0.0)
            logger.info("Input data shape after numeric conversion: %s", input_data.shape)

            # Make predictions
            predictions = model.predict(input_data)
            probabilities = model.predict_proba(input_data)

            # Prepare response
            response = {
                "predictions": predictions.tolist(),
                "probabilities": probabilities.tolist(),
                "model_metadata": {
                    "model_type": metadata.get("model_type", "Unknown"),
                    "version": metadata.get("version", "1.0"),
                    "feature_count": len(input_data.columns),
                },
            }

            # Add class labels if available
            if hasattr(model, "classes_"):
                response["class_labels"] = model.classes_.tolist()

            logger.info("Generated predictions for %d instances", len(predictions))

            return response

        except Exception as e:
            logger.error("Prediction failed: %s", str(e))
            raise ValueError(f"Prediction error: {str(e)}")

    def format_output(self, prediction: Dict[str, Any], accept: str = "application/json") -> str:
        """Format the prediction output."""
        logger.info("Formatting output with accept type: %s", accept)

        if accept == "application/json":
            return json.dumps(prediction)

        elif accept == "text/csv":
            # Convert predictions to CSV format
            predictions_df = pd.DataFrame(
                {
                    "prediction": prediction["predictions"],
                    "probability_class_0": [
                        p[0] if len(p) > 0 else 0 for p in prediction["probabilities"]
                    ],
                    "probability_class_1": [
                        p[1] if len(p) > 1 else 0 for p in prediction["probabilities"]
                    ],
                    "probability_class_2": [
                        p[2] if len(p) > 2 else 0 for p in prediction["probabilities"]
                    ],
                }
            )
            return predictions_df.to_csv(index=False)

        else:
            # Default to JSON
            return json.dumps(prediction)


# Global inference service instance for SageMaker
_inference_service: Optional[InferenceService] = None


# SageMaker entry point functions
def model_fn(model_dir: str) -> InferenceService:
    """Load the model for SageMaker inference."""
    global _inference_service
    _inference_service = InferenceService()
    _inference_service.load_model(model_dir)
    return _inference_service


def input_fn(request_body: str, request_content_type: str = "application/json") -> pd.DataFrame:
    """Parse input data for SageMaker prediction."""
    # Use the global service if available, otherwise create a temporary one
    service = _inference_service if _inference_service else InferenceService()
    return service.parse_input(request_body, request_content_type)


def predict_fn(input_data: pd.DataFrame, model: InferenceService) -> Dict[str, Any]:
    """Make predictions for SageMaker."""
    return model.predict(input_data)


def output_fn(prediction: Dict[str, Any], accept: str = "application/json") -> str:
    """Format output for SageMaker."""
    service = _inference_service if _inference_service else InferenceService()
    return service.format_output(prediction, accept)


# Export the SageMaker handler functions
__all__ = ["model_fn", "input_fn", "predict_fn", "output_fn"]


if __name__ == "__main__":
    """Entry point for local testing."""
    import tempfile

    from sklearn.ensemble import RandomForestClassifier

    logger.info("Testing inference functions locally...")

    # Create dummy model for testing
    dummy_model = RandomForestClassifier(n_estimators=10, random_state=42)
    X_dummy = np.random.rand(100, 5)
    y_dummy = np.random.randint(0, 3, 100)
    dummy_model.fit(X_dummy, y_dummy)

    # Create temporary model directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save dummy model
        model_path = os.path.join(temp_dir, "model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(dummy_model, f)

        # Save feature names
        feature_names = [f"feature_{i}" for i in range(5)]
        feature_names_path = os.path.join(temp_dir, "feature_names.pkl")
        with open(feature_names_path, "wb") as f:
            pickle.dump(feature_names, f)

        # Save metadata
        metadata = {"model_type": "RandomForestClassifier", "feature_count": 5, "version": "1.0"}
        metadata_path = os.path.join(temp_dir, "metadata.json")
        with open(metadata_path, "w") as meta_file:
            json.dump(metadata, meta_file)

        # Test inference service
        service = model_fn(temp_dir)

        # Test input parsing
        test_input = {"instances": [{f"feature_{i}": np.random.rand() for i in range(5)}]}
        parsed_input = input_fn(json.dumps(test_input))

        # Test prediction
        prediction_result = predict_fn(parsed_input, service)

        # Test output formatting
        formatted_output = output_fn(prediction_result)

        logger.info("Local testing completed successfully!")
        logger.info("Prediction result: %s", prediction_result)
        logger.info("Formatted output: %s", formatted_output)
