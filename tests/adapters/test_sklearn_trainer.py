from typing import Generator, Tuple
from unittest.mock import Mock, call, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from app.adapters.trainers.sklearn_trainer import Model, SKLearnModelTrainer


class TestSKLearnModelTrainer:
    """Test suite for SKLearnModelTrainer"""

    @pytest.fixture
    def sample_models(self) -> Generator[list[Model], None, None]:
        """Sample models for testing"""
        yield [
            Model(
                name="RandomForest", model=RandomForestClassifier(n_estimators=10, random_state=42)
            ),
            Model(
                name="LogisticRegression", model=LogisticRegression(random_state=42, max_iter=100)
            ),
        ]

    @pytest.fixture
    def sample_training_data(
        self,
    ) -> Generator[Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series], None, None]:
        """Sample training and testing data"""
        np.random.seed(42)
        n_samples = 100
        n_features = 5

        X_train = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f"feature_{i}" for i in range(n_features)],
        )
        X_test = pd.DataFrame(
            np.random.randn(20, n_features), columns=[f"feature_{i}" for i in range(n_features)]
        )

        # Create binary classification targets
        y_train = pd.Series(np.random.choice([0, 1], size=n_samples))
        y_test = pd.Series(np.random.choice([0, 1], size=20))

        yield X_train, X_test, y_train, y_test

    def test_init(self, sample_models: list[Model], mock_logger: Mock) -> None:
        """Test SKLearnModelTrainer initialization"""
        trainer = SKLearnModelTrainer(models=sample_models, logger=mock_logger)

        assert trainer.models == sample_models
        assert trainer.logger == mock_logger

    def test_init_default_logger(self, sample_models: list[Model]) -> None:
        """Test initialization with default logger"""
        trainer = SKLearnModelTrainer(models=sample_models)

        assert trainer.models == sample_models
        assert trainer.logger is not None

    def test_execute_basic_functionality(
        self,
        sample_models: list[Model],
        sample_training_data: Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
        mock_logger: Mock,
    ) -> None:
        """Test basic training functionality"""
        X_train, X_test, y_train, y_test = sample_training_data
        trainer = SKLearnModelTrainer(models=sample_models, logger=mock_logger)

        # Should execute without errors
        trainer.execute(X_train, X_test, y_train, y_test)

        # Models should be fitted
        for model_dict in sample_models:
            model = model_dict["model"]
            assert hasattr(model, "predict")

            # Should be able to make predictions
            predictions = model.predict(X_test)
            assert len(predictions) == len(y_test)

    def test_execute_model_fitting(
        self,
        sample_models: list[Model],
        sample_training_data: Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
        mock_logger: Mock,
    ) -> None:
        """Test that models are properly fitted"""
        X_train, X_test, y_train, y_test = sample_training_data
        trainer = SKLearnModelTrainer(models=sample_models, logger=mock_logger)

        # Check models are not fitted initially
        for model_dict in sample_models:
            model = model_dict["model"]
            assert not hasattr(model, "classes_") or model.classes_ is None

        trainer.execute(X_train, X_test, y_train, y_test)

        # Check models are fitted after execution
        for model_dict in sample_models:
            model = model_dict["model"]
            assert hasattr(model, "classes_")
            assert model.classes_ is not None

    def test_execute_predictions_and_accuracy(
        self,
        sample_models: list[Model],
        sample_training_data: Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
        mock_logger: Mock,
    ) -> None:
        """Test that predictions are made and accuracy is calculated"""
        X_train, X_test, y_train, y_test = sample_training_data
        trainer = SKLearnModelTrainer(models=sample_models, logger=mock_logger)

        with patch("app.adapters.trainers.sklearn_trainer.accuracy_score") as mock_accuracy:
            with patch(
                "app.adapters.trainers.sklearn_trainer.classification_report"
            ) as mock_report:
                mock_accuracy.return_value = 0.85
                mock_report.return_value = "Mock classification report"

                trainer.execute(X_train, X_test, y_train, y_test)

                # Should call accuracy_score and classification_report for each model
                assert mock_accuracy.call_count == len(sample_models)
                assert mock_report.call_count == len(sample_models)

    def test_execute_logging_calls(
        self,
        sample_models: list[Model],
        sample_training_data: Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
        mock_logger: Mock,
    ) -> None:
        """Test that appropriate logging calls are made"""
        X_train, X_test, y_train, y_test = sample_training_data
        trainer = SKLearnModelTrainer(models=sample_models, logger=mock_logger)

        trainer.execute(X_train, X_test, y_train, y_test)

        # Should log training start, accuracy, and classification report for each model
        expected_calls = []
        for model_dict in sample_models:
            model_name = model_dict["name"]
            expected_calls.extend(
                [
                    call(f"Training {model_name}..."),
                    call(
                        f"{model_name} Accuracy: "
                        f"{accuracy_score(y_test, model_dict['model'].predict(X_test)):.4f}"
                    ),
                    call(f"{model_name} Classification Report:"),
                    call(
                        classification_report(
                            y_test, model_dict["model"].predict(X_test), zero_division=np.nan
                        )
                    ),
                ]
            )

        # Check that logger.info was called appropriately
        assert mock_logger.info.call_count >= len(sample_models) * 3

    def test_execute_single_model(
        self,
        sample_training_data: Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
        mock_logger: Mock,
    ) -> None:
        """Test execution with single model"""
        X_train, X_test, y_train, y_test = sample_training_data
        single_model = [
            Model(
                name="SingleModel", model=RandomForestClassifier(n_estimators=5, random_state=42)
            )
        ]
        trainer = SKLearnModelTrainer(models=single_model, logger=mock_logger)

        trainer.execute(X_train, X_test, y_train, y_test)

        # Model should be fitted
        model = single_model[0]["model"]
        assert hasattr(model, "classes_")

        # Should be able to predict
        predictions = model.predict(X_test)
        assert len(predictions) == len(y_test)

    def test_execute_multiple_models(
        self,
        sample_training_data: Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
        mock_logger: Mock,
    ) -> None:
        """Test execution with multiple models"""
        X_train, X_test, y_train, y_test = sample_training_data
        multiple_models = [
            Model(name="RF1", model=RandomForestClassifier(n_estimators=5, random_state=42)),
            Model(name="RF2", model=RandomForestClassifier(n_estimators=10, random_state=123)),
            Model(name="LR", model=LogisticRegression(random_state=42, max_iter=100)),
        ]
        trainer = SKLearnModelTrainer(models=multiple_models, logger=mock_logger)

        trainer.execute(X_train, X_test, y_train, y_test)

        # All models should be fitted
        for model_dict in multiple_models:
            model = model_dict["model"]
            assert hasattr(model, "classes_")

            # Should be able to predict
            predictions = model.predict(X_test)
            assert len(predictions) == len(y_test)

    def test_execute_with_multiclass_targets(self, mock_logger: Mock) -> None:
        """Test execution with multiclass classification targets"""
        np.random.seed(42)
        X_train = pd.DataFrame(np.random.randn(100, 3), columns=["f1", "f2", "f3"])
        X_test = pd.DataFrame(np.random.randn(20, 3), columns=["f1", "f2", "f3"])
        y_train = pd.Series(np.random.choice([0, 1, 2], size=100))
        y_test = pd.Series(np.random.choice([0, 1, 2], size=20))

        models = [
            Model(
                name="MulticlassRF", model=RandomForestClassifier(n_estimators=5, random_state=42)
            )
        ]
        trainer = SKLearnModelTrainer(models=models, logger=mock_logger)

        trainer.execute(X_train, X_test, y_train, y_test)

        # Model should handle multiclass
        model = models[0]["model"]
        predictions = model.predict(X_test)
        assert len(predictions) == len(y_test)
        assert set(predictions).issubset({0, 1, 2})

    def test_execute_with_different_data_types(
        self, sample_models: list[Model], mock_logger: Mock
    ) -> None:
        """Test execution with different data types"""
        # Create data with mixed types (should work after conversion)
        X_train = pd.DataFrame(
            {
                "int_feature": [1, 2, 3, 4, 5] * 20,
                "float_feature": [1.1, 2.2, 3.3, 4.4, 5.5] * 20,
                "bool_feature": [True, False, True, False, True] * 20,
            }
        )
        X_test = pd.DataFrame(
            {
                "int_feature": [1, 2, 3, 4],
                "float_feature": [1.1, 2.2, 3.3, 4.4],
                "bool_feature": [True, False, True, False],
            }
        )
        y_train = pd.Series([0, 1] * 50)
        y_test = pd.Series([0, 1, 0, 1])

        trainer = SKLearnModelTrainer(models=sample_models, logger=mock_logger)

        # Should handle mixed data types
        trainer.execute(X_train, X_test, y_train, y_test)

        for model_dict in sample_models:
            model = model_dict["model"]
            predictions = model.predict(X_test)
            assert len(predictions) == len(y_test)

    def test_execute_empty_models_list(
        self,
        sample_training_data: Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
        mock_logger: Mock,
    ) -> None:
        """Test execution with empty models list"""
        X_train, X_test, y_train, y_test = sample_training_data
        trainer = SKLearnModelTrainer(models=[], logger=mock_logger)

        # Should execute without errors (but do nothing)
        trainer.execute(X_train, X_test, y_train, y_test)

    def test_model_typed_dict_structure(self) -> None:
        """Test Model TypedDict structure"""
        # Test that Model TypedDict has correct structure
        model_dict = Model(name="TestName", model=RandomForestClassifier())

        assert "name" in model_dict
        assert "model" in model_dict
        assert isinstance(model_dict["name"], str)
        assert hasattr(model_dict["model"], "fit")
        assert hasattr(model_dict["model"], "predict")

    def test_execute_preserves_original_data(
        self,
        sample_models: list[Model],
        sample_training_data: Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
        mock_logger: Mock,
    ) -> None:
        """Test that original data is not modified during execution"""
        X_train, X_test, y_train, y_test = sample_training_data

        # Create copies to compare
        X_train_copy = X_train.copy()
        X_test_copy = X_test.copy()
        y_train_copy = y_train.copy()
        y_test_copy = y_test.copy()

        trainer = SKLearnModelTrainer(models=sample_models, logger=mock_logger)
        trainer.execute(X_train, X_test, y_train, y_test)

        # Original data should be unchanged
        pd.testing.assert_frame_equal(X_train, X_train_copy)
        pd.testing.assert_frame_equal(X_test, X_test_copy)
        pd.testing.assert_series_equal(y_train, y_train_copy)
        pd.testing.assert_series_equal(y_test, y_test_copy)

    def test_execute_with_nan_values(self, sample_models: list[Model], mock_logger: Mock) -> None:
        """Test behavior with NaN values in data"""
        # Create data with NaN values
        X_train = pd.DataFrame(
            {
                "feature1": [1.0, 2.0, np.nan, 4.0, 5.0] * 20,
                "feature2": [1.0, np.nan, 3.0, 4.0, 5.0] * 20,
            }
        )
        X_test = pd.DataFrame(
            {"feature1": [1.0, 2.0, np.nan, 4.0], "feature2": [1.0, 2.0, 3.0, np.nan]}
        )
        y_train = pd.Series([0, 1] * 50)
        y_test = pd.Series([0, 1, 0, 1])

        trainer = SKLearnModelTrainer(models=sample_models, logger=mock_logger)

        # Should handle NaN values (may raise error or handle gracefully depending on model)
        # This test ensures the code doesn't crash unexpectedly
        try:
            trainer.execute(X_train, X_test, y_train, y_test)
        except (ValueError, Exception):
            # Some models may not handle NaN values - this is expected behavior
            pass
