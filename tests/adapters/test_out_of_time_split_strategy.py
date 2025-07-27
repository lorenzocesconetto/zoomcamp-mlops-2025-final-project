import pandas as pd
import pytest

from app.adapters.split_strategies.out_of_time import OutOfTimeSplitStrategy


class TestOutOfTimeSplitStrategy:
    """Test suite for OutOfTimeSplitStrategy"""

    def test_execute_basic_split(self, sample_enriched_data: pd.DataFrame) -> None:
        """Test basic out-of-time split functionality"""
        test_size = 0.2

        X_train, X_test, y_train, y_test = OutOfTimeSplitStrategy.execute(
            sample_enriched_data, test_size
        )

        # Check return types
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(y_test, pd.Series)

        # Check sizes
        total_size = len(sample_enriched_data)
        expected_train_size = int(total_size * (1 - test_size))
        expected_test_size = total_size - expected_train_size

        assert len(X_train) == expected_train_size
        assert len(X_test) == expected_test_size
        assert len(y_train) == expected_train_size
        assert len(y_test) == expected_test_size

    def test_execute_chronological_order(self, sample_enriched_data: pd.DataFrame) -> None:
        """Test that split maintains chronological order"""
        test_size = 0.3

        X_train, X_test, y_train, y_test = OutOfTimeSplitStrategy.execute(
            sample_enriched_data, test_size
        )

        # Training data should come before test data chronologically
        train_end_time = X_train.index.max()
        test_start_time = X_test.index.min()

        assert train_end_time < test_start_time, "Training data should end before test data starts"

    def test_execute_feature_target_separation(self, sample_enriched_data: pd.DataFrame) -> None:
        """Test that features and target are properly separated"""
        test_size = 0.25

        X_train, X_test, y_train, y_test = OutOfTimeSplitStrategy.execute(
            sample_enriched_data, test_size
        )

        # Target column should not be in feature sets
        assert "target" not in X_train.columns
        assert "target" not in X_test.columns

        # Target series should have correct name
        assert y_train.name == "target"
        assert y_test.name == "target"

        # All other columns should be in feature sets
        expected_feature_cols = [col for col in sample_enriched_data.columns if col != "target"]
        assert list(X_train.columns) == expected_feature_cols
        assert list(X_test.columns) == expected_feature_cols

    def test_execute_different_test_sizes(self, sample_enriched_data: pd.DataFrame) -> None:
        """Test split with different test sizes"""
        test_sizes = [0.1, 0.2, 0.3, 0.5]
        total_size = len(sample_enriched_data)

        for test_size in test_sizes:
            X_train, X_test, y_train, y_test = OutOfTimeSplitStrategy.execute(
                sample_enriched_data, test_size
            )

            expected_train_size = int(total_size * (1 - test_size))
            expected_test_size = total_size - expected_train_size

            assert len(X_train) == expected_train_size
            assert len(X_test) == expected_test_size
            assert len(y_train) == expected_train_size
            assert len(y_test) == expected_test_size

    def test_execute_edge_case_small_test_size(self, sample_enriched_data: pd.DataFrame) -> None:
        """Test split with very small test size"""
        test_size = 0.01  # 1%

        X_train, X_test, y_train, y_test = OutOfTimeSplitStrategy.execute(
            sample_enriched_data, test_size
        )

        # Should still work with small test sizes
        assert len(X_test) >= 1
        assert len(y_test) >= 1
        assert len(X_train) + len(X_test) == len(sample_enriched_data)

    def test_execute_edge_case_large_test_size(self, sample_enriched_data: pd.DataFrame) -> None:
        """Test split with large test size"""
        test_size = 0.9  # 90%

        X_train, X_test, y_train, y_test = OutOfTimeSplitStrategy.execute(
            sample_enriched_data, test_size
        )

        # Should still work with large test sizes
        assert len(X_train) >= 1
        assert len(y_train) >= 1
        assert len(X_train) + len(X_test) == len(sample_enriched_data)

    def test_execute_index_preservation(self, sample_enriched_data: pd.DataFrame) -> None:
        """Test that original indices are preserved in splits"""
        test_size = 0.2
        original_index = sample_enriched_data.index

        X_train, X_test, y_train, y_test = OutOfTimeSplitStrategy.execute(
            sample_enriched_data, test_size
        )

        # Combined indices should match original
        combined_X_index = X_train.index.union(X_test.index)
        combined_y_index = y_train.index.union(y_test.index)

        pd.testing.assert_index_equal(combined_X_index.sort_values(), original_index)
        pd.testing.assert_index_equal(combined_y_index.sort_values(), original_index)

    def test_execute_no_data_leakage(self, sample_enriched_data: pd.DataFrame) -> None:
        """Test that there's no data leakage between train and test sets"""
        test_size = 0.3

        X_train, X_test, y_train, y_test = OutOfTimeSplitStrategy.execute(
            sample_enriched_data, test_size
        )

        # No overlap in indices
        train_indices = set(X_train.index)
        test_indices = set(X_test.index)

        assert len(train_indices.intersection(test_indices)) == 0
        assert len(train_indices.union(test_indices)) == len(sample_enriched_data)

    def test_execute_data_integrity(self, sample_enriched_data: pd.DataFrame) -> None:
        """Test that data values are preserved after split"""
        test_size = 0.25

        X_train, X_test, y_train, y_test = OutOfTimeSplitStrategy.execute(
            sample_enriched_data, test_size
        )

        # Reconstruct original data and compare
        reconstructed_X = pd.concat([X_train, X_test]).sort_index()
        reconstructed_y = pd.concat([y_train, y_test]).sort_index()

        original_X = sample_enriched_data.drop("target", axis=1)
        original_y = sample_enriched_data["target"]

        pd.testing.assert_frame_equal(reconstructed_X, original_X)
        pd.testing.assert_series_equal(reconstructed_y, original_y)

    def test_execute_with_single_row(self) -> None:
        """Test split behavior with single row dataset"""
        single_row_data = pd.DataFrame(
            {"feature1": [1.0], "feature2": [2.0], "target": [1]},
            index=[pd.Timestamp("2024-01-01")],
        )

        test_size = 0.5

        X_train, X_test, y_train, y_test = OutOfTimeSplitStrategy.execute(
            single_row_data, test_size
        )

        # With single row, train should be empty, test should have the row
        assert len(X_train) == 0
        assert len(X_test) == 1
        assert len(y_train) == 0
        assert len(y_test) == 1

    def test_execute_with_two_rows(self) -> None:
        """Test split behavior with two-row dataset"""
        two_row_data = pd.DataFrame(
            {"feature1": [1.0, 2.0], "feature2": [3.0, 4.0], "target": [0, 1]},
            index=[pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")],
        )

        test_size = 0.5

        X_train, X_test, y_train, y_test = OutOfTimeSplitStrategy.execute(two_row_data, test_size)

        # With two rows and 50% split, should get 1 in train, 1 in test
        assert len(X_train) == 1
        assert len(X_test) == 1
        assert len(y_train) == 1
        assert len(y_test) == 1

    def test_execute_static_method(self) -> None:
        """Test that execute is a static method"""
        # Should be callable without instantiation
        # For static methods, we check that it doesn't have 'self' parameter
        import inspect

        sig = inspect.signature(OutOfTimeSplitStrategy.execute)
        assert "self" not in sig.parameters

    def test_execute_missing_target_column(self) -> None:
        """Test behavior when target column is missing"""
        data_no_target = pd.DataFrame({"feature1": [1.0, 2.0, 3.0], "feature2": [4.0, 5.0, 6.0]})

        with pytest.raises(KeyError):
            OutOfTimeSplitStrategy.execute(data_no_target, 0.3)

    def test_execute_empty_dataframe(self) -> None:
        """Test behavior with empty DataFrame"""
        empty_df = pd.DataFrame()

        with pytest.raises((KeyError, IndexError)):
            OutOfTimeSplitStrategy.execute(empty_df, 0.3)

    def test_execute_all_numeric_targets(self) -> None:
        """Test with various numeric target values"""
        data_with_numeric_targets = pd.DataFrame(
            {
                "feature1": range(10),
                "feature2": range(10, 20),
                "target": [0, 1, 2, 0, 1, 2, 0, 1, 2, 0],
            },
            index=pd.date_range("2024-01-01", periods=10, freq="1h"),
        )

        X_train, X_test, y_train, y_test = OutOfTimeSplitStrategy.execute(
            data_with_numeric_targets, 0.3
        )

        # Check that target values are preserved
        assert set(y_train.unique()).issubset({0, 1, 2})
        assert set(y_test.unique()).issubset({0, 1, 2})
