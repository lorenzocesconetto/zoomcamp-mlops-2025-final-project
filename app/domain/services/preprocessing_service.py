#!/usr/bin/env python3
"""
SageMaker Processing Service for Crypto Data Preprocessing
Domain service that handles data preprocessing for the ML pipeline.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Tuple

import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PreprocessingService:
    """Domain service for preprocessing crypto data for ML pipeline."""

    def __init__(self, threshold_percent: float = 0.5, time_window_minutes: int = 30):
        self.threshold_percent = threshold_percent
        self.time_window_minutes = time_window_minutes

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the OHLCV data."""
        logger.info("Adding technical indicators...")

        # Simple Moving Averages
        for window in [5, 10, 20, 50]:
            df[f"sma_{window}"] = df["close"].rolling(window=window).mean()

        # Exponential Moving Averages
        for window in [12, 26]:
            df[f"ema_{window}"] = df["close"].ewm(span=window).mean()

        # MACD
        df["macd"] = df["ema_12"] - df["ema_26"]
        df["macd_signal"] = df["macd"].ewm(span=9).mean()
        df["macd_histogram"] = df["macd"] - df["macd_signal"]

        # RSI
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()  # type: ignore
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()  # type: ignore
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        bb_window = 20
        bb_std = 2
        bb_sma = df["close"].rolling(window=bb_window).mean()
        bb_std_val = df["close"].rolling(window=bb_window).std()
        df["bb_upper"] = bb_sma + (bb_std_val * bb_std)
        df["bb_lower"] = bb_sma - (bb_std_val * bb_std)
        df["bb_middle"] = bb_sma
        df["bb_width"] = df["bb_upper"] - df["bb_lower"]
        df["bb_position"] = (df["close"] - df["bb_lower"]) / df["bb_width"]

        # Volume indicators
        df["volume_sma_20"] = df["volume"].rolling(window=20).mean()
        df["volume_ratio"] = df["volume"] / df["volume_sma_20"]

        # Price changes
        for period in [1, 3, 5, 10]:
            df[f"price_change_{period}"] = df["close"].pct_change(periods=period)
            df[f"volume_change_{period}"] = df["volume"].pct_change(periods=period)

        # Volatility
        df["volatility_10"] = df["close"].rolling(window=10).std()
        df["volatility_30"] = df["close"].rolling(window=30).std()

        logger.info("Added technical indicators. DataFrame shape: %s", df.shape)
        return df

    def create_target_variable(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create target variable based on TP/SL strategy."""
        logger.info(
            "Creating target variable with %.1f%% threshold and %d minute window...",
            self.threshold_percent,
            self.time_window_minutes,
        )

        targets = []

        for i in range(len(df) - self.time_window_minutes):
            current_price = df.iloc[i]["close"]
            future_prices = df.iloc[i + 1 : i + 1 + self.time_window_minutes]["close"]

            # Calculate thresholds
            take_profit_price = current_price * (1 + self.threshold_percent / 100)
            stop_loss_price = current_price * (1 - self.threshold_percent / 100)

            # Check which threshold is hit first
            tp_hit = (future_prices >= take_profit_price).any()
            sl_hit = (future_prices <= stop_loss_price).any()

            if tp_hit and sl_hit:
                # Both hit - check which comes first
                tp_idx = (future_prices >= take_profit_price).idxmax()
                sl_idx = (future_prices <= stop_loss_price).idxmax()
                target = 1 if tp_idx < sl_idx else 0
            elif tp_hit:
                target = 1  # Take profit hit first
            elif sl_hit:
                target = 0  # Stop loss hit first
            else:
                target = 2  # Neither threshold hit

            targets.append(target)

        # Add targets to dataframe
        df_with_target = df.iloc[: -self.time_window_minutes].copy()
        df_with_target["target"] = targets

        logger.info("Target distribution: %s", pd.Series(targets).value_counts().to_dict())
        return df_with_target

    def split_data(
        self, df: pd.DataFrame, test_size: float = 0.3
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data using out-of-time strategy."""
        logger.info("Splitting data with test_size=%.2f", test_size)

        # Sort by timestamp if not already sorted
        df = df.sort_index()

        # Calculate split point
        split_idx = int(len(df) * (1 - test_size))

        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]

        logger.info("Train set size: %d, Test set size: %d", len(train_df), len(test_df))
        return train_df, test_df

    def execute(self, input_path: str, output_path: str) -> None:
        """Main preprocessing function."""
        logger.info("Starting preprocessing. Input: %s, Output: %s", input_path, output_path)

        # Read raw data
        input_files = list(Path(input_path).glob("*.csv"))
        if not input_files:
            raise ValueError("No CSV files found in %s", input_path)

        # Process the first CSV file found (assuming single file for now)
        data_file = input_files[0]
        logger.info("Processing file: %s", data_file)

        df = pd.read_csv(data_file, index_col=0, parse_dates=True)
        logger.info("Loaded data with shape: %s", df.shape)

        # Add technical indicators
        df = self.add_technical_indicators(df)

        # Create target variable
        df = self.create_target_variable(df)

        # Remove rows with NaN values
        initial_shape = df.shape
        df = df.dropna()
        logger.info("Removed NaN values: %s -> %s", initial_shape, df.shape)

        # Split into train and test
        train_df, test_df = self.split_data(df)

        # Prepare feature columns (exclude target)
        feature_columns = [col for col in df.columns if col != "target"]

        # Save processed data
        os.makedirs(output_path, exist_ok=True)

        # Save training data
        train_features = train_df[feature_columns]
        train_targets = train_df["target"]

        train_features.to_csv(f"{output_path}/train_features.csv", index=True)
        train_targets.to_csv(f"{output_path}/train_targets.csv", index=True)

        # Save test data
        test_features = test_df[feature_columns]
        test_targets = test_df["target"]

        test_features.to_csv(f"{output_path}/test_features.csv", index=True)
        test_targets.to_csv(f"{output_path}/test_targets.csv", index=True)

        # Save feature names for later use
        pd.Series(feature_columns).to_csv(
            f"{output_path}/feature_names.csv", index=False, header=False
        )

        logger.info("Preprocessing complete. Files saved to %s", output_path)
        logger.info("Feature columns: %d", len(feature_columns))
        logger.info("Training samples: %d", len(train_df))
        logger.info("Test samples: %d", len(test_df))


def main() -> None:
    """Entry point for SageMaker processing job."""
    parser = argparse.ArgumentParser(description="Preprocess crypto OHLCV data")
    parser.add_argument(
        "--input-path",
        type=str,
        default="/opt/ml/processing/input",
        help="Path to input data directory",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="/opt/ml/processing/output",
        help="Path to output data directory",
    )

    args = parser.parse_args()

    try:
        service = PreprocessingService()
        service.execute(args.input_path, args.output_path)
        logger.info("Preprocessing job completed successfully!")
    except Exception as e:
        logger.error("Preprocessing job failed: %s", str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
