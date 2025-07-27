import argparse
import os
import warnings

import pandas as pd
from ccxt import Exchange, binance
from ccxt.base.types import ConstructorArgs
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier

from app.adapters.dataset_retriever import OHLCVDatasetRetriever
from app.adapters.preprocessors import TechnicalIndicatorEnricherPreProcessor
from app.adapters.split_strategies import OutOfTimeSplitStrategy
from app.adapters.target_builders import TPSLTargetBuilder
from app.adapters.trainers import Model, SKLearnModelTrainer
from app.helpers.logger import build_logger

warnings.filterwarnings("ignore")

logger = build_logger("e2e-pipeline")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for model training configuration."""
    parser = argparse.ArgumentParser(
        description="Train ML model for cryptocurrency price prediction"
    )

    parser.add_argument(
        "--threshold-percent",
        type=float,
        default=0.5,
        help="Price movement threshold percentage (default: 0.5)",
    )

    parser.add_argument(
        "--time-window-minutes",
        type=int,
        default=30,
        help="Prediction time window in minutes (default: 30)",
    )

    parser.add_argument(
        "--test-size", type=float, default=0.3, help="Test set size as fraction (default: 0.3)"
    )

    parser.add_argument(
        "--start-date",
        type=str,
        default="2025-04-01T00:00:00Z",
        help="Start date for data collection (ISO format)",
    )

    parser.add_argument(
        "--end-date",
        type=str,
        default="2025-06-30T23:59:59Z",
        help="End date for data collection (ISO format)",
    )

    parser.add_argument(
        "--symbol", type=str, default="BTC/USDT", help="Trading symbol (default: BTC/USDT)"
    )

    parser.add_argument("--timeframe", type=str, default="1m", help="Data timeframe (default: 1m)")

    parser.add_argument(
        "--skip-data-fetch",
        action="store_true",
        help="Skip data fetching if raw data file already exists",
    )

    return parser.parse_args()


def setup_exchange() -> Exchange:
    """Initialize and configure the exchange connection."""
    load_dotenv()

    exchange_config: ConstructorArgs = {
        "apiKey": os.getenv("BINANCE_API_KEY", ""),
        "secret": os.getenv("BINANCE_API_SECRET", ""),
        "sandbox": os.getenv("BINANCE_SANDBOX", "False").lower() == "true",
    }

    return binance(exchange_config)


def fetch_data(
    exchange: Exchange,
    symbol: str,
    timeframe: str,
    start_date: str,
    end_date: str,
    skip_fetch: bool = False,
) -> pd.DataFrame:
    """Fetch OHLCV data from the exchange or load from existing file."""
    raw_data_filename = (
        "./data/raw_data-"
        f"{start_date[:10].replace('-', '_')}-{end_date[:10].replace('-', '_')}.csv"
    )

    raw_file_exists = os.path.exists(raw_data_filename)
    if skip_fetch or raw_file_exists:
        if not raw_file_exists:
            raise FileNotFoundError(f"Raw data file not found: {raw_data_filename}")

        logger.info("Skipping data fetch and loading existing raw data from %s", raw_data_filename)
        df = pd.read_csv(raw_data_filename, index_col=0, parse_dates=True)
        logger.info("Loaded data shape: %s", df.shape)
        return df

    logger.info("Fetching %s data from %s to %s...", symbol, start_date, end_date)

    df = OHLCVDatasetRetriever(exchange).execute(
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
    )

    # Save raw data
    df.to_csv(raw_data_filename)
    logger.info("Raw data saved to: %s", raw_data_filename)

    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Enrich data with technical indicators and prepare for training."""
    logger.info("Enriching data with technical indicators...")

    # Enrich with technical indicators
    TechnicalIndicatorEnricherPreProcessor.execute(df)
    df.dropna(inplace=True)  # Drop rows with NaN values

    logger.info("Data shape after preprocessing: %s", df.shape)
    return df


def create_target_variable(
    df: pd.DataFrame, exchange: Exchange, threshold_percent: float, time_window_minutes: int
) -> pd.DataFrame:
    """Create target variable for the ML model."""
    logger.info(
        "Creating target variable with %.1f%% threshold and %dmin window...",
        threshold_percent,
        time_window_minutes,
    )

    df["target"] = TPSLTargetBuilder(exchange, threshold_percent, time_window_minutes).execute(df)

    # Remove rows that can't have targets (too close to end)
    df = df.iloc[:-time_window_minutes]

    logger.info("Target variable distribution:")
    logger.info(df["target"].value_counts())

    return df


def split_dataset(
    df: pd.DataFrame, test_size: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split dataset into training and testing sets."""
    logger.info("Splitting dataset with test_size=%.2f...", test_size)

    X_train, X_test, y_train, y_test = OutOfTimeSplitStrategy.execute(df, test_size)

    logger.info("Training set size: %d", len(X_train))
    logger.info("Test set size: %d", len(X_test))

    return X_train, X_test, y_train, y_test


def train_models(
    X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series
) -> dict[str, Model]:
    """Train ML models and return the trained models."""
    logger.info("Training models...")

    rf = RandomForestClassifier(n_estimators=40, random_state=42)
    models: dict[str, Model] = {
        "RandomForest": {
            "name": "RandomForest",
            "model": rf,
        },
    }

    SKLearnModelTrainer([model for _, model in models.items()]).execute(
        X_train,
        X_test,
        y_train,
        y_test,
    )

    return models


def display_feature_importance(models: dict[str, Model], df: pd.DataFrame) -> None:
    """Display feature importance for trained models."""
    logger.info("Analyzing feature importance...")

    # Get the Random Forest model
    rf_model = models["RandomForest"]["model"]
    feature_names = [col for col in df.columns if col != "target"]

    feature_importance = pd.DataFrame(
        {"feature": feature_names, "importance": rf_model.feature_importances_}
    ).sort_values("importance", ascending=False)

    logger.info("Top 10 Most Important Features: \n %s", feature_importance.head(10))


def save_processed_data(df: pd.DataFrame, filename: str) -> None:
    """Save the processed and enriched dataset."""
    df.to_csv(filename, index=True)
    logger.info("Processed data saved to: %s", filename)


def main() -> None:
    """Main function to orchestrate the entire ML pipeline."""
    args = parse_arguments()

    logger.info("=== ML Model Training Pipeline ===")
    logger.info("Configuration:")
    logger.info("  - Threshold: %.1f%%", args.threshold_percent)
    logger.info("  - Time window: %d minutes", args.time_window_minutes)
    logger.info("  - Test size: %.2f", args.test_size)
    logger.info("  - Date range: %s to %s", args.start_date, args.end_date)
    logger.info("  - Symbol: %s", args.symbol)
    logger.info("  - Timeframe: %s", args.timeframe)
    logger.info("  - Skip data fetch: %s", args.skip_data_fetch)
    print()

    exchange = setup_exchange()
    df = fetch_data(
        exchange, args.symbol, args.timeframe, args.start_date, args.end_date, args.skip_data_fetch
    )
    df = preprocess_data(df)
    df = create_target_variable(df, exchange, args.threshold_percent, args.time_window_minutes)

    # Fix the filename to handle symbols with forward slashes
    safe_symbol = args.symbol.replace("/", "_")
    save_processed_data(
        df,
        f"./data/enriched-{safe_symbol}-{args.time_window_minutes}m-"
        f"{args.start_date[:10]}-{args.end_date[:10]}.csv",
    )
    X_train, X_test, y_train, y_test = split_dataset(df, args.test_size)
    models = train_models(X_train, X_test, y_train, y_test)
    display_feature_importance(models, df)
    logger.info("\n=== Pipeline completed successfully! ===")


if __name__ == "__main__":
    main()
