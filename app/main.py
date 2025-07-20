import os

import pandas as pd
from ccxt import Exchange, binance
from ccxt.base.types import ConstructorArgs
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier

from app.dataset_retriever import OHLCVDatasetRetriever
from app.preprocessors import TechnicalIndicatorEnricherPreProcessor
from app.split_strategies import OutOfTimeSplitStrategy
from app.target_builders import TPSLTargetBuilder
from app.trainers import SKLearnModelTrainer

load_dotenv()

threshold_percent = 0.5  # 1% price movement
time_window_minutes = 30  # 30-minute prediction window
test_size = 0.3  # 30% for testing

exchange_config: ConstructorArgs = {
    "apiKey": os.getenv("BINANCE_API_KEY", ""),
    "secret": os.getenv("BINANCE_API_SECRET", ""),
    "sandbox": False,
    # "rateLimit": 1200,
    "enableRateLimit": False,
}
exchange: Exchange = binance(exchange_config)

# Fetch OHLCV data
df = OHLCVDatasetRetriever(exchange).execute(
    symbol="BTC/USDT", timeframe="1m", limit=3 * 3600 * 24
)

# Enrich with technical indicators
TechnicalIndicatorEnricherPreProcessor.add_technical_indicators(df)
df.dropna(inplace=True)  # Drop rows with NaN values such as those created by rolling calculations

# Calculate target variable
df["target"] = TPSLTargetBuilder(exchange, threshold_percent, time_window_minutes).execute(df)

# Remove rows that can't have targets (too close to end)
df = df.iloc[:-time_window_minutes]

df.to_csv("btc_usdt_ohlcv_enriched.csv", index=True)
print("Target variable distribution:")
print(df["target"].value_counts())

# Split dataset in train and test
X_train, X_test, y_train, y_test = OutOfTimeSplitStrategy.execute(df, test_size)


# Train baseline models
rf = RandomForestClassifier(n_estimators=30, random_state=42)
models = {
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

# Display feature importance for Random Forest
feature_names = [col for col in df.columns if col != "target"]

feature_importance = pd.DataFrame(
    {"feature": feature_names, "importance": rf.feature_importances_}
).sort_values("importance", ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))
