# Adapter Tests

This directory contains comprehensive unit tests for all adapters in the application, using pytest as the testing framework.

## Test Structure

The tests are organized to mirror the adapter structure:

```
tests/
├── conftest.py                              # Shared fixtures and configuration
├── adapters/
│   ├── test_ohlcv_dataset_retriever.py     # Tests for OHLCV data retrieval
│   ├── test_technical_indicator_enricher.py # Tests for technical indicators
│   ├── test_out_of_time_split_strategy.py  # Tests for data splitting
│   ├── test_tp_sl_target_builder.py        # Tests for target building
│   └── test_sklearn_trainer.py             # Tests for model training
└── README.md                               # This file
```

## Test Coverage

### Dataset Retriever (`test_ohlcv_dataset_retriever.py`)

- **13 tests** covering the `OHLCVDatasetRetriever` adapter
- Tests initialization, data fetching, retry logic, pagination, error handling
- Mocks CCXT exchange interactions to avoid external dependencies
- Validates data conversion and filtering functionality

### Technical Indicator Enricher (`test_technical_indicator_enricher.py`)

- **18 tests** covering the `TechnicalIndicatorEnricherPreProcessor` adapter
- Tests individual technical indicators (RSI, MACD, Bollinger Bands, etc.)
- Validates comprehensive indicator addition and data integrity
- Tests time-based features and cyclical encoding
- Ensures proper data type handling and index preservation

### Split Strategy (`test_out_of_time_split_strategy.py`)

- **15 tests** covering the `OutOfTimeSplitStrategy` adapter
- Tests chronological splitting functionality
- Validates feature/target separation and data integrity
- Tests edge cases (small datasets, different split sizes)
- Ensures no data leakage between train/test sets

### Target Builder (`test_tp_sl_target_builder.py`)

- **4 tests** covering the `TPSLTargetBuilder` adapter
- Tests take-profit/stop-loss target generation
- Validates threshold detection logic
- Tests initialization and basic functionality

### Model Trainer (`test_sklearn_trainer.py`)

- **16 tests** covering the `SKLearnModelTrainer` adapter
- Tests model training and evaluation workflow
- Validates logging and metrics calculation
- Tests multiple model handling and different data types
- Ensures data integrity and proper error handling

## Fixtures

The `conftest.py` file provides shared fixtures:

- `sample_ohlcv_data`: Sample OHLCV DataFrame with realistic price data
- `sample_enriched_data`: Sample DataFrame with target column for splitting tests
- `mock_exchange`: Mock CCXT exchange object
- `mock_logger`: Mock logger for testing logging functionality

## Running Tests

### Run all tests:

```bash
python -m pytest tests/
```

### Run with verbose output:

```bash
python -m pytest tests/ -v
```

### Run specific adapter tests:

```bash
python -m pytest tests/adapters/test_ohlcv_dataset_retriever.py -v
```

### Run with coverage:

```bash
python -m pytest tests/ --cov=app/adapters
```

## Test Philosophy

The tests follow these principles:

1. **Isolation**: Each test is independent and uses mocks to avoid external dependencies
2. **Comprehensive Coverage**: Tests cover happy paths, edge cases, and error conditions
3. **Data Integrity**: Tests ensure original data is not modified during processing
4. **Realistic Scenarios**: Uses realistic financial data patterns and parameters
5. **Clear Assertions**: Each test has clear, specific assertions about expected behavior

## Dependencies

The test suite requires:

- `pytest` - Testing framework
- `pytest-mock` - Mocking utilities
- `pandas` - Data manipulation (same as main app)
- `numpy` - Numerical operations
- `scikit-learn` - ML models for trainer tests
- `unittest.mock` - Built-in mocking (Python standard library)

All dependencies are included in the `dev` dependency group in `pyproject.toml`.
