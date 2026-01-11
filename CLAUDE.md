# CLAUDE.md

## Project Overview

Cryptocurrency price movement prediction MLOps project. Predicts whether a crypto price will hit a take-profit (+x%) or stop-loss (-x%) threshold within a configurable time window.

**Classification targets:**
- Class 0: Price drops by threshold first
- Class 1: Price rises by threshold first
- Class 2: Both/neither within timeframe

## Tech Stack

| Category | Technology |
|----------|------------|
| Language | Python 3.13 |
| Package Manager | uv |
| ML | scikit-learn (RandomForest) |
| Data | pandas, numpy, ccxt (Binance API) |
| Infrastructure | Terraform, AWS SageMaker |
| CI/CD | GitHub Actions |
| Testing | pytest (>85% coverage required) |
| Quality | black, isort, flake8, mypy, pre-commit |

## Project Structure

```
app/
├── domain/
│   ├── ports/           # Abstract interfaces (ABCs)
│   └── services/        # Business logic (training, preprocessing, inference)
├── adapters/            # Implementations of ports
│   ├── dataset_retriever/   # OHLCV fetcher (Binance via ccxt)
│   ├── preprocessors/       # Technical indicator enrichment
│   ├── target_builders/     # TP/SL target generation
│   ├── split_strategies/    # Time-series split
│   └── trainers/            # Model training
├── entrypoints/
│   ├── command_line/    # Local CLI (run_local.py)
│   └── sagemaker/       # SageMaker job entry points
└── helpers/             # Utilities (logger)

infra/                   # Terraform IaC
├── shared-resources/    # S3, IAM, KMS
├── model-training/      # SageMaker pipelines
├── model-deployment/    # Endpoints, auto-scaling
└── model-tracking/      # Experiments, registry

tests/                   # pytest unit tests
├── adapters/            # Adapter tests
└── conftest.py          # Shared fixtures
```

## Essential Commands

```bash
# Setup
make install              # uv sync + pre-commit install

# Development
make run                  # Run local end-to-end pipeline
make lint                 # black, isort, flake8, mypy
make test                 # Run pytest
make test-cov             # pytest with coverage report

# Cleanup
make clean-test           # Remove test artifacts
```

## Key Entry Points

| Entry Point | Purpose |
|-------------|---------|
| [run_local.py](app/entrypoints/command_line/run_local.py) | Local pipeline execution |
| [preprocessing.py](app/entrypoints/sagemaker/preprocessing.py) | SageMaker preprocessing job |
| [train.py](app/entrypoints/sagemaker/train.py) | SageMaker training job |
| [inference.py](app/entrypoints/sagemaker/inference.py) | SageMaker inference handler |
| [evaluate.py](app/entrypoints/sagemaker/evaluate.py) | SageMaker evaluation job |

## CLI Arguments

The local pipeline ([run_local.py:23-73](app/entrypoints/command_line/run_local.py#L23-L73)) accepts:

| Argument | Default | Description |
|----------|---------|-------------|
| `--threshold-percent` | 0.5 | TP/SL threshold percentage |
| `--time-window-minutes` | 30 | Lookahead window |
| `--test-size` | 0.3 | Test split ratio |
| `--start-date` | 2025-04-01 | Data start date |
| `--end-date` | 2025-06-30 | Data end date |
| `--symbol` | BTC/USDT | Trading pair |
| `--timeframe` | 1m | OHLCV timeframe |
| `--skip-data-fetch` | false | Use cached data |

## Environment Variables

```bash
BINANCE_API_KEY=...       # Exchange API key
BINANCE_API_SECRET=...    # Exchange API secret
BINANCE_SANDBOX=true      # Use sandbox mode
```

## Testing

Tests are in `tests/adapters/` with fixtures defined in [conftest.py](tests/conftest.py).

```bash
make test-cov             # Run with coverage (>85% required)
```

Key fixtures: `sample_ohlcv_data`, `sample_enriched_data`, `mock_exchange`, `mock_logger`

## Infrastructure Deployment

See [infra/README.md](infra/README.md) for Terraform deployment details.

```bash
cd infra
make init                 # Initialize Terraform
make plan                 # Preview changes
make apply                # Deploy infrastructure
```

CI/CD via GitHub Actions: [deploy.yml](.github/workflows/deploy.yml)

## Additional Documentation

When working on specific areas, consult these files:

| Topic | File |
|-------|------|
| Architectural patterns & conventions | [.claude/docs/architectural_patterns.md](.claude/docs/architectural_patterns.md) |
| Infrastructure details | [infra/README.md](infra/README.md) |
| Deployment procedures | [infra/DEPLOYMENT.md](infra/DEPLOYMENT.md) |
| Test guidelines | [tests/README.md](tests/README.md) |
| MLOps evaluation criteria | [EVALUATION.md](EVALUATION.md) |
