# Architectural Patterns

This document describes the design patterns and conventions used throughout the codebase.

## Hexagonal Architecture (Ports & Adapters)

The application follows hexagonal architecture to decouple business logic from infrastructure.

**Directory mapping:**
- `app/domain/ports/` - Abstract interfaces (ABCs)
- `app/adapters/` - Concrete implementations
- `app/domain/services/` - Business logic orchestration
- `app/entrypoints/` - Application entry points (CLI, SageMaker)

**Port definitions:**
- `DatasetRetriever` - [dataset_retriever.py:7-31](app/domain/ports/dataset_retriever.py#L7-L31)
- `Preprocessor` - [preprocessor.py:6-20](app/domain/ports/preprocessor.py#L6-L20)
- `TargetBuilder` - [target_builder.py:7-21](app/domain/ports/target_builder.py#L7-L21)
- `SplitStrategy` - [split_strategy.py:7-25](app/domain/ports/split_strategy.py#L7-L25)
- `Trainer` - [trainer.py:6-22](app/domain/ports/trainer.py#L6-L22)

**Adapter implementations:**
- `OHLCVDatasetRetriever` - [ohlcv.py:17-143](app/adapters/dataset_retriever/ohlcv.py#L17-L143)
- `TechnicalIndicatorEnricherPreProcessor` - [techinical_indicator_enricher.py:7-192](app/adapters/preprocessors/techinical_indicator_enricher.py#L7-L192)
- `TPSLTargetBuilder` - [tp_sl.py:16-84](app/adapters/target_builders/tp_sl.py#L16-L84)
- `OutOfTimeSplitStrategy` - [out_of_time.py:6-29](app/adapters/split_strategies/out_of_time.py#L6-L29)
- `SKLearnModelTrainer` - [sklearn_trainer.py:20-45](app/adapters/trainers/sklearn_trainer.py#L20-L45)

---

## Dependency Injection

Constructor injection is used to provide dependencies to adapters and services.

**Pattern example** - [ohlcv.py:18-37](app/adapters/dataset_retriever/ohlcv.py#L18-L37):
```
def __init__(self, exchange: Exchange, max_retries: int = 5,
             request_delay: int = 1, logger: logging.Logger = logger)
```

**Composition root** - [run_local.py:236-252](app/entrypoints/command_line/run_local.py#L236-L252):
Dependencies are instantiated and wired together at the entry point.

---

## Strategy Pattern

Multiple implementations can be swapped for the same port interface:
- Split strategies: `OutOfTimeSplitStrategy` (could add `RandomSplitStrategy`)
- Target builders: `TPSLTargetBuilder` (could add alternative labeling)
- Trainers: `SKLearnModelTrainer` (could add `XGBoostTrainer`)

---

## Pipeline Pattern

Data flows through a linear transformation chain:
```
fetch_data → preprocess → create_target → split → train → evaluate
```

See orchestration in [run_local.py:180-257](app/entrypoints/command_line/run_local.py#L180-L257).

---

## Error Handling

**Retry with exponential backoff** - [ohlcv.py:109-126](app/adapters/dataset_retriever/ohlcv.py#L109-L126):
Used for external API calls (Binance) with configurable max retries.

**Try-except with logging** - [training_service.py:236-251](app/domain/services/training_service.py#L236-L251):
All entry points wrap execution in try-except, log errors, and exit with code 1.

---

## Configuration Management

**CLI arguments** - [run_local.py:23-73](app/entrypoints/command_line/run_local.py#L23-L73):
Argparse for local execution parameters.

**Environment variables**:
- Exchange credentials: `BINANCE_API_KEY`, `BINANCE_API_SECRET`
- SageMaker hyperparameters: `SM_HP_*` prefix - [training_service.py:173-213](app/domain/services/training_service.py#L173-L213)

**Terraform variables** - [infra/variables.tf](infra/variables.tf):
Infrastructure configuration (instance types, timeouts, environment).

---

## Testing Patterns

**Fixture-based testing** - [conftest.py:9-65](tests/conftest.py#L9-L65):
- `sample_ohlcv_data` - generates realistic OHLCV DataFrame
- `mock_exchange` - mocks CCXT exchange to avoid API calls

**Test organization**:
- One test file per adapter in `tests/adapters/`
- Test files mirror source structure
- Pytest marks: `@pytest.mark.slow`, `@pytest.mark.integration`, `@pytest.mark.unit`

---

## Infrastructure as Code

**Modular Terraform** - [main.tf:36-106](infra/main.tf#L36-L106):
Four modules with explicit dependencies:
1. `shared_resources` - S3, IAM, KMS
2. `model_tracking` - Experiments, registry (depends on 1)
3. `model_training` - SageMaker pipelines (depends on 1, 2)
4. `model_deployment` - Endpoints, scaling (depends on 1, 2)

**Resource tagging** - [locals.tf:9-15](infra/locals.tf#L9-L15):
All resources tagged with Project, ManagedBy, Owner, Environment.

---

## Logging Convention

Standard logger setup - [logger.py:4-23](app/helpers/logger.py#L4-L23):
```
Format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

Logger is injected via constructor with default module-level logger.
