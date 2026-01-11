# Final Project

## Evaluation criteria

- **Problem description**

  - [x] 2 points: The problem is well described and it's clear what the problem the project solves

  > **Explanation**: Clearly stated in the [README.md](README.md) file, it's a cryptocurrency price movement prediction problem with specific classification targets (0: drop, 1: rise, 2: both/neither). Well-defined use case with take profit/stop loss thresholds.

- **Cloud**

  - [x] 4 points: The project is developed on the cloud and IaC tools are used for provisioning the infrastructure

  > **Explanation**: Full AWS cloud deployment using Terraform IaC with modular architecture. Infrastructure includes SageMaker, S3, CloudWatch, IAM roles, and auto-scaling configurations across 4 modules (shared-resources, model-tracking, model-training, model-deployment).

- **Experiment tracking and model registry**

  - [x] 4 points: Both experiment tracking and model registry are used

  > **Explanation**: SageMaker Experiments for tracking training runs and hyperparameters. SageMaker Model Registry for versioning and managing model artifacts with approval workflow and cross-account access policies.

- **Workflow orchestration**

  - [x] 4 points: Fully deployed workflow

  > **Explanation**: AWS SageMaker Pipelines orchestrate end-to-end ML workflow with preprocessing, training, evaluation, and model registration steps. Includes daily monitoring pipeline logging to CloudWatch.

- **Model deployment**

  - [x] 4 points: The model deployment code is containerized and could be deployed to cloud or special tools for model deployment are used

  > **Explanation**: SageMaker endpoints with containerized inference using scikit-learn Docker images. Auto-scaling configuration, blue/green deployment Lambda function, and data capture for monitoring.

- **Model monitoring**

  - [x] 4 points: Comprehensive model monitoring that sends alerts or runs a conditional workflow (e.g. retraining, generating debugging dashboard, switching to a different model) if the defined metrics threshold is violated

  > **Explanation**: CloudWatch dashboards track training jobs and endpoint performance. SNS alerts for model accuracy drops below 60% threshold. Data capture monitors inputs/outputs for drift detection with configurable sampling percentage.

- **Reproducibility**

  - [x] 4 points: Instructions are clear, it's easy to run the code, and it works. The versions for all the dependencies are specified.

  > **Explanation**: Clear documentation with step-by-step setup instructions. Dependency versions pinned in `pyproject.toml` with exact Terraform version constraints. Makefile provides easy commands for local execution.

- **Best practices**
  - [x] There are unit tests (1 point) - **pytest** configured with coverage, HTML reports, and mock support. Tests for all adapters including OHLCV retriever, technical indicators, trainers, etc.
  - [ ] There is an integration test (1 point)
  - [x] Linter and/or code formatter are used (1 point) - **isort**, **flake8**, and **mypy** for code formatting and type checking
  - [x] There's a Makefile (1 point) - **Comprehensive Makefile** with install, run, test, lint, and help commands
  - [x] There are pre-commit hooks (1 point) - **pre-commit** configuration with standard hooks and mypy type checking
  - [x] There's a CI/CD pipeline (2 points) - **GitHub Actions** workflow with lint/test/deploy stages using AWS OIDC for secure cloud deployment
