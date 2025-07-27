# Crypto Price Movement Prediction

> **MLOps 2025 Final Project** - Machine Learning pipeline for predicting cryptocurrency price movements

## Problem Statement

Predicts whether a cryptocurrency price will first reach a **+x%** gain or **-x%** loss within a given timeframe (e.g., 15 minutes).

**Classification Labels:**

- `0`: Price drops by -x% first
- `1`: Price rises by +x% first
- `2`: Price hits both thresholds within the same minute or neither within timeframe

## Features

- 📊 **OHLCV Data Pipeline** - Automated retrieval and processing of financial time series
- 🔧 **Technical Indicators** - Enrichment with popular trading indicators
- 🎯 **Target Engineering** - Take profit/stop loss target variable creation
- ⏰ **Backtesting** - Out-of-time data splitting for realistic evaluation
- 🤖 **ML Training** - Scikit-learn integration with modular trainer system

## Quick Start

```bash
# Install dependencies and pre-commit hook
make install

# Run the pipeline locally
make run

# Show all available commands
make help
```

## Project Structure & Architecture

- We're following the `Hexagonal Architecture` (Ports and Adapters) pattern.
  - The `app` folder contains the core application logic.
  - The `app/domain` folder contains the domain logic.
  - The `app/adapters` folder contains the adapters for the application.
  - The `app/helpers` folder contains the helpers for the application.
  - The `app/domain/ports` folder contains the ports for the application.
  - The `scripts` folder contains the scripts for the application.

```
app/
├── adapters/            # External integrations & implementations
│   ├── dataset_retriever/    # OHLCV data fetching
│   ├── preprocessors/        # Technical indicator enrichment
│   ├── target_builders/      # Target Strategy implementations
│   ├── split_strategies/     # Backtesting data splits
│   └── trainers/            # ML model training
├── domain/              # Core business logic
│   └── ports/           # Interface definitions
└── helpers/             # Utility functions
```

## Pipeline orchestration

- We use `AWS SageMaker Pipelines` to orchestrate our pipelines
- The current pipelines exist:
  - Training pipeline: builds dataset and trains the ML model
  - Monitoring pipeline: runs daily and logs results to `CloudWatch`

## Model deployment

- We use Amazon SageMaker to deploy a real-time model.

## Infrastructure as Code (IaC)

- Terraform is used to manage the infrastructure as code, and is defined in the `infra` folder.

## CI/CD

- We use `GitHub Actions` with `Terraform` to deploy the entire infrastructure automatically.
- You should manually create the S3 bucket for the Terraform state, since Terraform can't create resources for you before it's setup.
- Create the environments in GitHub: `dev`, `staging`, `prod` (you should be fine creating only `dev`).
- Make sure `AWS_ACCOUNT_ID`, `AWS_REGION`, `TERRAFORM_STATE_BUCKET` are set as (repository or environment) variables in the GitHub Actions workflow.
- You may run the destroy workflow (`.github/workflows/destroy.yml`) to easily destroy the infrastructure and prevent unexpected costs.
- To set up the GitHub Actions workflow, you need to set up the AWS OIDC provider and IAM role.

1. **Set Up AWS OIDC Provider:**

   - Navigate to AWS `IAM` → Click on `Identity Providers` on the left sidebar → click on `Add Provider`
   - Select `OpenID Connect` (OIDC) as the provider type
   - Enter GitHub's OIDC URL: `https://token.actions.githubusercontent.com`
   - For the audience, use: `sts.amazonaws.com`
   - Click `Add Provider` to save the configuration

2. **Create IAM Role for GitHub Actions:**

   - Navigate to AWS `IAM` → Click on `Roles` on the left sidebar → click on `Create Role`
   - Select `Web Identity` and choose the GitHub OIDC provider you created
   - For the audience, pick `sts.amazonaws.com` from the dropdown
   - Name the role `tf-deploy-role`, or rename the `ROLE_NAME` variable in `.github/workflows/deploy.yml` to your desired role name.
   - Enter your GitHub repository information (you can use your username if you don't have an organization). Once created, the trust policy should look like this:

     ```json
     {
       "Version": "2012-10-17",
       "Statement": [
         {
           "Effect": "Allow",
           "Principal": {
             "Federated": "arn:aws:iam::<AWS_ACCOUNT_ID>:oidc-provider/token.actions.githubusercontent.com"
           },
           "Action": "sts:AssumeRoleWithWebIdentity",
           "Condition": {
             "StringEquals": {
               "token.actions.githubusercontent.com:aud": "sts.amazonaws.com",
               "token.actions.githubusercontent.com:sub": "repo:<GITHUB_ORG>/<REPO>:ref:refs/heads/main"
             }
           }
         }
       ]
     }
     ```

   - Assign the appropriate policies to the role. For simplicity and speed, you can assign `AdministratorAccess` policy to the role. It's not recommended for production environments, but it's fine for this project.

## Requirements

- Python 3.13+
- Binance API credentials (optional)
- Dependencies managed via `uv`
