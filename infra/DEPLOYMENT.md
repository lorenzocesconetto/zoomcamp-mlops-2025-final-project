# Deployment Guide - Crypto Price Prediction MLOps Infrastructure

TODO: Review this documentation file

This guide explains how to deploy and use the AWS SageMaker infrastructure for the crypto price prediction project.

## Overview

The infrastructure creates a complete MLOps pipeline with:

- **Data Storage**: S3 buckets for raw data, processed data, model artifacts, and code
- **Model Training**: SageMaker Pipeline with preprocessing, training, evaluation, and registration steps
- **Model Deployment**: SageMaker Endpoints with auto-scaling and blue/green deployment
- **Model Monitoring**: Experiment tracking, model registry, CloudWatch dashboards, and alerting

## Prerequisites

1. **AWS CLI configured** with appropriate credentials
2. **Terraform** >= 1.0 installed
3. **AWS permissions** for SageMaker, S3, IAM, CloudWatch, Lambda, and SNS
4. **Binance API keys** (for data collection)

## Step 1: Deploy Infrastructure

### 1.1 Configure Terraform Variables

```bash
cd infra
cp terraform.tfvars.example terraform.tfvars
```

Edit `terraform.tfvars` with your settings:

```hcl
# Basic Configuration
project_name = "crypto-price-prediction"
environment  = "dev"
aws_region   = "us-east-1"

# Instance Types (adjust based on your needs and budget)
training_instance_type    = "ml.m5.large"      # For model training
processing_instance_type  = "ml.m5.large"      # For data processing
inference_instance_type   = "ml.m5.large"      # For real-time inference

# Auto Scaling (adjust based on expected traffic)
min_capacity                    = 1             # Minimum instances
max_capacity                    = 3             # Maximum instances
target_invocations_per_instance = 100           # Scale trigger

# Monitoring
enable_data_capture     = true                 # Enable model monitoring
data_capture_percentage = 20                   # Capture 20% of requests
```

### 1.2 Deploy with Terraform

```bash
# Initialize and deploy
make setup-dev

# Or manually:
terraform init
terraform plan
terraform apply
```

### 1.3 Verify Deployment

Check the outputs to get important resource names:

```bash
terraform output
```

You should see outputs like:

- S3 bucket names
- SageMaker experiment and model registry names
- Training pipeline ARN
- Endpoint URL

## Step 2: Prepare Pipeline Code

### 2.1 Upload Pipeline Scripts

The infrastructure expects these scripts in the pipeline code S3 bucket:

```bash
# Get the bucket name from Terraform output
PIPELINE_BUCKET=$(terraform output -raw pipeline_code_bucket_name)

# Upload sample scripts (modify as needed for your use case)
aws s3 cp sample-pipeline-code/preprocessing.py s3://$PIPELINE_BUCKET/preprocessing.py
aws s3 cp sample-pipeline-code/train.py s3://$PIPELINE_BUCKET/train.py
aws s3 cp sample-pipeline-code/evaluate.py s3://$PIPELINE_BUCKET/evaluate.py
aws s3 cp sample-pipeline-code/inference.py s3://$PIPELINE_BUCKET/inference.py
```

### 2.2 Adapt Your Existing Code

You can integrate your existing `app/` code with the SageMaker pipeline:

**Option A: Modify Existing Adapters**

Update your existing adapters to work with SageMaker:

```python
# In app/adapters/trainers/sagemaker_trainer.py
from app.domain.ports.trainer import Trainer
import boto3

class SageMakerTrainer(Trainer):
    def __init__(self, pipeline_name: str):
        self.sagemaker = boto3.client('sagemaker')
        self.pipeline_name = pipeline_name

    def execute(self, X_train, X_test, y_train, y_test):
        # Upload data to S3 and trigger SageMaker pipeline
        # Implementation details...
        pass
```

**Option B: Use Pipeline Scripts**

The sample scripts (`preprocessing.py`, `train.py`, etc.) contain the same logic as your existing adapters but adapted for SageMaker. You can:

1. Copy logic from your existing adapters
2. Modify the sample scripts
3. Upload to S3 and use with the pipeline

## Step 3: Run Training Pipeline

### 3.1 Upload Training Data

```bash
# Get the raw data bucket name
RAW_BUCKET=$(terraform output -raw raw_data_bucket_name)

# Upload your OHLCV data
aws s3 cp your_crypto_data.csv s3://$RAW_BUCKET/crypto_data.csv
```

### 3.2 Execute Pipeline

```bash
# Get pipeline name from Terraform output
PIPELINE_NAME=$(terraform output -raw training_pipeline_name)

# Start pipeline execution
aws sagemaker start-pipeline-execution \
    --pipeline-name $PIPELINE_NAME \
    --pipeline-parameters Name=ModelApprovalStatus,Value=PendingManualApproval
```

### 3.3 Monitor Pipeline Execution

```bash
# List pipeline executions
aws sagemaker list-pipeline-executions --pipeline-name $PIPELINE_NAME

# Get execution details
aws sagemaker describe-pipeline-execution --pipeline-execution-arn <execution-arn>
```

## Step 4: Model Deployment

### 4.1 Approve Model in Registry

Once training completes, approve the model for deployment:

```bash
# List model packages
MODEL_PACKAGE_GROUP=$(terraform output -raw model_package_group_name)
aws sagemaker list-model-packages --model-package-group-name $MODEL_PACKAGE_GROUP

# Approve a model (replace with actual model package ARN)
aws sagemaker update-model-package \
    --model-package-arn <model-package-arn> \
    --model-approval-status Approved
```

### 4.2 Deploy Model

The infrastructure automatically creates an endpoint, but you can trigger blue/green deployment:

```bash
# Get Lambda function name
LAMBDA_FUNC=$(terraform output -raw blue_green_lambda_function_name)

# Trigger deployment of latest approved model
aws lambda invoke \
    --function-name $LAMBDA_FUNC \
    --payload '{}' \
    response.json

cat response.json
```

## Step 5: Model Inference

### 5.1 Test Endpoint

```python
import boto3
import json
import pandas as pd

# Get endpoint name from Terraform output
endpoint_name = "crypto-price-prediction-dev-endpoint"  # Replace with actual name

runtime = boto3.client('sagemaker-runtime')

# Prepare sample data (replace with real feature data)
sample_data = {
    "instances": [
        {
            "open": 50000.0,
            "high": 51000.0,
            "low": 49500.0,
            "close": 50500.0,
            "volume": 1000000.0,
            # Add all your features here...
        }
    ]
}

# Make prediction
response = runtime.invoke_endpoint(
    EndpointName=endpoint_name,
    ContentType='application/json',
    Body=json.dumps(sample_data)
)

result = json.loads(response['Body'].read().decode())
print("Prediction:", result)
```

### 5.2 Integrate with Your Application

Update your existing local training script to use the endpoint:

```python
# In app/entrypoints/command_line/run_cloud.py
import boto3
import json

def predict_with_sagemaker(features_df: pd.DataFrame, endpoint_name: str):
    runtime = boto3.client('sagemaker-runtime')

    # Convert DataFrame to the expected format
    instances = features_df.to_dict('records')
    payload = {"instances": instances}

    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='application/json',
        Body=json.dumps(payload)
    )

    return json.loads(response['Body'].read().decode())
```

## Step 6: Monitoring and Maintenance

### 6.1 CloudWatch Dashboard

Access your ML pipeline dashboard:

```bash
# Get dashboard name
DASHBOARD_NAME=$(terraform output -raw cloudwatch_dashboard_name)
echo "Dashboard: https://console.aws.amazon.com/cloudwatch/home?region=us-east-1#dashboards:name=$DASHBOARD_NAME"
```

### 6.2 Set Up Alerts

Configure SNS subscriptions for alerts:

```bash
# Get SNS topic ARN
SNS_TOPIC=$(terraform output -raw sns_topic_arn)

# Subscribe to alerts (replace with your email)
aws sns subscribe \
    --topic-arn $SNS_TOPIC \
    --protocol email \
    --notification-endpoint your-email@example.com
```

### 6.3 Monitor Data Drift

If you enabled data capture, monitor model performance:

```python
import boto3
import pandas as pd

# Analyze captured data
s3 = boto3.client('s3')
bucket_name = "your-model-artifacts-bucket"
prefix = "data-capture/"

# List captured data files
response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
for obj in response.get('Contents', []):
    print(f"Captured data: {obj['Key']}")
```

## Step 7: CI/CD Integration

### 7.1 Model Training Pipeline Trigger

Create `.github/workflows/train-model.yml`:

```yaml
name: Train Model

on:
  push:
    branches: [main]
    paths: ["app/**", "infra/sample-pipeline-code/**"]

jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v1
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: us-east-1

      - name: Upload pipeline code
        run: |
          aws s3 sync infra/sample-pipeline-code/ s3://$(terraform output -raw pipeline_code_bucket_name)/

      - name: Trigger training pipeline
        run: |
          aws sagemaker start-pipeline-execution \
            --pipeline-name $(terraform output -raw training_pipeline_name)
```

## Step 8: Cost Optimization

### 8.1 Choose Right Instance Types

- **Development**: `ml.m5.large` ($0.115/hour)
- **Production**: `ml.c5.xlarge` ($0.192/hour) for CPU-intensive workloads
- **GPU Training**: `ml.p3.2xlarge` ($3.06/hour) for large models

### 8.2 Optimize Auto Scaling

```hcl
# In terraform.tfvars
min_capacity = 0  # Scale to zero when not in use
max_capacity = 2  # Limit maximum instances
target_invocations_per_instance = 200  # Higher threshold = fewer instances
```

### 8.3 Schedule Training

Use EventBridge to schedule training during off-peak hours:

```bash
# Create a rule to trigger training daily at 2 AM UTC
aws events put-rule \
    --name crypto-daily-training \
    --schedule-expression "cron(0 2 * * ? *)"
```

## Step 9: Troubleshooting

### 9.1 Common Issues

**Pipeline Execution Fails:**

```bash
# Check logs
aws logs describe-log-streams \
    --log-group-name /aws/sagemaker/ProcessingJobs

# Get specific execution details
aws sagemaker describe-pipeline-execution \
    --pipeline-execution-arn <arn>
```

**Endpoint Deployment Issues:**

```bash
# Check endpoint status
aws sagemaker describe-endpoint --endpoint-name <endpoint-name>

# Check logs
aws logs describe-log-streams \
    --log-group-name /aws/sagemaker/Endpoints/crypto-price-prediction-dev
```

**Permission Errors:**

- Verify IAM roles have correct policies
- Check trust relationships between services
- Ensure S3 bucket policies allow SageMaker access

### 9.2 Debug Commands

```bash
# List all SageMaker resources
aws sagemaker list-training-jobs
aws sagemaker list-processing-jobs
aws sagemaker list-endpoints
aws sagemaker list-model-packages --model-package-group-name <group-name>

# Check resource status
aws sagemaker describe-training-job --training-job-name <job-name>
aws sagemaker describe-processing-job --processing-job-name <job-name>
```

## Step 10: Cleanup

When you're done, clean up resources to avoid costs:

```bash
cd infra
make destroy

# Or manually:
terraform destroy
```

**Warning**: This will delete all resources including S3 buckets and their contents. Make sure to backup any important data first.

## Next Steps

1. **Hyperparameter Tuning**: Use SageMaker Hyperparameter Tuning Jobs
2. **Model Monitoring**: Implement data drift detection and model quality monitoring
3. **A/B Testing**: Deploy multiple model variants for comparison
4. **Batch Inference**: Set up SageMaker Batch Transform for bulk predictions
5. **Feature Store**: Use SageMaker Feature Store for feature management

## Support

For issues:

1. Check CloudWatch logs for detailed error messages
2. Review the [SageMaker Developer Guide](https://docs.aws.amazon.com/sagemaker/)
3. Open issues in the project repository
4. Check the infrastructure README.md for detailed configuration options
