# Crypto Price Prediction - Infrastructure

This folder contains the Terraform infrastructure as code (IaC) for deploying a complete MLOps pipeline using AWS SageMaker for crypto price prediction.

## Architecture Overview

The infrastructure is modularized into four main components:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Shared         │    │  Model          │    │  Model          │    │  Model          │
│  Resources      │───▶│  Tracking       │───▶│  Training       │───▶│  Deployment     │
│                 │    │                 │    │                 │    │                 │
│ • S3 Buckets    │    │ • Experiments   │    │ • SageMaker     │    │ • Endpoints     │
│ • IAM Roles     │    │ • Model Registry│    │   Pipeline      │    │ • Auto Scaling  │
│ • Permissions   │    │ • CloudWatch    │    │ • Training Jobs │    │ • Blue/Green    │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Module Structure

### 1. Shared Resources (`shared-resources/`)

- **S3 Buckets**: Raw data, processed data, model artifacts, pipeline code
- **IAM Roles**: SageMaker execution and pipeline roles with appropriate permissions
- **Security**: Encryption, versioning, and access controls

### 2. Model Tracking (`model-tracking/`)

- **SageMaker Experiments**: Track training runs and hyperparameters
- **Model Registry**: Version and manage model artifacts
- **CloudWatch**: Dashboards, logs, and metrics monitoring
- **Alerting**: SNS topics for ML pipeline notifications

### 3. Model Training (`model-training/`)

- **SageMaker Pipeline**: End-to-end ML workflow orchestration
- **Processing Jobs**: Data preprocessing and feature engineering
- **Training Jobs**: Model training with experiment tracking
- **Model Evaluation**: Automated model validation and metrics
- **Model Registration**: Automatic registration to model registry

### 4. Model Deployment (`model-deployment/`)

- **SageMaker Endpoints**: Real-time inference endpoints
- **Auto Scaling**: Dynamic scaling based on traffic
- **Data Capture**: Monitor model inputs/outputs for drift detection
- **Blue/Green Deployment**: Automated deployment of new model versions

## Prerequisites

1. **AWS CLI** configured with appropriate credentials
2. **Terraform** >= 1.10.1 < 1.11 installed
3. **AWS Account** with sufficient permissions
4. **S3 Bucket** for Terraform state (optional but recommended)

## Quick Start

### 1. Configure Variables

Copy the example terraform variables file:

```bash
cp terraform.tfvars.example terraform.tfvars
```

Edit `terraform.tfvars` with your specific configuration.

### 2. Deploy Infrastructure

```bash
# Initialize Terraform
terraform init

# Plan the deployment
terraform plan

# Apply the infrastructure
terraform apply
```

### 3. Verify Deployment

After successful deployment, you should see outputs including:

- S3 bucket names for data storage
- SageMaker experiment and model registry names
- Training pipeline ARN
- Endpoint URL for inference

## Usage

### Running the Training Pipeline

1. **Upload Training Code**: Place your preprocessing, training, evaluation, and inference scripts in the pipeline code S3 bucket
2. **Upload Data**: Place raw OHLCV data in the raw data S3 bucket
3. **Execute Pipeline**: Trigger the SageMaker pipeline through AWS Console or AWS CLI

```bash
aws sagemaker start-pipeline-execution \
    --pipeline-name crypto-price-prediction-dev-training-pipeline \
    --pipeline-parameters Name=ModelApprovalStatus,Value=Approved
```

### Model Inference

Once the model is deployed, you can make predictions:

```python
import boto3
import json

runtime = boto3.client('sagemaker-runtime')

# Prepare input data
input_data = {
    "instances": [
        # Your feature data here
    ]
}

# Make prediction
response = runtime.invoke_endpoint(
    EndpointName='crypto-price-prediction-dev-endpoint',
    ContentType='application/json',
    Body=json.dumps(input_data)
)

result = json.loads(response['Body'].read().decode())
print(result)
```

### Blue/Green Deployment

Trigger automatic deployment of new approved models:

```bash
aws lambda invoke \
    --function-name crypto-price-prediction-dev-blue-green-deploy \
    --payload '{}' \
    response.json
```

## Monitoring

### CloudWatch Dashboard

Access the ML pipeline dashboard at:

```
https://console.aws.amazon.com/cloudwatch/home?region=<your-region>#dashboards:name=crypto-price-prediction-<env>-ml-pipeline
```

### Key Metrics to Monitor

- **Training Jobs**: Success/failure rates, duration
- **Endpoint Performance**: Latency, throughput, error rates
- **Model Accuracy**: Custom metrics from evaluation
- **Resource Utilization**: Instance usage, auto-scaling events

### Alerts

Configure SNS subscriptions to receive alerts when:

- Model accuracy drops below threshold
- Endpoint errors increase
- Training jobs fail

### Auto Scaling

- Set appropriate `min_capacity` and `max_capacity` based on traffic patterns
- Adjust `target_invocations_per_instance` to balance cost and performance

### Data Capture

- Reduce `data_capture_percentage` if full monitoring isn't needed
- Set appropriate S3 lifecycle policies for captured data

## Security

### IAM Roles

- **Principle of Least Privilege**: Roles have minimum required permissions
- **Cross-Service Access**: Proper trust relationships between services
- **Resource-Specific**: Policies scoped to specific S3 buckets and resources

### Data Encryption

- **S3**: Server-side encryption enabled on all buckets
- **SageMaker**: Encryption at rest and in transit
- **CloudWatch**: Encrypted log groups

### Network Security

- All SageMaker jobs run in VPC (can be configured)
- Security groups restrict access to necessary ports
- Private endpoints available for enhanced security

## Troubleshooting

### Common Issues

1. **Permission Errors**

   - Verify IAM roles have correct policies attached
   - Check cross-service trust relationships

2. **Training Job Failures**

   - Check CloudWatch logs in `/aws/sagemaker/TrainingJobs/`
   - Verify training script and dependencies

3. **Endpoint Deployment Issues**

   - Ensure model artifacts are correctly formatted
   - Check inference script for errors

4. **Pipeline Execution Failures**
   - Review pipeline definition JSON
   - Verify step dependencies and data flow

### Debugging Commands

```bash
# Check SageMaker training job logs
aws logs describe-log-streams --log-group-name /aws/sagemaker/TrainingJobs/crypto-price-prediction-dev

# List pipeline executions
aws sagemaker list-pipeline-executions --pipeline-name crypto-price-prediction-dev-training-pipeline

# Describe endpoint status
aws sagemaker describe-endpoint --endpoint-name crypto-price-prediction-dev-endpoint
```

## Cleanup

To avoid ongoing costs, destroy the infrastructure when not needed:

```bash
terraform destroy
```

**Note**: This will delete all resources including S3 buckets and their contents. Ensure you have backups of important data.
