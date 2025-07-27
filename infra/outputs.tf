# Shared Resources Outputs
output "raw_data_bucket_name" {
  description = "Name of the S3 bucket for raw data"
  value       = module.shared_resources.raw_data_bucket_name
}

output "processed_data_bucket_name" {
  description = "Name of the S3 bucket for processed data"
  value       = module.shared_resources.processed_data_bucket_name
}

output "model_artifacts_bucket_name" {
  description = "Name of the S3 bucket for model artifacts"
  value       = module.shared_resources.model_artifacts_bucket_name
}

output "pipeline_code_bucket_name" {
  description = "Name of the S3 bucket for pipeline code"
  value       = module.shared_resources.pipeline_code_bucket_name
}

output "sagemaker_execution_role_arn" {
  description = "ARN of the SageMaker execution role"
  value       = module.shared_resources.sagemaker_execution_role_arn
}

output "sagemaker_pipeline_role_arn" {
  description = "ARN of the SageMaker pipeline role"
  value       = module.shared_resources.sagemaker_pipeline_role_arn
}

# Model Tracking Outputs
output "experiment_name" {
  description = "Name of the SageMaker experiment"
  value       = module.model_tracking.experiment_name
}

output "model_package_group_name" {
  description = "Name of the SageMaker model package group"
  value       = module.model_tracking.model_package_group_name
}

output "cloudwatch_dashboard_name" {
  description = "Name of the CloudWatch dashboard"
  value       = module.model_tracking.dashboard_name
}

output "sns_topic_arn" {
  description = "ARN of the SNS topic for ML alerts"
  value       = module.model_tracking.sns_topic_arn
}

# Model Training Outputs
output "training_pipeline_name" {
  description = "Name of the SageMaker training pipeline"
  value       = module.model_training.pipeline_name
}

output "training_pipeline_arn" {
  description = "ARN of the SageMaker training pipeline"
  value       = module.model_training.pipeline_arn
}

# Model Deployment Outputs
output "model_endpoint_name" {
  description = "Name of the SageMaker endpoint"
  value       = module.model_deployment.endpoint_name
}

output "model_endpoint_url" {
  description = "URL for invoking the SageMaker endpoint"
  value       = module.model_deployment.endpoint_url
}

output "blue_green_lambda_function_name" {
  description = "Name of the blue/green deployment Lambda function"
  value       = module.model_deployment.blue_green_lambda_function_name
}

# Environment Information
output "environment" {
  description = "Environment name"
  value       = var.environment
}

output "aws_region" {
  description = "AWS region"
  value       = local.aws_region
}

output "project_name" {
  description = "Project name"
  value       = local.project_name
}
