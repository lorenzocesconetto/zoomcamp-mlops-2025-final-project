output "raw_data_bucket_name" {
  description = "Name of the S3 bucket for raw data"
  value       = aws_s3_bucket.raw_data.bucket
}

output "raw_data_bucket_arn" {
  description = "ARN of the S3 bucket for raw data"
  value       = aws_s3_bucket.raw_data.arn
}

output "processed_data_bucket_name" {
  description = "Name of the S3 bucket for processed data"
  value       = aws_s3_bucket.processed_data.bucket
}

output "processed_data_bucket_arn" {
  description = "ARN of the S3 bucket for processed data"
  value       = aws_s3_bucket.processed_data.arn
}

output "model_artifacts_bucket_name" {
  description = "Name of the S3 bucket for model artifacts"
  value       = aws_s3_bucket.model_artifacts.bucket
}

output "model_artifacts_bucket_arn" {
  description = "ARN of the S3 bucket for model artifacts"
  value       = aws_s3_bucket.model_artifacts.arn
}

output "pipeline_code_bucket_name" {
  description = "Name of the S3 bucket for pipeline code"
  value       = aws_s3_bucket.pipeline_code.bucket
}

output "pipeline_code_bucket_arn" {
  description = "ARN of the S3 bucket for pipeline code"
  value       = aws_s3_bucket.pipeline_code.arn
}

output "sagemaker_execution_role_arn" {
  description = "ARN of the SageMaker execution role"
  value       = aws_iam_role.sagemaker_execution_role.arn
}

output "sagemaker_pipeline_role_arn" {
  description = "ARN of the SageMaker pipeline role"
  value       = aws_iam_role.sagemaker_pipeline_role.arn
}

output "project_name" {
  description = "Project name"
  value       = var.project_name
}

output "environment" {
  description = "Environment name"
  value       = var.environment
}

output "aws_region" {
  description = "AWS region"
  value       = var.aws_region
}

output "pipeline_code_objects" {
  description = "Map of pipeline code S3 objects"
  value = {
    app_code = aws_s3_object.app_code_tar.source_hash
  }
}
