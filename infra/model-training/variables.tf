variable "project_name" {
  description = "Name of the project"
  type        = string
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
}

variable "aws_region" {
  description = "AWS region"
  type        = string
}

variable "sagemaker_execution_role_arn" {
  description = "ARN of the SageMaker execution role"
  type        = string
}

variable "sagemaker_pipeline_role_arn" {
  description = "ARN of the SageMaker pipeline role"
  type        = string
}

variable "raw_data_bucket_name" {
  description = "Name of the S3 bucket for raw data"
  type        = string
}

variable "processed_data_bucket_name" {
  description = "Name of the S3 bucket for processed data"
  type        = string
}

variable "model_artifacts_bucket_name" {
  description = "Name of the S3 bucket for model artifacts"
  type        = string
}

variable "pipeline_code_bucket_name" {
  description = "Name of the S3 bucket for pipeline code"
  type        = string
}

variable "experiment_name" {
  description = "Name of the SageMaker experiment"
  type        = string
}

variable "model_package_group_name" {
  description = "Name of the SageMaker model package group"
  type        = string
}

variable "tags" {
  description = "Common tags to apply to all resources"
  type        = map(string)
}

variable "training_instance_type" {
  description = "Instance type for training jobs"
  type        = string
}

variable "processing_instance_type" {
  description = "Instance type for processing jobs"
  type        = string
}

variable "max_runtime_in_seconds" {
  description = "Maximum runtime for training jobs in seconds"
  type        = number
}

variable "pipeline_code_objects" {
  description = "Map of pipeline code S3 objects to ensure they're uploaded before pipeline creation"
  type        = map(string)
}
