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

variable "model_package_group_name" {
  description = "Name of the SageMaker model package group"
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

variable "tags" {
  description = "Common tags to apply to all resources"
  type        = map(string)
}

variable "inference_instance_type" {
  description = "Instance type for inference endpoints"
  type        = string
}

variable "initial_instance_count" {
  description = "Initial number of instances for the endpoint"
  type        = number
}

variable "max_capacity" {
  description = "Maximum number of instances for auto scaling"
  type        = number
}

variable "min_capacity" {
  description = "Minimum number of instances for auto scaling"
  type        = number
}

variable "target_invocations_per_instance" {
  description = "Target invocations per instance for auto scaling"
  type        = number
}

variable "enable_data_capture" {
  description = "Enable data capture for the endpoint"
  type        = bool
}

variable "data_capture_percentage" {
  description = "Percentage of requests to capture"
  type        = number
}
