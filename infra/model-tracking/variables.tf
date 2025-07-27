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

variable "model_artifacts_bucket_name" {
  description = "Name of the S3 bucket for model artifacts"
  type        = string
}

variable "tags" {
  description = "Common tags to apply to all resources"
  type        = map(string)
}

variable "model_approval_status" {
  description = "Default model approval status for model registry"
  type        = string
  validation {
    condition = contains([
      "Approved",
      "Rejected",
      "PendingManualApproval"
    ], var.model_approval_status)
    error_message = "Model approval status must be one of: Approved, Rejected, PendingManualApproval."
  }
}
