# Project Configuration
variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "dev"
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be one of: dev, staging, prod."
  }
}

# Training Configuration
variable "training_instance_type" {
  description = "Instance type for training jobs"
  type        = string
  default     = "ml.m5.large"
  validation {
    condition     = can(regex("^ml\\.", var.training_instance_type))
    error_message = "Training instance type must be a valid SageMaker ML instance type."
  }
}

variable "processing_instance_type" {
  description = "Instance type for processing jobs"
  type        = string
  default     = "ml.m5.large"
  validation {
    condition     = can(regex("^ml\\.", var.processing_instance_type))
    error_message = "Processing instance type must be a valid SageMaker ML instance type."
  }
}

variable "max_runtime_in_seconds" {
  description = "Maximum runtime for training jobs in seconds"
  type        = number
  default     = 3600 # 1 hour
  validation {
    condition     = var.max_runtime_in_seconds > 0 && var.max_runtime_in_seconds <= 86400
    error_message = "Max runtime must be between 1 second and 24 hours (86400 seconds)."
  }
}

# Inference Configuration
variable "inference_instance_type" {
  description = "Instance type for inference endpoints"
  type        = string
  default     = "ml.t3.medium"
  validation {
    condition     = can(regex("^ml\\.", var.inference_instance_type))
    error_message = "Inference instance type must be a valid SageMaker ML instance type."
  }
}

variable "initial_instance_count" {
  description = "Initial number of instances for the endpoint"
  type        = number
  default     = 1
  validation {
    condition     = var.initial_instance_count > 0 && var.initial_instance_count <= 10
    error_message = "Initial instance count must be between 1 and 10."
  }
}

# Auto Scaling Configuration
variable "max_capacity" {
  description = "Maximum number of instances for auto scaling"
  type        = number
  default     = 3
  validation {
    condition     = var.max_capacity > 0 && var.max_capacity <= 20
    error_message = "Max capacity must be between 1 and 20."
  }
}

variable "min_capacity" {
  description = "Minimum number of instances for auto scaling"
  type        = number
  default     = 1
  validation {
    condition     = var.min_capacity > 0
    error_message = "Min capacity must be greater than 0."
  }
}

variable "target_invocations_per_instance" {
  description = "Target invocations per instance for auto scaling"
  type        = number
  default     = 100
  validation {
    condition     = var.target_invocations_per_instance > 0
    error_message = "Target invocations per instance must be greater than 0."
  }
}

# Data Capture Configuration
variable "enable_data_capture" {
  description = "Enable data capture for the endpoint"
  type        = bool
  default     = true
}

variable "data_capture_percentage" {
  description = "Percentage of requests to capture"
  type        = number
  default     = 20
  validation {
    condition     = var.data_capture_percentage >= 0 && var.data_capture_percentage <= 100
    error_message = "Data capture percentage must be between 0 and 100."
  }
}

# Model Registry Configuration
variable "model_approval_status" {
  description = "Default model approval status for model registry"
  type        = string
  default     = "Approved"
  validation {
    condition = contains([
      "Approved",
      "Rejected",
      "PendingManualApproval"
    ], var.model_approval_status)
    error_message = "Model approval status must be one of: Approved, Rejected, PendingManualApproval."
  }
}
