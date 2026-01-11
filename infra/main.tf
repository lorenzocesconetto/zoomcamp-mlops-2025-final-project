# Terraform configuration
terraform {
  required_version = ">=1.10.1, <1.11"
  backend "s3" {
    # key, region and bucket are replaced dynamically during CI/CD
    key          = "crypto-price-prediction/dev/terraform.tfstate"
    region       = "us-east-1"
    bucket       = "terraform-state-bucket"
    use_lockfile = true
  }

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    local = {
      source  = "hashicorp/local"
      version = "~> 2.0"
    }
    archive = {
      source  = "hashicorp/archive"
      version = "~> 2.0"
    }
  }
}

# AWS Provider configuration
provider "aws" {
  region = local.aws_region
  default_tags {
    tags = local.common_tags
  }
}

# Shared Resources Module
module "shared_resources" {
  source = "./shared-resources"

  project_name = local.project_name
  environment  = var.environment
  aws_region   = local.aws_region
  tags         = local.common_tags
}

# Model Tracking Module
module "model_tracking" {
  source = "./model-tracking"

  project_name                 = local.project_name
  environment                  = var.environment
  aws_region                   = local.aws_region
  sagemaker_execution_role_arn = module.shared_resources.sagemaker_execution_role_arn
  model_artifacts_bucket_name  = module.shared_resources.model_artifacts_bucket_name
  model_approval_status        = var.model_approval_status
  tags                         = local.common_tags

  depends_on = [module.shared_resources]
}

# Model Training Module
module "model_training" {
  source = "./model-training"

  project_name                 = local.project_name
  environment                  = var.environment
  aws_region                   = local.aws_region
  sagemaker_execution_role_arn = module.shared_resources.sagemaker_execution_role_arn
  sagemaker_pipeline_role_arn  = module.shared_resources.sagemaker_pipeline_role_arn
  raw_data_bucket_name         = module.shared_resources.raw_data_bucket_name
  processed_data_bucket_name   = module.shared_resources.processed_data_bucket_name
  model_artifacts_bucket_name  = module.shared_resources.model_artifacts_bucket_name
  pipeline_code_bucket_name    = module.shared_resources.pipeline_code_bucket_name
  experiment_name              = local.experiment_name
  model_package_group_name     = module.model_tracking.model_package_group_name
  training_instance_type       = var.training_instance_type
  processing_instance_type     = var.processing_instance_type
  max_runtime_in_seconds       = var.max_runtime_in_seconds
  pipeline_code_objects        = module.shared_resources.pipeline_code_objects
  tags                         = local.common_tags

  depends_on = [module.shared_resources, module.model_tracking]
}

# Model Deployment Module
module "model_deployment" {
  source = "./model-deployment"

  project_name                    = local.project_name
  environment                     = var.environment
  aws_region                      = local.aws_region
  sagemaker_execution_role_arn    = module.shared_resources.sagemaker_execution_role_arn
  model_package_group_name        = module.model_tracking.model_package_group_name
  model_artifacts_bucket_name     = module.shared_resources.model_artifacts_bucket_name
  pipeline_code_bucket_name       = module.shared_resources.pipeline_code_bucket_name
  inference_instance_type         = var.inference_instance_type
  initial_instance_count          = var.initial_instance_count
  max_capacity                    = var.max_capacity
  min_capacity                    = var.min_capacity
  target_invocations_per_instance = var.target_invocations_per_instance
  enable_data_capture             = var.enable_data_capture
  data_capture_percentage         = var.data_capture_percentage
  tags                            = local.common_tags
  model_data_path                 = var.model_data_path
  inference_code_hash             = module.shared_resources.inference_code_hash

  depends_on = [module.shared_resources, module.model_tracking]
}
