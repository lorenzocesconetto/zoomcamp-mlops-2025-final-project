locals {
  project_name    = "crypto-price-prediction"
  repository      = "crypto-price-prediction"
  owner           = "mlops-team"
  managed_by      = "terraform"
  aws_region      = "us-east-1"
  experiment_name = "${local.project_name}-${var.environment}-experiment"

  common_tags = {
    Project     = local.project_name
    ManagedBy   = local.managed_by
    Owner       = local.owner
    Repository  = local.repository
    Environment = var.environment
  }
}
