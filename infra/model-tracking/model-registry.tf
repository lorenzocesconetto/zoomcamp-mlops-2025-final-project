# SageMaker Model Package Group for model registry
resource "aws_sagemaker_model_package_group" "crypto_prediction_models" {
  model_package_group_name        = "${var.project_name}-${var.environment}-models"
  model_package_group_description = "Model registry for crypto price prediction models"
}

# Model Package Group Policy for cross-account access (if needed)
resource "aws_sagemaker_model_package_group_policy" "crypto_prediction_models_policy" {
  model_package_group_name = aws_sagemaker_model_package_group.crypto_prediction_models.model_package_group_name

  resource_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "AddPermissionsForModelRegistry"
        Effect = "Allow"
        Principal = {
          AWS = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:root"
        }
        Action = [
          "sagemaker:DescribeModelPackage",
          "sagemaker:ListModelPackages",
          "sagemaker:UpdateModelPackage",
          "sagemaker:CreateModel"
        ]
        Resource = "arn:aws:sagemaker:${var.aws_region}:${data.aws_caller_identity.current.account_id}:model-package/${aws_sagemaker_model_package_group.crypto_prediction_models.model_package_group_name}/*"
      }
    ]
  })
}
