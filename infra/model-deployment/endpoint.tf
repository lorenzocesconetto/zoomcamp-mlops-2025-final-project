resource "aws_sagemaker_model" "crypto_prediction_model" {
  name               = "${var.project_name}-${var.environment}-model"
  execution_role_arn = var.sagemaker_execution_role_arn

  primary_container {
    image          = "683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:0.23-1-cpu-py3"
    model_data_url = "s3://${var.model_artifacts_bucket_name}/default-model/model.tar.gz"

    environment = {
      "SAGEMAKER_PROGRAM"          = "inference.py"
      "SAGEMAKER_SUBMIT_DIRECTORY" = "s3://${var.pipeline_code_bucket_name}/app-code.zip"
      "SAGEMAKER_REGION"           = var.aws_region
    }
  }
}

resource "aws_sagemaker_endpoint_configuration" "crypto_prediction_endpoint_config" {
  name = "${var.project_name}-${var.environment}-endpoint-config"

  production_variants {
    variant_name           = "primary"
    model_name             = aws_sagemaker_model.crypto_prediction_model.name
    initial_instance_count = var.initial_instance_count
    instance_type          = var.inference_instance_type
    initial_variant_weight = 1
  }

  dynamic "data_capture_config" {
    for_each = var.enable_data_capture ? [1] : []
    content {
      enable_capture              = true
      initial_sampling_percentage = var.data_capture_percentage
      destination_s3_uri          = "s3://${var.model_artifacts_bucket_name}/data-capture/"
      capture_options {
        capture_mode = "Input"
      }
      capture_options {
        capture_mode = "Output"
      }
      capture_content_type_header {
        csv_content_types  = ["text/csv"]
        json_content_types = ["application/json"]
      }
    }
  }

  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_sagemaker_endpoint" "crypto_prediction_endpoint" {
  name                 = "${var.project_name}-${var.environment}-endpoint"
  endpoint_config_name = aws_sagemaker_endpoint_configuration.crypto_prediction_endpoint_config.name
}

resource "aws_appautoscaling_target" "sagemaker_target" {
  max_capacity       = var.max_capacity
  min_capacity       = var.min_capacity
  resource_id        = "endpoint/${aws_sagemaker_endpoint.crypto_prediction_endpoint.name}/variant/primary"
  scalable_dimension = "sagemaker:variant:DesiredInstanceCount"
  service_namespace  = "sagemaker"
}

resource "aws_appautoscaling_policy" "sagemaker_scaling_policy" {
  name               = "${var.project_name}-${var.environment}-scaling-policy"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.sagemaker_target.resource_id
  scalable_dimension = aws_appautoscaling_target.sagemaker_target.scalable_dimension
  service_namespace  = aws_appautoscaling_target.sagemaker_target.service_namespace

  target_tracking_scaling_policy_configuration {
    target_value = var.target_invocations_per_instance

    predefined_metric_specification {
      predefined_metric_type = "SageMakerVariantInvocationsPerInstance"
    }

    scale_out_cooldown = 300 # 5 minutes
    scale_in_cooldown  = 300 # 5 minutes
  }
}

# Lambda function for blue/green deployments (optional)
resource "aws_lambda_function" "blue_green_deployment" {
  filename      = "blue_green_deployment.zip"
  function_name = "${var.project_name}-${var.environment}-blue-green-deploy"
  role          = aws_iam_role.lambda_execution_role.arn
  handler       = "lambda_function.lambda_handler"
  runtime       = "python3.9"
  timeout       = 300

  environment {
    variables = {
      ENDPOINT_NAME            = aws_sagemaker_endpoint.crypto_prediction_endpoint.name
      MODEL_PACKAGE_GROUP_NAME = var.model_package_group_name
    }
  }
}

# IAM role for Lambda function
resource "aws_iam_role" "lambda_execution_role" {
  name = "${var.project_name}-${var.environment}-lambda-execution-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "lambda.amazonaws.com"
        }
      }
    ]
  })
}

# IAM policy for Lambda function
resource "aws_iam_policy" "lambda_sagemaker_policy" {
  name        = "${var.project_name}-${var.environment}-lambda-sagemaker-policy"
  description = "Policy for Lambda to manage SageMaker endpoints"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "sagemaker:CreateEndpointConfig",
          "sagemaker:CreateEndpoint",
          "sagemaker:UpdateEndpoint",
          "sagemaker:DeleteEndpoint",
          "sagemaker:DeleteEndpointConfig",
          "sagemaker:DescribeEndpoint",
          "sagemaker:DescribeEndpointConfig",
          "sagemaker:DescribeModel",
          "sagemaker:ListModelPackages",
          "sagemaker:DescribeModelPackage",
          "sagemaker:CreateModel",
          "sagemaker:DeleteModel"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "iam:PassRole"
        ]
        Resource = var.sagemaker_execution_role_arn
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "lambda_sagemaker_policy" {
  role       = aws_iam_role.lambda_execution_role.name
  policy_arn = aws_iam_policy.lambda_sagemaker_policy.arn
}

# Create a placeholder zip file for the Lambda function
resource "local_file" "lambda_code" {
  content  = <<EOF
import json
import boto3
import os

def lambda_handler(event, context):
    """
    Lambda function for blue/green deployment of SageMaker models
    """
    sagemaker = boto3.client('sagemaker')
    endpoint_name = os.environ['ENDPOINT_NAME']
    model_package_group_name = os.environ['MODEL_PACKAGE_GROUP_NAME']

    try:
        # Get the latest approved model from model registry
        response = sagemaker.list_model_packages(
            ModelPackageGroupName=model_package_group_name,
            ModelApprovalStatus='Approved',
            SortBy='CreationTime',
            SortOrder='Descending',
            MaxResults=1
        )

        if not response['ModelPackageSummaryList']:
            return {
                'statusCode': 400,
                'body': json.dumps('No approved models found in model registry')
            }

        latest_model_package_arn = response['ModelPackageSummaryList'][0]['ModelPackageArn']

        # TODO: Implement blue/green deployment logic
        # 1. Create new endpoint configuration with the latest model
        # 2. Update endpoint to use new configuration
        # 3. Monitor deployment and rollback if needed

        return {
            'statusCode': 200,
            'body': json.dumps(f'Deployment initiated for model: {latest_model_package_arn}')
        }

    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps(f'Error: {str(e)}')
        }
EOF
  filename = "blue_green_deployment.py"
}

# Create zip file for Lambda deployment
data "archive_file" "lambda_zip" {
  type        = "zip"
  source_file = local_file.lambda_code.filename
  output_path = "blue_green_deployment.zip"
  depends_on  = [local_file.lambda_code]
}

# Data source for current AWS region
data "aws_region" "current" {}
