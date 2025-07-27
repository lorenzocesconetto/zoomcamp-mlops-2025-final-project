##############################################################
# IAM role for SageMaker execution
##############################################################
resource "aws_iam_role" "sagemaker_execution_role" {
  name = "${var.project_name}-${var.environment}-sagemaker-execution-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "sagemaker.amazonaws.com"
        }
      }
    ]
  })

  tags = var.tags
}

resource "aws_iam_role_policy_attachment" "sagemaker_execution_policy" {
  role       = aws_iam_role.sagemaker_execution_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
}

resource "aws_iam_role_policy_attachment" "sagemaker_s3_policy" {
  role       = aws_iam_role.sagemaker_execution_role.name
  policy_arn = aws_iam_policy.sagemaker_s3_policy.arn
}

resource "aws_iam_role_policy_attachment" "sagemaker_cloudwatch_policy" {
  role       = aws_iam_role.sagemaker_execution_role.name
  policy_arn = aws_iam_policy.sagemaker_cloudwatch_policy.arn
}

##############################################################
# IAM role for SageMaker Pipelines
##############################################################
resource "aws_iam_role" "sagemaker_pipeline_role" {
  name = "${var.project_name}-${var.environment}-sagemaker-pipeline-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "sagemaker.amazonaws.com"
        }
      }
    ]
  })

  tags = var.tags
}

resource "aws_iam_role_policy_attachment" "sagemaker_pipeline_policy" {
  role       = aws_iam_role.sagemaker_pipeline_role.name
  policy_arn = aws_iam_policy.sagemaker_pipeline_policy.arn
}

resource "aws_iam_role_policy_attachment" "sagemaker_pipeline_s3_policy" {
  role       = aws_iam_role.sagemaker_pipeline_role.name
  policy_arn = aws_iam_policy.sagemaker_s3_policy.arn
}

resource "aws_iam_role_policy_attachment" "sagemaker_pipeline_cloudwatch_policy" {
  role       = aws_iam_role.sagemaker_pipeline_role.name
  policy_arn = aws_iam_policy.sagemaker_cloudwatch_policy.arn
}
