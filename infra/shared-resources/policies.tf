resource "aws_iam_policy" "sagemaker_cloudwatch_policy" {
  name        = "${var.project_name}-${var.environment}-sagemaker-cloudwatch-policy"
  description = "Policy for SageMaker to write CloudWatch logs"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
          "logs:DescribeLogStreams",
          "logs:TagLogGroup",
          "logs:UntagLogGroup",
          "logs:UntagResource",
          "logs:TagResource"
        ]
        Resource = "*"
      }
    ]
  })
}

resource "aws_iam_policy" "sagemaker_s3_policy" {
  name        = "${var.project_name}-${var.environment}-sagemaker-s3-policy"
  description = "Policy for SageMaker to access project S3 buckets"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket",
          "s3:PutObjectTagging",
          "s3:GetObjectTagging",
          "s3:DeleteObjectTagging"
        ]
        Resource = [
          aws_s3_bucket.raw_data.arn,
          "${aws_s3_bucket.raw_data.arn}/*",
          aws_s3_bucket.processed_data.arn,
          "${aws_s3_bucket.processed_data.arn}/*",
          aws_s3_bucket.model_artifacts.arn,
          "${aws_s3_bucket.model_artifacts.arn}/*",
          aws_s3_bucket.pipeline_code.arn,
          "${aws_s3_bucket.pipeline_code.arn}/*"
        ]
      }
    ]
  })

  tags = var.tags
}

resource "aws_iam_policy" "sagemaker_pipeline_policy" {
  name        = "${var.project_name}-${var.environment}-sagemaker-pipeline-policy"
  description = "Policy for SageMaker Pipelines to manage training jobs and experiments"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "sagemaker:CreateTrainingJob",
          "sagemaker:DescribeTrainingJob",
          "sagemaker:StopTrainingJob",
          "sagemaker:CreateProcessingJob",
          "sagemaker:DescribeProcessingJob",
          "sagemaker:StopProcessingJob",
          "sagemaker:CreateModel",
          "sagemaker:CreateEndpoint",
          "sagemaker:CreateEndpointConfig",
          "sagemaker:DescribeEndpoint",
          "sagemaker:InvokeEndpoint",
          "sagemaker:UpdateEndpoint",
          "sagemaker:DeleteEndpoint",
          "sagemaker:DeleteEndpointConfig",
          "sagemaker:DescribeModel",
          "sagemaker:DeleteModel",
          "sagemaker:CreateExperiment",
          "sagemaker:CreateTrial",
          "sagemaker:CreateTrialComponent",
          "sagemaker:DescribeExperiment",
          "sagemaker:DescribeTrial",
          "sagemaker:DescribeTrialComponent",
          "sagemaker:ListExperiments",
          "sagemaker:ListTrials",
          "sagemaker:ListTrialComponents",
          "sagemaker:AssociateTrialComponent",
          "sagemaker:DisassociateTrialComponent"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "iam:PassRole"
        ]
        Resource = aws_iam_role.sagemaker_execution_role.arn
      }
    ]
  })

  tags = var.tags
}
