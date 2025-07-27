# CloudWatch Log Group for SageMaker training jobs
resource "aws_cloudwatch_log_group" "sagemaker_training_logs" {
  name              = "/aws/sagemaker/TrainingJobs/${var.project_name}-${var.environment}"
  retention_in_days = 30
}

# CloudWatch Log Group for SageMaker endpoints
resource "aws_cloudwatch_log_group" "sagemaker_endpoint_logs" {
  name              = "/aws/sagemaker/Endpoints/${var.project_name}-${var.environment}"
  retention_in_days = 30
}

# CloudWatch Log Group for SageMaker processing jobs
resource "aws_cloudwatch_log_group" "sagemaker_processing_logs" {
  name              = "/aws/sagemaker/ProcessingJobs/${var.project_name}-${var.environment}"
  retention_in_days = 30
}

# CloudWatch Dashboard for monitoring model performance
resource "aws_cloudwatch_dashboard" "ml_pipeline_dashboard" {
  dashboard_name = "${var.project_name}-${var.environment}-ml-pipeline"

  dashboard_body = jsonencode({
    widgets = [
      {
        type   = "metric"
        x      = 0
        y      = 0
        width  = 12
        height = 6

        properties = {
          metrics = [
            ["AWS/SageMaker", "TrainingJobsCompleted"],
            ["AWS/SageMaker", "TrainingJobsFailed"],
            ["AWS/SageMaker", "TrainingJobsInProgress"]
          ]
          view    = "timeSeries"
          stacked = false
          region  = var.aws_region
          title   = "Training Job Status"
          period  = 300
        }
      },
      {
        type   = "metric"
        x      = 0
        y      = 6
        width  = 12
        height = 6

        properties = {
          metrics = [
            ["AWS/SageMaker", "EndpointInvocations"],
            ["AWS/SageMaker", "EndpointLatency"],
            ["AWS/SageMaker", "EndpointInvocationErrors"]
          ]
          view    = "timeSeries"
          stacked = false
          region  = var.aws_region
          title   = "Endpoint Performance"
          period  = 300
        }
      }
    ]
  })
}

# Custom CloudWatch metrics for model performance
resource "aws_cloudwatch_metric_alarm" "model_accuracy_alarm" {
  alarm_name          = "${var.project_name}-${var.environment}-model-accuracy-alarm"
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "ModelAccuracy"
  namespace           = "CryptoPrediction/ModelPerformance"
  period              = "300"
  statistic           = "Average"
  threshold           = "0.6"
  alarm_description   = "This metric monitors model accuracy"
  alarm_actions       = [aws_sns_topic.ml_alerts.arn]

  tags = var.tags
}

# SNS topic for ML pipeline alerts
resource "aws_sns_topic" "ml_alerts" {
  name = "${var.project_name}-${var.environment}-ml-alerts"
  tags = var.tags
}
