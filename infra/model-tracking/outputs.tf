output "model_package_group_name" {
  description = "Name of the SageMaker model package group"
  value       = aws_sagemaker_model_package_group.crypto_prediction_models.model_package_group_name
}

output "model_package_group_arn" {
  description = "ARN of the SageMaker model package group"
  value       = aws_sagemaker_model_package_group.crypto_prediction_models.arn
}

output "training_logs_group_name" {
  description = "Name of the CloudWatch log group for training jobs"
  value       = aws_cloudwatch_log_group.sagemaker_training_logs.name
}

output "endpoint_logs_group_name" {
  description = "Name of the CloudWatch log group for endpoints"
  value       = aws_cloudwatch_log_group.sagemaker_endpoint_logs.name
}

output "processing_logs_group_name" {
  description = "Name of the CloudWatch log group for processing jobs"
  value       = aws_cloudwatch_log_group.sagemaker_processing_logs.name
}

output "dashboard_name" {
  description = "Name of the CloudWatch dashboard"
  value       = aws_cloudwatch_dashboard.ml_pipeline_dashboard.dashboard_name
}

output "sns_topic_arn" {
  description = "ARN of the SNS topic for ML alerts"
  value       = aws_sns_topic.ml_alerts.arn
}

output "model_accuracy_alarm_name" {
  description = "Name of the model accuracy CloudWatch alarm"
  value       = aws_cloudwatch_metric_alarm.model_accuracy_alarm.alarm_name
}
