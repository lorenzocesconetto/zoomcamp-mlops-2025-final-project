output "model_name" {
  description = "Name of the SageMaker model"
  value       = aws_sagemaker_model.crypto_prediction_model.name
}

output "model_arn" {
  description = "ARN of the SageMaker model"
  value       = aws_sagemaker_model.crypto_prediction_model.arn
}

output "endpoint_config_name" {
  description = "Name of the SageMaker endpoint configuration"
  value       = aws_sagemaker_endpoint_configuration.crypto_prediction_endpoint_config.name
}

output "endpoint_config_arn" {
  description = "ARN of the SageMaker endpoint configuration"
  value       = aws_sagemaker_endpoint_configuration.crypto_prediction_endpoint_config.arn
}

output "endpoint_name" {
  description = "Name of the SageMaker endpoint"
  value       = aws_sagemaker_endpoint.crypto_prediction_endpoint.name
}

output "endpoint_arn" {
  description = "ARN of the SageMaker endpoint"
  value       = aws_sagemaker_endpoint.crypto_prediction_endpoint.arn
}

output "endpoint_url" {
  description = "URL for invoking the SageMaker endpoint"
  value       = "https://runtime.sagemaker.${var.aws_region}.amazonaws.com/endpoints/${aws_sagemaker_endpoint.crypto_prediction_endpoint.name}/invocations"
}

output "auto_scaling_target_arn" {
  description = "ARN of the auto scaling target"
  value       = aws_appautoscaling_target.sagemaker_target.arn
}

output "auto_scaling_policy_arn" {
  description = "ARN of the auto scaling policy"
  value       = aws_appautoscaling_policy.sagemaker_scaling_policy.arn
}

output "blue_green_lambda_function_name" {
  description = "Name of the blue/green deployment Lambda function"
  value       = aws_lambda_function.blue_green_deployment.function_name
}

output "blue_green_lambda_function_arn" {
  description = "ARN of the blue/green deployment Lambda function"
  value       = aws_lambda_function.blue_green_deployment.arn
}
