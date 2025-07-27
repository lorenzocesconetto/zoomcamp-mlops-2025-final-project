output "pipeline_name" {
  description = "Name of the SageMaker pipeline"
  value       = aws_sagemaker_pipeline.crypto_prediction_pipeline.pipeline_name
}

output "pipeline_arn" {
  description = "ARN of the SageMaker pipeline"
  value       = aws_sagemaker_pipeline.crypto_prediction_pipeline.arn
}

output "pipeline_definition" {
  description = "Pipeline definition JSON"
  value       = aws_sagemaker_pipeline.crypto_prediction_pipeline.pipeline_definition
  sensitive   = true
}
