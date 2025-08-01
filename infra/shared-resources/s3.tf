##############################################################
# S3 bucket for raw data
##############################################################
resource "aws_s3_bucket" "raw_data" {
  bucket = "${var.project_name}-${var.environment}-raw-data"
}

resource "aws_s3_bucket_versioning" "raw_data" {
  bucket = aws_s3_bucket.raw_data.id
  versioning_configuration {
    status = "Enabled"
  }
}

##############################################################
# S3 bucket for processed data
##############################################################
resource "aws_s3_bucket" "processed_data" {
  bucket = "${var.project_name}-${var.environment}-processed-data"
}

resource "aws_s3_bucket_versioning" "processed_data" {
  bucket = aws_s3_bucket.processed_data.id
  versioning_configuration {
    status = "Enabled"
  }
}

##############################################################
# S3 bucket for model artifacts
##############################################################
resource "aws_s3_bucket" "model_artifacts" {
  bucket = "${var.project_name}-${var.environment}-model-artifacts"
}

resource "aws_s3_bucket_versioning" "model_artifacts" {
  bucket = aws_s3_bucket.model_artifacts.id
  versioning_configuration {
    status = "Enabled"
  }
}

##############################################################
# S3 bucket for pipeline code
##############################################################
resource "aws_s3_bucket" "pipeline_code" {
  bucket = "${var.project_name}-${var.environment}-pipeline-code"
}

resource "aws_s3_bucket_versioning" "pipeline_code" {
  bucket = aws_s3_bucket.pipeline_code.id
  versioning_configuration {
    status = "Enabled"
  }
}
