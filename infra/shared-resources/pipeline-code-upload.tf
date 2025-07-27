##############################################################
# Upload pipeline scripts to S3
##############################################################

# Archive the entire app directory
data "archive_file" "app_code" {
  type        = "zip"
  source_dir  = "${path.root}/../app"
  output_path = "${path.root}/.terraform/app-code.zip"
}

# Upload preprocessing script
resource "aws_s3_object" "preprocessing_script" {
  bucket = aws_s3_bucket.pipeline_code.bucket
  key    = "preprocessing.py"
  source = "${path.root}/../app/entrypoints/sagemaker/preprocessing.py"
  etag   = filemd5("${path.root}/../app/entrypoints/sagemaker/preprocessing.py")
}

# Upload training script
resource "aws_s3_object" "training_script" {
  bucket = aws_s3_bucket.pipeline_code.bucket
  key    = "train.py"
  source = "${path.root}/../app/entrypoints/sagemaker/train.py"
  etag   = filemd5("${path.root}/../app/entrypoints/sagemaker/train.py")
}

# Upload evaluation script
resource "aws_s3_object" "evaluation_script" {
  bucket = aws_s3_bucket.pipeline_code.bucket
  key    = "evaluate.py"
  source = "${path.root}/../app/entrypoints/sagemaker/evaluate.py"
  etag   = filemd5("${path.root}/../app/entrypoints/sagemaker/evaluate.py")
}

# Upload inference script
resource "aws_s3_object" "inference_script" {
  bucket = aws_s3_bucket.pipeline_code.bucket
  key    = "inference.py"
  source = "${path.root}/../app/entrypoints/sagemaker/inference.py"
  etag   = filemd5("${path.root}/../app/entrypoints/sagemaker/inference.py")
}

# Upload the entire app code as a zip file for dependencies
resource "aws_s3_object" "app_code_zip" {
  bucket = aws_s3_bucket.pipeline_code.bucket
  key    = "app-code.zip"
  source = data.archive_file.app_code.output_path
  etag   = data.archive_file.app_code.output_md5
}
