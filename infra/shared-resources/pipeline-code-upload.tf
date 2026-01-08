##############################################################
# Upload pipeline scripts to S3
##############################################################

# Create tar.gz archive
resource "null_resource" "app_code_tar" {
  triggers = {
    # Recreate if any Python file in the app directory changes
    app_hash = sha256(join("", [for f in fileset("${path.root}/../app", "**/*.py") : filesha256("${path.root}/../app/${f}")]))
  }

  provisioner "local-exec" {
    command     = "tar -czf ${path.root}/.terraform/app-code.tar.gz -C ${path.root}/../app ."
    working_dir = path.root
  }
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

# Upload the entire app code as a tar.gz file for dependencies
resource "aws_s3_object" "app_code_tar" {
  bucket = aws_s3_bucket.pipeline_code.bucket
  key    = "app-code.tar.gz"
  source = "${path.root}/.terraform/app-code.tar.gz"
  # Use the trigger hash to force re-upload when files change
  source_hash = null_resource.app_code_tar.triggers.app_hash

  depends_on = [null_resource.app_code_tar]
}
