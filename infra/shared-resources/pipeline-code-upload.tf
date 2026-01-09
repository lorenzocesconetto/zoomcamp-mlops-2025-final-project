##############################################################
# Upload pipeline scripts to S3
##############################################################

# Create tar.gz archive for the full app code (used by training/preprocessing jobs)
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

# Upload the entire app code as a tar.gz file
resource "aws_s3_object" "app_code_tar" {
  bucket = aws_s3_bucket.pipeline_code.bucket
  key    = "app-code.tar.gz"
  source = "${path.root}/.terraform/app-code.tar.gz"
  # Use the trigger hash to force re-upload when files change
  source_hash = null_resource.app_code_tar.triggers.app_hash

  depends_on = [null_resource.app_code_tar]
}

##############################################################
# Upload inference script separately for SageMaker endpoint
##############################################################
# Create tar.gz archive for the inference script only
resource "null_resource" "inference_code_tar" {
  triggers = {
    # Recreate if the inference script changes
    inference_hash = filesha256("${path.root}/../app/entrypoints/sagemaker/inference.py")
  }

  provisioner "local-exec" {
    command     = "tar -czf ${path.root}/.terraform/inference-code.tar.gz -C ${path.root}/../app/entrypoints/sagemaker inference.py"
    working_dir = path.root
  }
}

# Upload the inference code as a tar.gz file
resource "aws_s3_object" "inference_code_tar" {
  bucket = aws_s3_bucket.pipeline_code.bucket
  key    = "inference-code.tar.gz"
  source = "${path.root}/.terraform/inference-code.tar.gz"
  # Use the trigger hash to force re-upload when the script changes
  source_hash = null_resource.inference_code_tar.triggers.inference_hash

  depends_on = [null_resource.inference_code_tar]
}
