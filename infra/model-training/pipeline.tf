# SageMaker Pipeline for ML training workflow
resource "aws_sagemaker_pipeline" "crypto_prediction_pipeline" {
  pipeline_name         = "${var.project_name}-${var.environment}-training-pipeline"
  pipeline_display_name = "${var.project_name}-${var.environment}-training-pipeline"
  role_arn              = var.sagemaker_pipeline_role_arn

  pipeline_definition = jsonencode({
    Version = "2020-12-01"
    Metadata = {
      GeneratedBy = "terraform"
    }
    Parameters = [
      {
        Name         = "ProcessingInstanceType"
        Type         = "String"
        DefaultValue = var.processing_instance_type
      },
      {
        Name         = "TrainingInstanceType"
        Type         = "String"
        DefaultValue = var.training_instance_type
      },
      {
        Name         = "ModelApprovalStatus"
        Type         = "String"
        DefaultValue = "PendingManualApproval"
      },
      {
        Name         = "InputDataUrl"
        Type         = "String"
        DefaultValue = "s3://${var.raw_data_bucket_name}/"
      },
      {
        Name         = "ProcessedDataUrl"
        Type         = "String"
        DefaultValue = "s3://${var.processed_data_bucket_name}/"
      }
    ]
    Steps = [
      {
        Name = "DataProcessing"
        Type = "Processing"
        Arguments = {
          ProcessingResources = {
            ClusterConfig = {
              InstanceType   = { "Get" = "Parameters.ProcessingInstanceType" }
              InstanceCount  = 1
              VolumeSizeInGB = 30
            }
          }
          AppSpecification = {
            ImageUri = "${data.aws_caller_identity.current.account_id}.dkr.ecr.${var.aws_region}.amazonaws.com/sagemaker-scikit-learn:0.23-1-cpu-py3"
            ContainerEntrypoint = [
              "python3", "/opt/ml/processing/code/preprocessing.py"
            ]
          }
          RoleArn = var.sagemaker_execution_role_arn
          ProcessingInputs = [
            {
              InputName  = "raw-data"
              AppManaged = false
              S3Input = {
                S3Uri                  = { "Get" = "Parameters.InputDataUrl" }
                LocalPath              = "/opt/ml/processing/input"
                S3DataType             = "S3Prefix"
                S3InputMode            = "File"
                S3DataDistributionType = "FullyReplicated"
                S3CompressionType      = "None"
              }
            },
            {
              InputName  = "code"
              AppManaged = false
              S3Input = {
                S3Uri                  = "s3://${var.pipeline_code_bucket_name}/preprocessing.py"
                LocalPath              = "/opt/ml/processing/code"
                S3DataType             = "S3Prefix"
                S3InputMode            = "File"
                S3DataDistributionType = "FullyReplicated"
                S3CompressionType      = "None"
              }
            }
          ]
          ProcessingOutputConfig = {
            Outputs = [
              {
                OutputName = "processed-data"
                AppManaged = false
                S3Output = {
                  S3Uri        = { "Get" = "Parameters.ProcessedDataUrl" }
                  LocalPath    = "/opt/ml/processing/output"
                  S3UploadMode = "EndOfJob"
                }
              }
            ]
          }
        }
      },
      {
        Name = "ModelTraining"
        Type = "Training"
        Arguments = {
          AlgorithmSpecification = {
            TrainingImage     = "${data.aws_caller_identity.current.account_id}.dkr.ecr.${var.aws_region}.amazonaws.com/sagemaker-scikit-learn:0.23-1-cpu-py3"
            TrainingInputMode = "File"
          }
          InputDataConfig = [
            {
              ChannelName = "training"
              DataSource = {
                S3DataSource = {
                  S3DataType             = "S3Prefix"
                  S3Uri                  = { "Get" = "Parameters.ProcessedDataUrl" }
                  S3DataDistributionType = "FullyReplicated"
                }
              }
              ContentType     = "text/csv"
              CompressionType = "None"
            }
          ]
          OutputDataConfig = {
            S3OutputPath = "s3://${var.model_artifacts_bucket_name}/training-jobs/"
          }
          ResourceConfig = {
            InstanceType   = { "Get" = "Parameters.TrainingInstanceType" }
            InstanceCount  = 1
            VolumeSizeInGB = 30
          }
          RoleArn = var.sagemaker_execution_role_arn
          StoppingCondition = {
            MaxRuntimeInSeconds = var.max_runtime_in_seconds
          }
          HyperParameters = {
            "n_estimators" = "100"
            "max_depth"    = "10"
            "random_state" = "42"
          }
          Environment = {
            "SAGEMAKER_PROGRAM"          = "train.py"
            "SAGEMAKER_SUBMIT_DIRECTORY" = "s3://${var.pipeline_code_bucket_name}/"
            "SAGEMAKER_REGION"           = var.aws_region
          }
        }
        DependsOn = ["DataProcessing"]
      },
      {
        Name = "ModelEvaluation"
        Type = "Processing"
        Arguments = {
          ProcessingResources = {
            ClusterConfig = {
              InstanceType   = { "Get" = "Parameters.ProcessingInstanceType" }
              InstanceCount  = 1
              VolumeSizeInGB = 30
            }
          }
          AppSpecification = {
            ImageUri = "${data.aws_caller_identity.current.account_id}.dkr.ecr.${var.aws_region}.amazonaws.com/sagemaker-scikit-learn:0.23-1-cpu-py3"
            ContainerEntrypoint = [
              "python3", "/opt/ml/processing/code/evaluate.py"
            ]
          }
          RoleArn = var.sagemaker_execution_role_arn
          ProcessingInputs = [
            {
              InputName  = "model"
              AppManaged = false
              S3Input = {
                S3Uri                  = "s3://${var.model_artifacts_bucket_name}/training-jobs/"
                LocalPath              = "/opt/ml/processing/model"
                S3DataType             = "S3Prefix"
                S3InputMode            = "File"
                S3DataDistributionType = "FullyReplicated"
                S3CompressionType      = "None"
              }
            },
            {
              InputName  = "test-data"
              AppManaged = false
              S3Input = {
                S3Uri                  = { "Get" = "Parameters.ProcessedDataUrl" }
                LocalPath              = "/opt/ml/processing/test"
                S3DataType             = "S3Prefix"
                S3InputMode            = "File"
                S3DataDistributionType = "FullyReplicated"
                S3CompressionType      = "None"
              }
            },
            {
              InputName  = "code"
              AppManaged = false
              S3Input = {
                S3Uri                  = "s3://${var.pipeline_code_bucket_name}/evaluate.py"
                LocalPath              = "/opt/ml/processing/code"
                S3DataType             = "S3Prefix"
                S3InputMode            = "File"
                S3DataDistributionType = "FullyReplicated"
                S3CompressionType      = "None"
              }
            }
          ]
          ProcessingOutputConfig = {
            Outputs = [
              {
                OutputName = "evaluation"
                AppManaged = false
                S3Output = {
                  S3Uri        = "s3://${var.model_artifacts_bucket_name}/evaluation/"
                  LocalPath    = "/opt/ml/processing/evaluation"
                  S3UploadMode = "EndOfJob"
                }
              }
            ]
          }
        }
        DependsOn = ["ModelTraining"]
      },
      {
        Name = "RegisterModel"
        Type = "RegisterModel"
        Arguments = {
          ModelPackageGroupName = var.model_package_group_name
          ModelApprovalStatus   = { "Get" = "Parameters.ModelApprovalStatus" }
          InferenceSpecification = {
            Containers = [
              {
                Image        = "${data.aws_caller_identity.current.account_id}.dkr.ecr.${var.aws_region}.amazonaws.com/sagemaker-scikit-learn:0.23-1-cpu-py3"
                ModelDataUrl = "s3://${var.model_artifacts_bucket_name}/training-jobs/"
                Environment = {
                  "SAGEMAKER_PROGRAM"          = "inference.py"
                  "SAGEMAKER_SUBMIT_DIRECTORY" = "s3://${var.pipeline_code_bucket_name}/inference-code.tar.gz"
                }
              }
            ]
            SupportedContentTypes      = ["text/csv", "application/json"]
            SupportedResponseMIMETypes = ["application/json"]
          }
          ModelMetrics = {
            ModelQuality = {
              Statistics = {
                ContentType = "application/json"
                S3Uri       = "s3://${var.model_artifacts_bucket_name}/evaluation/evaluation.json"
              }
            }
          }
        }
        DependsOn = ["ModelEvaluation"]
      }
    ]
  })
}
