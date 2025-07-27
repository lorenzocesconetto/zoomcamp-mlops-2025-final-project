# Makefile for Crypto Price Prediction Infrastructure

.PHONY: help init plan apply destroy validate format check clean

# Default target
help:
	@echo "Available targets:"
	@echo "  init      - Initialize Terraform"
	@echo "  validate  - Validate Terraform configuration"
	@echo "  format    - Format Terraform files"
	@echo "  plan      - Create Terraform execution plan"
	@echo "  apply     - Apply Terraform changes"
	@echo "  destroy   - Destroy Terraform-managed infrastructure"
	@echo "  check     - Run validation and formatting checks"
	@echo "  clean     - Clean up temporary files"
	@echo ""
	@echo "Variables:"
	@echo "  ENV=dev|staging|prod  - Environment to deploy to (default: dev)"
	@echo "  REGION=us-east-1      - AWS region (default: us-east-1)"
	@echo ""
	@echo "Examples:"
	@echo "  make plan ENV=prod"
	@echo "  make apply ENV=staging REGION=us-west-2"

# Variables
ENV ?= dev
REGION ?= us-east-1
TF_VAR_FILE = terraform.tfvars

# Check if terraform.tfvars exists
check-tfvars:
	@if [ ! -f $(TF_VAR_FILE) ]; then \
		echo "Error: $(TF_VAR_FILE) not found!"; \
		echo "Please copy terraform.tfvars.example to terraform.tfvars and configure it."; \
		exit 1; \
	fi

# Initialize Terraform
init:
	@echo "Initializing Terraform..."
	terraform init

# Validate Terraform configuration
validate: init
	@echo "Validating Terraform configuration..."
	terraform validate

# Format Terraform files
format:
	@echo "Formatting Terraform files..."
	terraform fmt -recursive

# Create execution plan
plan: check-tfvars validate
	@echo "Creating Terraform execution plan for environment: $(ENV)"
	terraform plan \
		-var-file=$(TF_VAR_FILE) \
		-var="environment=$(ENV)" \
		-var="aws_region=$(REGION)" \
		-out=tfplan

# Apply changes
apply: check-tfvars validate
	@echo "Applying Terraform changes for environment: $(ENV)"
	@read -p "Are you sure you want to apply changes to $(ENV)? [y/N] " confirm && \
	if [ "$$confirm" = "y" ] || [ "$$confirm" = "Y" ]; then \
		terraform apply \
			-var-file=$(TF_VAR_FILE) \
			-var="environment=$(ENV)" \
			-var="aws_region=$(REGION)"; \
	else \
		echo "Aborted."; \
	fi

# Apply with auto-approval (use with caution)
apply-auto: check-tfvars validate
	@echo "Auto-applying Terraform changes for environment: $(ENV)"
	terraform apply \
		-var-file=$(TF_VAR_FILE) \
		-var="environment=$(ENV)" \
		-var="aws_region=$(REGION)" \
		-auto-approve

# Destroy infrastructure
destroy: check-tfvars
	@echo "Destroying Terraform infrastructure for environment: $(ENV)"
	@read -p "Are you sure you want to DESTROY all resources in $(ENV)? [y/N] " confirm && \
	if [ "$$confirm" = "y" ] || [ "$$confirm" = "Y" ]; then \
		terraform destroy \
			-var-file=$(TF_VAR_FILE) \
			-var="environment=$(ENV)" \
			-var="aws_region=$(REGION)"; \
	else \
		echo "Aborted."; \
	fi

# Run all checks
check: format validate
	@echo "All checks passed!"

# Show current state
show:
	@echo "Current Terraform state:"
	terraform show

# List resources
list:
	@echo "Terraform managed resources:"
	terraform state list

# Get outputs
output:
	@echo "Terraform outputs:"
	terraform output

# Refresh state
refresh: check-tfvars
	@echo "Refreshing Terraform state..."
	terraform refresh \
		-var-file=$(TF_VAR_FILE) \
		-var="environment=$(ENV)" \
		-var="aws_region=$(REGION)"

# Import existing resource (example)
import:
	@echo "Import existing AWS resource:"
	@echo "Usage: terraform import <resource_type.resource_name> <aws_resource_id>"
	@echo "Example: terraform import aws_s3_bucket.existing_bucket my-existing-bucket"

# Clean up temporary files
clean:
	@echo "Cleaning up temporary files..."
	rm -f tfplan
	rm -f terraform.tfstate.backup
	rm -f .terraform.lock.hcl
	rm -rf .terraform/

# Setup development environment
setup-dev:
	@echo "Setting up development environment..."
	@if [ ! -f terraform.tfvars ]; then \
		cp terraform.tfvars.example terraform.tfvars; \
		echo "Created terraform.tfvars from example. Please edit it with your values."; \
	fi
	$(MAKE) init
	$(MAKE) validate
	@echo "Development environment ready!"

# Quick deployment for development
dev-deploy: setup-dev
	$(MAKE) apply ENV=dev

# Production deployment with extra safety
prod-deploy:
	@echo "Production deployment requires manual confirmation at each step."
	$(MAKE) plan ENV=prod
	@read -p "Review the plan above. Continue with apply? [y/N] " confirm && \
	if [ "$$confirm" = "y" ] || [ "$$confirm" = "Y" ]; then \
		$(MAKE) apply ENV=prod; \
	else \
		echo "Production deployment cancelled."; \
	fi

# Upgrade provider versions
upgrade:
	@echo "Upgrading provider versions..."
	terraform init -upgrade

# Security scan (requires tfsec)
security-scan:
	@if command -v tfsec >/dev/null 2>&1; then \
		echo "Running security scan with tfsec..."; \
		tfsec .; \
	else \
		echo "tfsec not found. Install it with: brew install tfsec"; \
	fi

# Cost estimation (requires infracost)
cost-estimate:
	@if command -v infracost >/dev/null 2>&1; then \
		echo "Estimating infrastructure costs..."; \
		infracost breakdown --path .; \
	else \
		echo "infracost not found. Install it from: https://www.infracost.io/"; \
	fi

# Create new environment configuration
create-env:
	@read -p "Enter environment name (dev/staging/prod): " env_name && \
	if [ -f "terraform.tfvars.$$env_name" ]; then \
		echo "Configuration for $$env_name already exists!"; \
	else \
		cp terraform.tfvars.example terraform.tfvars.$$env_name; \
		echo "Created terraform.tfvars.$$env_name. Please edit it with environment-specific values."; \
	fi
