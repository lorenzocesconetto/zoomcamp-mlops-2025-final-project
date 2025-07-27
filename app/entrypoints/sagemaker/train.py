#!/usr/bin/env python3
"""
SageMaker Training Job Entry Point for Crypto Price Prediction
This script serves as the entry point for SageMaker training jobs.
"""

from app.domain.services.training_service import main

if __name__ == "__main__":
    main()
