#!/usr/bin/env python3
"""
SageMaker Model Evaluation Job Entry Point for Crypto Price Prediction
This script serves as the entry point for SageMaker evaluation jobs.
"""

from app.domain.services.evaluation_service import main

if __name__ == "__main__":
    main()
