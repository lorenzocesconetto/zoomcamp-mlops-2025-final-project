#!/usr/bin/env python3
"""
SageMaker Processing Job Entry Point for Crypto Data Preprocessing
This script serves as the entry point for SageMaker processing jobs.
"""

from app.domain.services.preprocessing_service import main

if __name__ == "__main__":
    main()
