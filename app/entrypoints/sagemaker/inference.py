#!/usr/bin/env python3
"""
SageMaker Inference Script Entry Point for Crypto Price Prediction
This script serves as the entry point for SageMaker inference endpoints.
"""

from app.domain.services.inference_service import (input_fn, model_fn,
                                                   output_fn, predict_fn)

# Export the SageMaker handler functions
__all__ = ["model_fn", "input_fn", "predict_fn", "output_fn"]
