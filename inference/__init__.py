"""
Inference module for AegisMedBot - Hospital AI Intelligence Platform.

This module handles all model inference operations including:
- Model loading and management
- Batch and streaming predictions
- Performance optimizations
- Caching strategies
- Model quantization and acceleration

The inference engine is designed to be:
- Production-ready with robust error handling
- Optimized for low latency in hospital settings
- Scalable for concurrent requests
- Memory efficient for large language models
"""

from .engine import InferenceEngine, ModelType, InferenceConfig
from .cache import InferenceCache, CacheStrategy
from .optimizations import ModelOptimizer, QuantizationType, OptimizationConfig

__all__ = [
    'InferenceEngine',
    'ModelType', 
    'InferenceConfig',
    'InferenceCache',
    'CacheStrategy',
    'ModelOptimizer',
    'QuantizationType',
    'OptimizationConfig'
]

__version__ = '1.0.0'