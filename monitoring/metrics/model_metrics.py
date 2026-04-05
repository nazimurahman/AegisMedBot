"""
Model Performance Metrics Collection Module
This module collects and exposes metrics for ML model performance monitoring.
Uses Prometheus client library to create and update metrics that can be scraped.
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional
from prometheus_client import Counter, Histogram, Gauge, Summary, Info
import time
from datetime import datetime
import logging
import asyncio
from dataclasses import dataclass
from collections import defaultdict

# Configure logging
logger = logging.getLogger(__name__)

# Define Prometheus metrics for model performance
class ModelMetricsCollector:
    """
    Collects and exposes metrics for machine learning models.
    Tracks inference latency, accuracy, memory usage, and model version information.
    """
    
    def __init__(self):
        """
        Initialize all Prometheus metrics for model monitoring.
        Each metric type serves a specific monitoring purpose.
        """
        
        # Counter: Tracks cumulative counts that only increase
        # Counts total inference requests across all models
        self.inference_requests = Counter(
            'model_inference_requests_total',
            'Total number of model inference requests',
            ['model_name', 'model_version', 'environment']
        )
        
        # Counter: Tracks successful predictions
        self.successful_predictions = Counter(
            'model_successful_predictions_total',
            'Total number of successful predictions',
            ['model_name', 'model_version']
        )
        
        # Counter: Tracks failed inferences with error type classification
        self.inference_errors = Counter(
            'model_inference_errors_total',
            'Total number of inference errors',
            ['model_name', 'model_version', 'error_type']
        )
        
        # Histogram: Measures distribution of inference duration
        # Buckets from milliseconds to seconds for detailed latency analysis
        self.inference_duration = Histogram(
            'model_inference_duration_seconds',
            'Time taken for model inference',
            ['model_name', 'model_version'],
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30)
        )
        
        # Histogram: Tracks confidence score distribution
        # Helps identify if models are becoming uncertain
        self.confidence_scores = Histogram(
            'model_confidence_scores',
            'Model confidence scores for predictions',
            ['model_name', 'model_version'],
            buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
        )
        
        # Gauge: Current values that can go up and down
        # Memory usage in bytes - critical for resource management
        self.model_memory_usage = Gauge(
            'model_memory_usage_bytes',
            'Current memory usage of model in bytes',
            ['model_name', 'model_version']
        )
        
        # Gauge: GPU utilization percentage if using CUDA
        self.gpu_utilization = Gauge(
            'model_gpu_utilization_percent',
            'GPU utilization percentage during inference',
            ['model_name', 'gpu_id']
        )
        
        # Gauge: Model accuracy - updated periodically from evaluation
        self.model_accuracy = Gauge(
            'model_accuracy_score',
            'Model accuracy score from evaluation',
            ['model_name', 'model_version']
        )
        
        # Summary: Quantiles for performance monitoring
        self.batch_processing_time = Summary(
            'model_batch_processing_seconds',
            'Time to process a batch of inputs',
            ['model_name', 'model_version']
        )
        
        # Gauge: Active model instances - helps with scaling decisions
        self.active_models = Gauge(
            'model_active_instances',
            'Number of active model instances',
            ['model_name']
        )
        
        # Info: Static information about models
        self.model_info = Info(
            'model_information',
            'Static information about deployed models',
            ['model_name']
        )
        
        # Counter: Tracks token usage for LLM models
        self.token_usage = Counter(
            'model_token_usage_total',
            'Total number of tokens processed by model',
            ['model_name', 'model_version', 'token_type']  # token_type: input, output
        )
        
        # Gauge: Model temperature for load balancing
        self.model_temperature = Gauge(
            'model_temperature',
            'Current load/temperature of model instance',
            ['model_name', 'instance_id']
        )
        
        # Store active model instances for cleanup
        self.active_instances = defaultdict(list)
        
        logger.info("Model metrics collector initialized successfully")
    
    def record_inference(
        self,
        model_name: str,
        model_version: str,
        start_time: float,
        success: bool = True,
        confidence: Optional[float] = None,
        error_type: Optional[str] = None,
        environment: str = "production",
        input_tokens: int = 0,
        output_tokens: int = 0
    ) -> None:
        """
        Record a single inference event with all relevant metrics.
        
        Args:
            model_name: Name of the model being used
            model_version: Version identifier for the model
            start_time: Timestamp when inference started
            success: Whether inference completed successfully
            confidence: Model confidence score (0-1) if applicable
            error_type: Type of error if inference failed
            environment: Deployment environment
            input_tokens: Number of input tokens processed (for LLMs)
            output_tokens: Number of output tokens generated (for LLMs)
        """
        
        # Calculate inference duration
        duration = time.time() - start_time
        
        # Record inference duration in histogram
        self.inference_duration.labels(
            model_name=model_name,
            model_version=model_version
        ).observe(duration)
        
        # Record inference request count
        self.inference_requests.labels(
            model_name=model_name,
            model_version=model_version,
            environment=environment
        ).inc()
        
        # Handle success or failure cases
        if success:
            # Increment success counter
            self.successful_predictions.labels(
                model_name=model_name,
                model_version=model_version
            ).inc()
            
            # Record confidence score if provided
            if confidence is not None:
                self.confidence_scores.labels(
                    model_name=model_name,
                    model_version=model_version
                ).observe(confidence)
        else:
            # Record error with classification
            error_type = error_type or "unknown_error"
            self.inference_errors.labels(
                model_name=model_name,
                model_version=model_version,
                error_type=error_type
            ).inc()
        
        # Record token usage for LLM models
        if input_tokens > 0:
            self.token_usage.labels(
                model_name=model_name,
                model_version=model_version,
                token_type="input"
            ).inc(input_tokens)
        
        if output_tokens > 0:
            self.token_usage.labels(
                model_name=model_name,
                model_version=model_version,
                token_type="output"
            ).inc(output_tokens)
    
    def update_memory_usage(
        self,
        model_name: str,
        model_version: str,
        memory_bytes: int
    ) -> None:
        """
        Update current memory usage for a model.
        Critical for capacity planning and detecting memory leaks.
        
        Args:
            model_name: Name of the model
            model_version: Version of the model
            memory_bytes: Current memory usage in bytes
        """
        self.model_memory_usage.labels(
            model_name=model_name,
            model_version=model_version
        ).set(memory_bytes)
    
    def update_gpu_metrics(
        self,
        model_name: str,
        gpu_id: int,
        utilization_percent: float,
        memory_used_bytes: Optional[int] = None
    ) -> None:
        """
        Update GPU utilization metrics for models running on CUDA.
        
        Args:
            model_name: Name of the model using GPU
            gpu_id: Identifier for the GPU device
            utilization_percent: GPU utilization percentage
            memory_used_bytes: GPU memory used in bytes (optional)
        """
        # Update GPU utilization gauge
        self.gpu_utilization.labels(
            model_name=model_name,
            gpu_id=str(gpu_id)
        ).set(utilization_percent)
        
        # If memory usage provided, update memory gauge
        if memory_used_bytes is not None:
            self.model_memory_usage.labels(
                model_name=model_name,
                model_version="gpu"
            ).set(memory_used_bytes)
    
    def update_accuracy(
        self,
        model_name: str,
        model_version: str,
        accuracy: float
    ) -> None:
        """
        Update model accuracy from evaluation results.
        Helps track model degradation over time.
        
        Args:
            model_name: Name of the model
            model_version: Version of the model
            accuracy: Accuracy score (0-100 or 0-1)
        """
        # Convert to percentage if needed
        if accuracy <= 1.0:
            accuracy = accuracy * 100
            
        self.model_accuracy.labels(
            model_name=model_name,
            model_version=model_version
        ).set(accuracy)
    
    def register_model(
        self,
        model_name: str,
        model_version: str,
        model_type: str,
        framework: str,
        parameters: int,
        description: str = ""
    ) -> None:
        """
        Register a new model with static information.
        Called when model is loaded into memory.
        
        Args:
            model_name: Name of the model
            model_version: Version identifier
            model_type: Type of model (transformer, lstm, etc.)
            framework: Framework used (pytorch, tensorflow, etc.)
            parameters: Number of trainable parameters
            description: Optional description of model purpose
        """
        # Set static model information
        self.model_info.labels(model_name=model_name).info({
            'version': model_version,
            'type': model_type,
            'framework': framework,
            'parameters': str(parameters),
            'description': description,
            'registered_at': datetime.now().isoformat()
        })
        
        # Increment active model instance count
        self.active_models.labels(model_name=model_name).inc()
        
        logger.info(f"Model registered: {model_name} v{model_version}")
    
    def unregister_model(self, model_name: str, model_version: str) -> None:
        """
        Unregister a model when it's unloaded.
        Helps track active model instances.
        
        Args:
            model_name: Name of the model
            model_version: Version being unloaded
        """
        # Decrement active model instance count
        self.active_models.labels(model_name=model_name).dec()
        
        # Clear memory usage gauge
        self.model_memory_usage.labels(
            model_name=model_name,
            model_version=model_version
        ).set(0)
        
        logger.info(f"Model unregistered: {model_name} v{model_version}")
    
    def record_batch_processing(
        self,
        model_name: str,
        model_version: str,
        start_time: float,
        batch_size: int
    ) -> None:
        """
        Record batch processing time for models that support batching.
        
        Args:
            model_name: Name of the model
            model_version: Model version
            start_time: Start time of batch processing
            batch_size: Number of items in batch
        """
        duration = time.time() - start_time
        self.batch_processing_time.labels(
            model_name=model_name,
            model_version=model_version
        ).observe(duration)
        
        # Also record per-item time for efficiency metrics
        per_item_time = duration / batch_size
        self.inference_duration.labels(
            model_name=model_name,
            model_version=model_version
        ).observe(per_item_time)
    
    def update_model_temperature(
        self,
        model_name: str,
        instance_id: str,
        queue_length: int,
        max_queue: int = 100
    ) -> None:
        """
        Update model temperature gauge for load balancing.
        Temperature indicates how busy a model instance is.
        
        Args:
            model_name: Name of the model
            instance_id: Unique identifier for model instance
            queue_length: Current request queue length
            max_queue: Maximum queue capacity
        """
        # Calculate temperature as percentage of queue capacity
        temperature = (queue_length / max_queue) * 100
        
        self.model_temperature.labels(
            model_name=model_name,
            instance_id=instance_id
        ).set(temperature)
    
    def get_cuda_metrics(self) -> Dict[str, Any]:
        """
        Collect comprehensive CUDA metrics if GPU is available.
        Returns dictionary with GPU memory usage, utilization, and temperature.
        """
        metrics = {}
        
        if torch.cuda.is_available():
            try:
                for gpu_id in range(torch.cuda.device_count()):
                    # Get GPU properties
                    device_props = torch.cuda.get_device_properties(gpu_id)
                    
                    # Get memory stats
                    memory_allocated = torch.cuda.memory_allocated(gpu_id)
                    memory_cached = torch.cuda.memory_reserved(gpu_id)
                    memory_free = memory_cached - memory_allocated
                    
                    # Get utilization (requires nvidia-smi or pynvml)
                    # Simplified version - actual implementation would use nvidia-ml-py
                    metrics[f'gpu_{gpu_id}'] = {
                        'name': device_props.name,
                        'memory_total_bytes': device_props.total_memory,
                        'memory_allocated_bytes': memory_allocated,
                        'memory_cached_bytes': memory_cached,
                        'memory_free_bytes': memory_free,
                        'utilization_percent': None  # Would fetch from nvidia-smi
                    }
            except Exception as e:
                logger.error(f"Error collecting CUDA metrics: {e}")
        
        return metrics
    
    async def collect_model_metrics_periodically(
        self,
        interval_seconds: int = 60
    ) -> None:
        """
        Background task to collect and update model metrics periodically.
        Runs as an async coroutine for continuous monitoring.
        
        Args:
            interval_seconds: How often to collect metrics
        """
        while True:
            try:
                # Collect CUDA metrics if available
                cuda_metrics = self.get_cuda_metrics()
                
                # Update GPU metrics for each GPU
                for gpu_id, gpu_data in cuda_metrics.items():
                    if gpu_data['utilization_percent'] is not None:
                        self.gpu_utilization.labels(
                            model_name="global",
                            gpu_id=gpu_id
                        ).set(gpu_data['utilization_percent'])
                
                # Additional periodic metrics collection
                # Could include CPU temperature, disk usage, etc.
                
                logger.debug(f"Periodic metrics collection completed at {datetime.now()}")
                
            except Exception as e:
                logger.error(f"Error in periodic metrics collection: {e}")
            
            # Wait for next collection interval
            await asyncio.sleep(interval_seconds)

# Create global instance for use across application
model_metrics = ModelMetricsCollector()

# Context manager for timing model inference
class InferenceTimer:
    """
    Context manager for timing model inference operations.
    Automatically records metrics when used with 'with' statement.
    
    Usage:
        with InferenceTimer('bert_model', 'v1') as timer:
            result = model.predict(input_data)
            timer.set_success(True, confidence=0.95)
    """
    
    def __init__(
        self,
        model_name: str,
        model_version: str,
        environment: str = "production"
    ):
        """
        Initialize timer with model information.
        
        Args:
            model_name: Name of the model
            model_version: Version of the model
            environment: Deployment environment
        """
        self.model_name = model_name
        self.model_version = model_version
        self.environment = environment
        self.start_time = None
        self.success = False
        self.confidence = None
        self.error_type = None
        self.input_tokens = 0
        self.output_tokens = 0
    
    def __enter__(self):
        """Start timing when entering context."""
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Record metrics when exiting context.
        Handles both success and exception cases.
        """
        if exc_type is not None:
            # Exception occurred during inference
            self.success = False
            self.error_type = exc_type.__name__
        else:
            self.success = True
        
        # Record metrics
        model_metrics.record_inference(
            model_name=self.model_name,
            model_version=self.model_version,
            start_time=self.start_time,
            success=self.success,
            confidence=self.confidence,
            error_type=self.error_type,
            environment=self.environment,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens
        )
    
    def set_success(
        self,
        confidence: Optional[float] = None,
        input_tokens: int = 0,
        output_tokens: int = 0
    ) -> None:
        """
        Mark inference as successful with additional metadata.
        
        Args:
            confidence: Model confidence score
            input_tokens: Number of input tokens processed
            output_tokens: Number of output tokens generated
        """
        self.success = True
        self.confidence = confidence
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens