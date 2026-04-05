"""
Core inference engine for AegisMedBot managing model loading, execution, and prediction pipelines.

This module provides the main inference interface for all AI models in the system,
including transformer-based language models, LSTM predictors, and specialized medical models.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import logging
import time
import asyncio
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import OrderedDict
import json

# Configure module logger
logger = logging.getLogger(__name__)

class ModelType(Enum):
    """
    Enumeration of supported model types in the inference engine.
    
    This allows the system to handle different model architectures with appropriate
    loading and inference strategies.
    """
    TRANSFORMER = "transformer"      # Transformer-based language models like BERT, GPT
    LSTM = "lstm"                     # LSTM models for time series prediction
    LINEAR = "linear"                 # Simple linear models for risk scoring
    ENSEMBLE = "ensemble"             # Ensemble of multiple models
    MEDICAL_TRANSFORMER = "medical_transformer"  # Domain-specific medical models

@dataclass
class InferenceConfig:
    """
    Configuration class for inference engine parameters.
    
    This dataclass centralizes all inference-related configuration options,
    making it easy to tune performance and memory usage.
    
    Attributes:
        device: Device to run inference on (cuda/cpu)
        batch_size: Maximum batch size for inference
        max_sequence_length: Maximum token sequence length for transformers
        use_fp16: Enable mixed precision inference
        use_cache: Enable result caching
        cache_ttl: Time to live for cached results in seconds
        timeout_seconds: Maximum inference time before timeout
        max_retries: Number of retry attempts on failure
        warmup_runs: Number of warmup runs to optimize performance
    """
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 32
    max_sequence_length: int = 512
    use_fp16: bool = True
    use_cache: bool = True
    cache_ttl: int = 3600  # 1 hour
    timeout_seconds: int = 30
    max_retries: int = 3
    warmup_runs: int = 5
    
    def __post_init__(self):
        """
        Validate configuration after initialization.
        
        Ensures that configuration values are within acceptable ranges
        and logs warnings for potentially problematic settings.
        """
        if self.batch_size < 1:
            raise ValueError(f"Batch size must be >= 1, got {self.batch_size}")
        
        if self.max_sequence_length < 1:
            raise ValueError(f"Max sequence length must be >= 1, got {self.max_sequence_length}")
        
        if self.timeout_seconds < 1:
            raise ValueError(f"Timeout must be >= 1 second, got {self.timeout_seconds}")
        
        if self.cache_ttl < 0:
            raise ValueError(f"Cache TTL must be >= 0, got {self.cache_ttl}")
        
        # Log configuration for debugging
        logger.info(f"Inference configuration initialized: device={self.device}, "
                   f"batch_size={self.batch_size}, use_fp16={self.use_fp16}")

class ModelRegistry:
    """
    Registry for managing multiple models in the inference system.
    
    This class implements a model registry pattern that allows the inference engine
    to load, store, and retrieve different models by name. It handles model lifecycle
    management including loading from disk and cleanup.
    
    The registry uses an OrderedDict to maintain model loading order and implements
    LRU (Least Recently Used) semantics for cache management.
    """
    
    def __init__(self, max_models: int = 10):
        """
        Initialize the model registry.
        
        Args:
            max_models: Maximum number of models to keep in memory simultaneously.
                       Models beyond this limit will be unloaded using LRU policy.
        """
        self._models = OrderedDict()  # Maintain insertion order for LRU
        self._max_models = max_models
        self._model_metadata = {}  # Store metadata about loaded models
        logger.info(f"Model registry initialized with max capacity: {max_models}")
    
    def register(self, name: str, model: nn.Module, metadata: Optional[Dict] = None):
        """
        Register a model in the registry.
        
        This method adds a model to the registry, implementing LRU eviction
        when the maximum capacity is reached.
        
        Args:
            name: Unique identifier for the model
            model: PyTorch model instance
            metadata: Optional metadata about the model (version, type, etc.)
        """
        if name in self._models:
            # Move existing model to end (most recently used)
            self._models.move_to_end(name)
            logger.info(f"Model '{name}' already exists, moved to end of LRU")
        else:
            # Check capacity and evict if needed
            if len(self._models) >= self._max_models:
                # Remove least recently used model
                oldest_name, oldest_model = self._models.popitem(last=False)
                logger.info(f"Evicted model '{oldest_name}' due to capacity limit")
                
                # Clear model from GPU memory
                if torch.cuda.is_available():
                    del oldest_model
                    torch.cuda.empty_cache()
        
        # Register the new model
        self._models[name] = model
        self._model_metadata[name] = metadata or {}
        logger.info(f"Model '{name}' registered successfully with metadata: {self._model_metadata[name]}")
    
    def get(self, name: str) -> Optional[nn.Module]:
        """
        Retrieve a model from the registry.
        
        This method implements LRU behavior by moving accessed models to the end
        of the ordered dictionary.
        
        Args:
            name: Model identifier
            
        Returns:
            Model instance if found, None otherwise
        """
        if name in self._models:
            # Move to end (most recently used)
            self._models.move_to_end(name)
            logger.debug(f"Retrieved model '{name}' from registry")
            return self._models[name]
        else:
            logger.warning(f"Model '{name}' not found in registry")
            return None
    
    def unregister(self, name: str) -> bool:
        """
        Remove a model from the registry and free memory.
        
        Args:
            name: Model identifier
            
        Returns:
            True if model was removed, False otherwise
        """
        if name in self._models:
            model = self._models.pop(name)
            self._model_metadata.pop(name, None)
            
            # Free GPU memory
            if torch.cuda.is_available():
                del model
                torch.cuda.empty_cache()
            
            logger.info(f"Model '{name}' unregistered and memory freed")
            return True
        
        logger.warning(f"Attempted to unregister non-existent model '{name}'")
        return False
    
    def get_metadata(self, name: str) -> Optional[Dict]:
        """
        Retrieve metadata for a registered model.
        
        Args:
            name: Model identifier
            
        Returns:
            Metadata dictionary if model exists, None otherwise
        """
        return self._model_metadata.get(name)
    
    def list_models(self) -> List[str]:
        """
        List all registered model names.
        
        Returns:
            List of model identifiers currently in registry
        """
        return list(self._models.keys())
    
    def clear(self):
        """
        Clear all models from registry and free all GPU memory.
        
        This method is useful during system shutdown or when resetting the inference engine.
        """
        for name in list(self._models.keys()):
            self.unregister(name)
        logger.info("Model registry cleared completely")

class InferenceEngine:
    """
    Core inference engine for AegisMedBot managing model inference operations.
    
    This class serves as the main interface for all AI model inference in the system.
    It handles model loading, batch processing, caching, performance optimization,
    and provides both synchronous and asynchronous inference methods.
    
    The engine implements:
    - Model registry for multi-model management
    - Automatic batching for efficiency
    - Result caching to avoid redundant computations
    - Mixed precision inference for speed
    - Comprehensive error handling and retries
    - Performance monitoring and metrics
    """
    
    def __init__(self, config: Optional[InferenceConfig] = None):
        """
        Initialize the inference engine with configuration.
        
        Args:
            config: Inference configuration. If None, defaults are used.
        """
        self.config = config or InferenceConfig()
        self.registry = ModelRegistry()
        self._performance_stats = {}  # Track inference times per model
        self._lock = asyncio.Lock()  # Lock for thread-safe operations
        
        # Set device and precision
        self.device = torch.device(self.config.device)
        self.dtype = torch.float16 if self.config.use_fp16 and self.device.type == 'cuda' else torch.float32
        
        logger.info(f"Inference engine initialized: device={self.device}, dtype={self.dtype}")
        
        # Warm up device if needed
        if self.config.warmup_runs > 0:
            self._warmup()
    
    def _warmup(self):
        """
        Perform warmup runs to optimize inference performance.
        
        Warmup helps:
        - Initialize CUDA kernels
        - Allocate memory pools
        - Optimize JIT compilation
        """
        logger.info(f"Performing {self.config.warmup_runs} warmup runs...")
        
        # Create dummy input for warmup
        dummy_input = torch.randn(1, 10).to(self.device)
        
        for i in range(self.config.warmup_runs):
            try:
                # Simple operation to warm up GPU
                _ = dummy_input @ dummy_input.T
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                logger.debug(f"Warmup run {i + 1}/{self.config.warmup_runs} completed")
            except Exception as e:
                logger.warning(f"Warmup run {i + 1} failed: {e}")
        
        # Clear any warmup artifacts
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        logger.info("Warmup completed")
    
    def load_model(
        self,
        model_name: str,
        model_path: Union[str, Path],
        model_type: ModelType,
        **kwargs
    ) -> bool:
        """
        Load a model from disk into the inference engine.
        
        This method handles loading different model types with appropriate
        configurations and adds them to the model registry.
        
        Args:
            model_name: Unique identifier for the model
            model_path: Path to saved model weights
            model_type: Type of model being loaded
            **kwargs: Additional model-specific arguments
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            logger.info(f"Loading model '{model_name}' from {model_path}")
            
            # Convert path to Path object for consistency
            model_path = Path(model_path)
            
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # Load model based on type
            if model_type == ModelType.TRANSFORMER:
                model = self._load_transformer_model(model_path, **kwargs)
            elif model_type == ModelType.LSTM:
                model = self._load_lstm_model(model_path, **kwargs)
            elif model_type == ModelType.LINEAR:
                model = self._load_linear_model(model_path, **kwargs)
            elif model_type == ModelType.ENSEMBLE:
                model = self._load_ensemble_model(model_path, **kwargs)
            elif model_type == ModelType.MEDICAL_TRANSFORMER:
                model = self._load_medical_transformer(model_path, **kwargs)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            # Move model to appropriate device and set precision
            model = model.to(self.device)
            
            if self.config.use_fp16 and self.device.type == 'cuda':
                model = model.half()  # Convert to half precision
            
            model.eval()  # Set to evaluation mode
            
            # Register model with metadata
            metadata = {
                'type': model_type.value,
                'path': str(model_path),
                'loaded_at': time.time(),
                'kwargs': kwargs
            }
            
            self.registry.register(model_name, model, metadata)
            
            logger.info(f"Model '{model_name}' loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model '{model_name}': {str(e)}", exc_info=True)
            return False
    
    def _load_transformer_model(self, path: Path, **kwargs) -> nn.Module:
        """
        Load transformer-based model.
        
        Args:
            path: Path to model weights
            **kwargs: Model configuration parameters
            
        Returns:
            Loaded transformer model
        """
        try:
            # Load checkpoint
            checkpoint = torch.load(path, map_location='cpu')
            
            # Extract model configuration
            config = checkpoint.get('config', {})
            model_class = kwargs.get('model_class', 'bert')  # Default to BERT-like
            
            # Placeholder for actual transformer loading
            # In production, this would use transformers library
            from transformers import AutoModelForSequenceClassification
            
            model = AutoModelForSequenceClassification.from_pretrained(
                str(path),
                num_labels=kwargs.get('num_labels', 2),
                **config
            )
            
            return model
            
        except Exception as e:
            logger.error(f"Error loading transformer model: {e}")
            raise
    
    def _load_lstm_model(self, path: Path, **kwargs) -> nn.Module:
        """
        Load LSTM model for time series prediction.
        
        Args:
            path: Path to model weights
            **kwargs: LSTM parameters (input_size, hidden_size, num_layers)
            
        Returns:
            Loaded LSTM model
        """
        try:
            checkpoint = torch.load(path, map_location='cpu')
            
            # Extract LSTM parameters
            input_size = kwargs.get('input_size', checkpoint.get('input_size', 10))
            hidden_size = kwargs.get('hidden_size', checkpoint.get('hidden_size', 128))
            num_layers = kwargs.get('num_layers', checkpoint.get('num_layers', 2))
            output_size = kwargs.get('output_size', checkpoint.get('output_size', 1))
            
            # Create LSTM model
            class LSTMPredictor(nn.Module):
                def __init__(self, input_size, hidden_size, num_layers, output_size):
                    super().__init__()
                    self.lstm = nn.LSTM(
                        input_size, hidden_size, num_layers,
                        batch_first=True, dropout=0.2
                    )
                    self.fc = nn.Linear(hidden_size, output_size)
                
                def forward(self, x):
                    lstm_out, _ = self.lstm(x)
                    last_output = lstm_out[:, -1, :]
                    return self.fc(last_output)
            
            model = LSTMPredictor(input_size, hidden_size, num_layers, output_size)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            return model
            
        except Exception as e:
            logger.error(f"Error loading LSTM model: {e}")
            raise
    
    def _load_linear_model(self, path: Path, **kwargs) -> nn.Module:
        """
        Load linear model for simple predictions.
        
        Args:
            path: Path to model weights
            **kwargs: Model parameters
            
        Returns:
            Loaded linear model
        """
        try:
            checkpoint = torch.load(path, map_location='cpu')
            input_dim = kwargs.get('input_dim', checkpoint.get('input_dim', 10))
            output_dim = kwargs.get('output_dim', checkpoint.get('output_dim', 1))
            
            model = nn.Linear(input_dim, output_dim)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            return model
            
        except Exception as e:
            logger.error(f"Error loading linear model: {e}")
            raise
    
    def _load_ensemble_model(self, path: Path, **kwargs) -> nn.Module:
        """
        Load ensemble of multiple models.
        
        Args:
            path: Path to ensemble configuration
            **kwargs: Ensemble parameters
            
        Returns:
            Ensemble model wrapper
        """
        try:
            with open(path / 'ensemble_config.json', 'r') as f:
                config = json.load(f)
            
            # Load individual models
            models = []
            for model_config in config['models']:
                model_path = path / model_config['path']
                model_type = ModelType(model_config['type'])
                model = self._load_model_by_type(model_path, model_type, **model_config.get('kwargs', {}))
                models.append(model)
            
            # Create ensemble wrapper
            class EnsembleModel(nn.Module):
                def __init__(self, models, weights=None):
                    super().__init__()
                    self.models = nn.ModuleList(models)
                    self.weights = weights or [1.0 / len(models)] * len(models)
                
                def forward(self, x):
                    outputs = [model(x) for model in self.models]
                    weighted_outputs = [w * out for w, out in zip(self.weights, outputs)]
                    return torch.stack(weighted_outputs).sum(dim=0)
            
            return EnsembleModel(models, config.get('weights'))
            
        except Exception as e:
            logger.error(f"Error loading ensemble model: {e}")
            raise
    
    def _load_medical_transformer(self, path: Path, **kwargs) -> nn.Module:
        """
        Load domain-specific medical transformer model.
        
        Args:
            path: Path to model weights
            **kwargs: Medical transformer parameters
            
        Returns:
            Loaded medical transformer model
        """
        try:
            # Use BiomedNLP or other medical-specific models
            from transformers import AutoModelForSequenceClassification
            
            model_name = kwargs.get('model_name', 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext')
            
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=kwargs.get('num_labels', 2),
                **kwargs
            )
            
            # Load fine-tuned weights if provided
            if path.exists() and not path.is_dir():
                state_dict = torch.load(path, map_location='cpu')
                model.load_state_dict(state_dict, strict=False)
            
            return model
            
        except Exception as e:
            logger.error(f"Error loading medical transformer: {e}")
            raise
    
    async def predict(
        self,
        model_name: str,
        inputs: Union[torch.Tensor, List[Any], np.ndarray],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run inference on a single input or batch.
        
        This is the main prediction method that handles:
        - Input validation and preprocessing
        - Model retrieval from registry
        - Inference with proper error handling
        - Result formatting and postprocessing
        - Performance metrics collection
        
        Args:
            model_name: Name of the model to use
            inputs: Input data for inference
            **kwargs: Additional inference parameters
            
        Returns:
            Dictionary containing predictions and metadata
        """
        start_time = time.time()
        
        # Acquire lock for thread safety
        async with self._lock:
            try:
                # Retrieve model from registry
                model = self.registry.get(model_name)
                if model is None:
                    raise ValueError(f"Model '{model_name}' not loaded")
                
                # Preprocess inputs to tensor format
                input_tensor = self._preprocess_input(inputs, **kwargs)
                
                # Move to appropriate device
                input_tensor = input_tensor.to(self.device)
                
                # Set precision
                if self.dtype == torch.float16:
                    input_tensor = input_tensor.half()
                
                # Perform inference with timeout
                with torch.no_grad():
                    # Use asyncio timeout to prevent hanging
                    try:
                        predictions = await asyncio.wait_for(
                            self._run_inference(model, input_tensor, **kwargs),
                            timeout=self.config.timeout_seconds
                        )
                    except asyncio.TimeoutError:
                        raise TimeoutError(f"Inference timed out after {self.config.timeout_seconds} seconds")
                
                # Postprocess results
                results = self._postprocess_output(predictions, **kwargs)
                
                # Calculate inference time
                inference_time = (time.time() - start_time) * 1000  # Convert to milliseconds
                
                # Update performance statistics
                self._update_stats(model_name, inference_time)
                
                # Prepare response
                response = {
                    'success': True,
                    'predictions': results,
                    'model': model_name,
                    'inference_time_ms': inference_time,
                    'device': str(self.device),
                    'precision': str(self.dtype)
                }
                
                # Add confidence if available
                if hasattr(results, 'confidence'):
                    response['confidence'] = results.confidence
                
                logger.info(f"Inference completed for model '{model_name}' in {inference_time:.2f}ms")
                return response
                
            except Exception as e:
                logger.error(f"Inference failed for model '{model_name}': {str(e)}", exc_info=True)
                return {
                    'success': False,
                    'error': str(e),
                    'model': model_name,
                    'inference_time_ms': (time.time() - start_time) * 1000
                }
    
    async def predict_batch(
        self,
        model_name: str,
        batch_inputs: List[Union[torch.Tensor, List[Any], np.ndarray]],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Run inference on a batch of inputs efficiently.
        
        This method implements batching for better throughput by grouping
        inputs into optimal batch sizes for GPU utilization.
        
        Args:
            model_name: Name of the model to use
            batch_inputs: List of input data points
            **kwargs: Additional inference parameters
            
        Returns:
            List of prediction dictionaries for each input
        """
        if not batch_inputs:
            return []
        
        # Process in optimal batch sizes
        batch_size = kwargs.get('batch_size', self.config.batch_size)
        results = []
        
        for i in range(0, len(batch_inputs), batch_size):
            batch = batch_inputs[i:i + batch_size]
            
            # Process batch
            batch_result = await self.predict(model_name, batch, **kwargs)
            
            # Extract individual predictions from batch result
            if batch_result['success']:
                predictions = batch_result['predictions']
                
                # Handle batch predictions
                if isinstance(predictions, torch.Tensor) and predictions.dim() > 1:
                    for j in range(len(batch)):
                        results.append({
                            'success': True,
                            'prediction': predictions[j].cpu().numpy(),
                            'confidence': batch_result.get('confidence', [None])[j] if isinstance(batch_result.get('confidence'), list) else None
                        })
                elif isinstance(predictions, list):
                    results.extend(predictions)
                else:
                    # Single prediction for entire batch
                    results.append(batch_result)
            else:
                # Add failure for all items in batch
                results.extend([batch_result] * len(batch))
        
        return results
    
    async def _run_inference(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Execute model inference with optional retries.
        
        Args:
            model: PyTorch model
            inputs: Input tensor
            **kwargs: Additional arguments for model forward pass
            
        Returns:
            Model output tensor
        """
        retries = kwargs.get('retries', self.config.max_retries)
        
        for attempt in range(retries):
            try:
                # Run forward pass
                outputs = model(inputs, **kwargs)
                
                # Handle different output types
                if isinstance(outputs, tuple):
                    # Some models return (logits, hidden_states)
                    outputs = outputs[0]
                
                return outputs
                
            except RuntimeError as e:
                if 'out of memory' in str(e) and self.device.type == 'cuda':
                    # Clear CUDA cache on OOM
                    torch.cuda.empty_cache()
                    
                    if attempt < retries - 1:
                        logger.warning(f"CUDA OOM on attempt {attempt + 1}, retrying...")
                        await asyncio.sleep(0.1)  # Brief pause before retry
                        continue
                
                raise e
            except Exception as e:
                if attempt < retries - 1:
                    logger.warning(f"Inference failed on attempt {attempt + 1}: {e}, retrying...")
                    await asyncio.sleep(0.05)
                    continue
                raise e
    
    def _preprocess_input(
        self,
        inputs: Union[torch.Tensor, List[Any], np.ndarray],
        **kwargs
    ) -> torch.Tensor:
        """
        Convert and normalize input data to tensor format.
        
        Args:
            inputs: Input data in various formats
            **kwargs: Preprocessing parameters
            
        Returns:
            Preprocessed torch tensor
        """
        # Handle different input types
        if isinstance(inputs, torch.Tensor):
            tensor = inputs
        elif isinstance(inputs, np.ndarray):
            tensor = torch.from_numpy(inputs)
        elif isinstance(inputs, list):
            # Convert list to tensor
            if all(isinstance(x, (int, float)) for x in inputs):
                # List of numbers
                tensor = torch.tensor(inputs, dtype=torch.float32)
            else:
                # List of sequences - need to pad
                tensor = self._pad_sequences(inputs, **kwargs)
        else:
            raise TypeError(f"Unsupported input type: {type(inputs)}")
        
        # Ensure correct dimensions
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)  # Add batch dimension
        
        # Normalize if requested
        if kwargs.get('normalize', False):
            tensor = self._normalize_tensor(tensor, **kwargs)
        
        return tensor
    
    def _pad_sequences(
        self,
        sequences: List[List[int]],
        max_length: Optional[int] = None,
        padding_value: int = 0
    ) -> torch.Tensor:
        """
        Pad variable-length sequences to same length.
        
        Args:
            sequences: List of token sequences
            max_length: Maximum sequence length (uses max if None)
            padding_value: Value to use for padding
            
        Returns:
            Padded tensor of shape [batch_size, max_length]
        """
        if not sequences:
            return torch.tensor([])
        
        # Determine maximum length
        if max_length is None:
            max_length = max(len(seq) for seq in sequences)
        else:
            max_length = min(max_length, max(len(seq) for seq in sequences))
        
        # Truncate and pad
        padded_sequences = []
        for seq in sequences:
            # Truncate if needed
            if len(seq) > max_length:
                seq = seq[:max_length]
            
            # Pad to max_length
            padding_needed = max_length - len(seq)
            padded_seq = seq + [padding_value] * padding_needed
            padded_sequences.append(padded_seq)
        
        return torch.tensor(padded_sequences, dtype=torch.long)
    
    def _normalize_tensor(
        self,
        tensor: torch.Tensor,
        mean: Optional[float] = None,
        std: Optional[float] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Normalize tensor to zero mean and unit variance.
        
        Args:
            tensor: Input tensor
            mean: Mean for normalization (compute if None)
            std: Standard deviation for normalization (compute if None)
            
        Returns:
            Normalized tensor
        """
        if mean is None:
            mean = tensor.mean().item()
        if std is None:
            std = tensor.std().item() + 1e-8  # Avoid division by zero
        
        return (tensor - mean) / std
    
    def _postprocess_output(
        self,
        outputs: torch.Tensor,
        **kwargs
    ) -> Union[torch.Tensor, np.ndarray, Dict[str, Any]]:
        """
        Convert model outputs to desired format.
        
        Args:
            outputs: Raw model output tensor
            **kwargs: Postprocessing parameters
            
        Returns:
            Processed outputs in requested format
        """
        # Apply softmax for classification tasks
        if kwargs.get('apply_softmax', False):
            outputs = torch.softmax(outputs, dim=-1)
        
        # Convert to numpy if requested
        if kwargs.get('return_numpy', True):
            outputs = outputs.cpu().numpy()
        
        # Extract single value if batch size is 1
        if kwargs.get('squeeze', True) and outputs.shape[0] == 1:
            if isinstance(outputs, np.ndarray):
                outputs = outputs[0]
            else:
                outputs = outputs.squeeze(0)
        
        # Apply threshold for binary classification
        if kwargs.get('threshold', None) is not None:
            if isinstance(outputs, np.ndarray):
                outputs = (outputs > kwargs['threshold']).astype(int)
            else:
                outputs = (outputs > kwargs['threshold']).int()
        
        return outputs
    
    def _update_stats(self, model_name: str, inference_time_ms: float):
        """
        Update performance statistics for a model.
        
        Args:
            model_name: Model identifier
            inference_time_ms: Inference time in milliseconds
        """
        if model_name not in self._performance_stats:
            self._performance_stats[model_name] = {
                'total_inferences': 0,
                'total_time_ms': 0,
                'min_time_ms': float('inf'),
                'max_time_ms': 0,
                'times': []  # Recent times for moving average
            }
        
        stats = self._performance_stats[model_name]
        stats['total_inferences'] += 1
        stats['total_time_ms'] += inference_time_ms
        stats['min_time_ms'] = min(stats['min_time_ms'], inference_time_ms)
        stats['max_time_ms'] = max(stats['max_time_ms'], inference_time_ms)
        stats['times'].append(inference_time_ms)
        
        # Keep only last 100 times for moving average
        if len(stats['times']) > 100:
            stats['times'].pop(0)
    
    def get_performance_stats(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Retrieve performance statistics for models.
        
        Args:
            model_name: Optional specific model name
            
        Returns:
            Performance statistics dictionary
        """
        if model_name:
            stats = self._performance_stats.get(model_name, {})
            if stats:
                # Calculate moving average
                avg_time = sum(stats['times']) / len(stats['times']) if stats['times'] else 0
                return {
                    'model_name': model_name,
                    'total_inferences': stats['total_inferences'],
                    'average_time_ms': stats['total_time_ms'] / stats['total_inferences'] if stats['total_inferences'] > 0 else 0,
                    'moving_avg_ms': avg_time,
                    'min_time_ms': stats['min_time_ms'] if stats['min_time_ms'] != float('inf') else 0,
                    'max_time_ms': stats['max_time_ms']
                }
            return {'model_name': model_name, 'total_inferences': 0}
        
        # Return stats for all models
        return {
            name: {
                'total_inferences': stats['total_inferences'],
                'average_time_ms': stats['total_time_ms'] / stats['total_inferences'] if stats['total_inferences'] > 0 else 0,
                'min_time_ms': stats['min_time_ms'] if stats['min_time_ms'] != float('inf') else 0,
                'max_time_ms': stats['max_time_ms']
            }
            for name, stats in self._performance_stats.items()
        }
    
    def clear_model_cache(self, model_name: Optional[str] = None):
        """
        Clear cached models from registry.
        
        Args:
            model_name: Optional specific model to clear
        """
        if model_name:
            self.registry.unregister(model_name)
            if model_name in self._performance_stats:
                del self._performance_stats[model_name]
            logger.info(f"Cleared model cache for '{model_name}'")
        else:
            self.registry.clear()
            self._performance_stats.clear()
            logger.info("Cleared all model caches")
        
        # Clear GPU cache
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a loaded model.
        
        Args:
            model_name: Model identifier
            
        Returns:
            Model information dictionary
        """
        model = self.registry.get(model_name)
        if model is None:
            return {'error': f'Model {model_name} not found'}
        
        metadata = self.registry.get_metadata(model_name)
        stats = self.get_performance_stats(model_name)
        
        return {
            'name': model_name,
            'type': metadata.get('type') if metadata else 'unknown',
            'device': str(self.device),
            'parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'metadata': metadata,
            'performance': stats
        }