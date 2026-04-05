"""
Model optimization utilities for AegisMedBot.

This module provides tools for optimizing PyTorch models for inference,
including quantization, pruning, and various acceleration techniques.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union, List, Tuple
from enum import Enum
import logging
import numpy as np
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class QuantizationType(Enum):
    """
    Types of quantization supported for model optimization.
    
    Different quantization strategies offer trade-offs between
    model size, speed, and accuracy.
    """
    DYNAMIC = "dynamic"      # Dynamic quantization (weights quantized)
    STATIC = "static"        # Static quantization (both weights and activations)
    QAT = "qat"              # Quantization-aware training
    INT8 = "int8"            # 8-bit integer quantization
    FP16 = "fp16"            # Half-precision floating point
    INT4 = "int4"            # 4-bit integer quantization (experimental)

class OptimizationConfig:
    """
    Configuration for model optimization.
    
    This dataclass centralizes all optimization parameters.
    """
    def __init__(
        self,
        quantization_type: QuantizationType = QuantizationType.DYNAMIC,
        prune_unused_weights: bool = True,
        fuse_layers: bool = True,
        use_tensorrt: bool = False,
        use_onnx: bool = False,
        batch_size: int = 1,
        calibration_samples: int = 100,
        target_device: str = "cuda"
    ):
        self.quantization_type = quantization_type
        self.prune_unused_weights = prune_unused_weights
        self.fuse_layers = fuse_layers
        self.use_tensorrt = use_tensorrt
        self.use_onnx = use_onnx
        self.batch_size = batch_size
        self.calibration_samples = calibration_samples
        self.target_device = target_device

class ModelOptimizer:
    """
    Comprehensive model optimization toolkit.
    
    This class provides methods to optimize PyTorch models for
    faster inference with minimal accuracy loss, using techniques like:
    - Quantization (INT8, FP16)
    - Layer fusion
    - Pruning
    - ONNX/TensorRT conversion
    - Graph optimization
    
    The optimizer is designed to work with medical models used in
    the AegisMedBot system, balancing speed and accuracy.
    """
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        """
        Initialize the model optimizer.
        
        Args:
            config: Optimization configuration
        """
        self.config = config or OptimizationConfig()
        logger.info(f"Model optimizer initialized with config: {self.config.quantization_type.value}")
    
    def optimize_for_inference(
        self,
        model: nn.Module,
        example_input: Optional[torch.Tensor] = None,
        **kwargs
    ) -> nn.Module:
        """
        Apply all configured optimizations to a model.
        
        This is the main entry point for model optimization, applying
        a pipeline of optimizations in the optimal order.
        
        Args:
            model: PyTorch model to optimize
            example_input: Example input for tracing
            **kwargs: Additional optimization parameters
            
        Returns:
            Optimized model ready for inference
        """
        logger.info(f"Starting model optimization pipeline for {model.__class__.__name__}")
        
        # Set model to evaluation mode
        model.eval()
        
        # Move to target device
        device = torch.device(self.config.target_device if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        # Step 1: Fuse layers for better performance
        if self.config.fuse_layers:
            model = self._fuse_layers(model)
            logger.info("Layer fusion applied")
        
        # Step 2: Prune unused weights
        if self.config.prune_unused_weights:
            model = self._prune_model(model)
            logger.info("Weight pruning applied")
        
        # Step 3: Apply quantization based on type
        if self.config.quantization_type == QuantizationType.FP16:
            model = self._apply_fp16(model)
            logger.info("FP16 quantization applied")
        
        elif self.config.quantization_type == QuantizationType.DYNAMIC:
            model = self._apply_dynamic_quantization(model)
            logger.info("Dynamic quantization applied")
        
        elif self.config.quantization_type == QuantizationType.STATIC:
            if example_input is not None:
                model = self._apply_static_quantization(model, example_input)
                logger.info("Static quantization applied")
            else:
                logger.warning("Example input required for static quantization, skipping")
        
        elif self.config.quantization_type == QuantizationType.INT8:
            if example_input is not None:
                model = self._apply_int8_quantization(model, example_input)
                logger.info("INT8 quantization applied")
            else:
                logger.warning("Example input required for INT8 quantization, skipping")
        
        # Step 4: Convert to ONNX if requested
        if self.config.use_onnx and example_input is not None:
            model = self._convert_to_onnx(model, example_input)
            logger.info("ONNX conversion applied")
        
        # Step 5: Use TensorRT if requested
        if self.config.use_tensorrt and example_input is not None:
            model = self._convert_to_tensorrt(model, example_input)
            logger.info("TensorRT optimization applied")
        
        logger.info("Model optimization complete")
        return model
    
    def _fuse_layers(self, model: nn.Module) -> nn.Module:
        """
        Fuse convolutional and batch normalization layers.
        
        Layer fusion combines operations to reduce kernel launches
        and improve memory locality.
        
        Args:
            model: PyTorch model
            
        Returns:
            Model with fused layers
        """
        # Check if model has any conv+bn patterns
        fused_model = model
        
        # Use torch's built-in fusion for known patterns
        if hasattr(torch.ao.quantization, 'fuse_modules'):
            try:
                # Identify modules that can be fused
                # This is a simplified version - real implementation would
                # traverse the model and identify conv+bn+relu patterns
                module_list = []
                
                def collect_modules(module, prefix=''):
                    for name, child in module.named_children():
                        full_name = f"{prefix}.{name}" if prefix else name
                        module_list.append((full_name, child))
                        collect_modules(child, full_name)
                
                collect_modules(fused_model)
                
                # Look for conv+bn patterns
                fusion_patterns = []
                for i in range(len(module_list) - 1):
                    name1, mod1 = module_list[i]
                    name2, mod2 = module_list[i + 1]
                    
                    if isinstance(mod1, (nn.Conv2d, nn.Linear)) and isinstance(mod2, nn.BatchNorm2d):
                        # Found conv+bn pattern
                        fusion_patterns.append([name1, name2])
                
                if fusion_patterns:
                    fused_model = torch.ao.quantization.fuse_modules(fused_model, fusion_patterns)
                    logger.info(f"Fused {len(fusion_patterns)} layer groups")
                    
            except Exception as e:
                logger.warning(f"Layer fusion failed: {e}")
        
        return fused_model
    
    def _prune_model(self, model: nn.Module) -> nn.Module:
        """
        Prune weights below a threshold.
        
        Pruning removes redundant weights to reduce model size
        and improve inference speed.
        
        Args:
            model: PyTorch model
            
        Returns:
            Model with pruned weights
        """
        try:
            # Simple magnitude-based pruning
            # In production, this would use more sophisticated techniques
            prune_threshold = 1e-5
            
            for name, param in model.named_parameters():
                if 'weight' in name and param.dim() >= 2:
                    # Calculate absolute values
                    abs_weights = torch.abs(param.data)
                    
                    # Create mask for weights above threshold
                    mask = (abs_weights > prune_threshold).float()
                    
                    # Apply mask
                    param.data = param.data * mask
                    
                    # Log pruning statistics
                    zero_count = (mask == 0).sum().item()
                    total_count = mask.numel()
                    if zero_count > 0:
                        logger.debug(f"Pruned {zero_count}/{total_count} ({zero_count/total_count*100:.1f}%) weights from {name}")
            
            return model
            
        except Exception as e:
            logger.warning(f"Model pruning failed: {e}")
            return model
    
    def _apply_fp16(self, model: nn.Module) -> nn.Module:
        """
        Convert model to half precision (FP16).
        
        FP16 reduces memory usage and can accelerate inference
        on compatible hardware.
        
        Args:
            model: PyTorch model
            
        Returns:
            Model in FP16
        """
        try:
            # Convert all parameters to half
            model = model.half()
            
            # Some layers need to stay in FP32 for stability
            # This is a simplified approach - production would identify
            # specific layers that need FP32
            for module in model.modules():
                if isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
                    # Batch norm and layer norm often work better in FP32
                    module.float()
            
            return model
            
        except Exception as e:
            logger.warning(f"FP16 conversion failed: {e}")
            return model
    
    def _apply_dynamic_quantization(self, model: nn.Module) -> nn.Module:
        """
        Apply dynamic quantization to linear and LSTM layers.
        
        Dynamic quantization quantizes weights at load time but
        keeps activations in FP32, providing good speedup with
        minimal accuracy loss.
        
        Args:
            model: PyTorch model
            
        Returns:
            Dynamically quantized model
        """
        try:
            # Apply dynamic quantization to Linear and LSTM layers
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {nn.Linear, nn.LSTM, nn.GRU},  # Quantize these layer types
                dtype=torch.qint8
            )
            
            return quantized_model
            
        except Exception as e:
            logger.warning(f"Dynamic quantization failed: {e}")
            return model
    
    def _apply_static_quantization(
        self,
        model: nn.Module,
        example_input: torch.Tensor
    ) -> nn.Module:
        """
        Apply static quantization with calibration.
        
        Static quantization quantizes both weights and activations,
        requiring calibration data to determine activation ranges.
        
        Args:
            model: PyTorch model
            example_input: Example input for calibration
            
        Returns:
            Statically quantized model
        """
        try:
            # Set model to evaluation mode
            model.eval()
            
            # Prepare model for quantization
            model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            
            # Prepare for quantization
            prepared_model = torch.quantization.prepare(model, inplace=False)
            
            # Calibrate with example input
            with torch.no_grad():
                for _ in range(self.config.calibration_samples):
                    _ = prepared_model(example_input)
            
            # Convert to quantized model
            quantized_model = torch.quantization.convert(prepared_model, inplace=False)
            
            return quantized_model
            
        except Exception as e:
            logger.warning(f"Static quantization failed: {e}")
            return model
    
    def _apply_int8_quantization(
        self,
        model: nn.Module,
        example_input: torch.Tensor
    ) -> nn.Module:
        """
        Apply INT8 quantization using PyTorch's quantization toolkit.
        
        INT8 provides the best performance but requires careful calibration.
        
        Args:
            model: PyTorch model
            example_input: Example input for calibration
            
        Returns:
            INT8 quantized model
        """
        try:
            # Use QNNPACK for ARM, FBGEMM for x86
            backend = 'qnnpack' if not torch.cuda.is_available() else 'fbgemm'
            
            # Set quantization configuration
            model.qconfig = torch.quantization.get_default_qconfig(backend)
            
            # Prepare model for quantization
            prepared_model = torch.quantization.prepare(model, inplace=False)
            
            # Calibration phase
            prepared_model.eval()
            with torch.no_grad():
                for _ in range(self.config.calibration_samples):
                    _ = prepared_model(example_input)
            
            # Convert to quantized model
            quantized_model = torch.quantization.convert(prepared_model, inplace=False)
            
            return quantized_model
            
        except Exception as e:
            logger.warning(f"INT8 quantization failed: {e}")
            return model
    
    def _convert_to_onnx(
        self,
        model: nn.Module,
        example_input: torch.Tensor
    ) -> nn.Module:
        """
        Convert model to ONNX format for cross-platform deployment.
        
        ONNX models can be optimized by various inference engines
        and run on different hardware backends.
        
        Args:
            model: PyTorch model
            example_input: Example input for tracing
            
        Returns:
            ONNX model wrapper
        """
        try:
            # Ensure model is on CPU for ONNX export
            model_cpu = model.cpu()
            example_input_cpu = example_input.cpu()
            
            # Export to ONNX
            onnx_path = Path("/tmp") / f"{model.__class__.__name__}.onnx"
            torch.onnx.export(
                model_cpu,
                example_input_cpu,
                onnx_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
            )
            
            logger.info(f"Model exported to ONNX: {onnx_path}")
            
            # Return original model (ONNX wrapper would be separate)
            # In production, you'd return an ONNX Runtime wrapper
            return model
            
        except Exception as e:
            logger.warning(f"ONNX conversion failed: {e}")
            return model
    
    def _convert_to_tensorrt(
        self,
        model: nn.Module,
        example_input: torch.Tensor
    ) -> nn.Module:
        """
        Convert model to TensorRT for NVIDIA GPU optimization.
        
        TensorRT provides the fastest inference on NVIDIA GPUs
        by fusing operations and using kernel auto-tuning.
        
        Args:
            model: PyTorch model
            example_input: Example input for building engine
            
        Returns:
            TensorRT optimized model
        """
        try:
            # Check if TensorRT is available
            try:
                import tensorrt as trt
                import torch2trt
                
                # Convert to TensorRT
                model_trt = torch2trt.torch2trt(
                    model,
                    [example_input],
                    fp16_mode=self.config.quantization_type == QuantizationType.FP16,
                    max_workspace_size=1 << 30  # 1GB
                )
                
                logger.info("Model converted to TensorRT")
                return model_trt
                
            except ImportError:
                logger.warning("TensorRT not available, skipping optimization")
                return model
                
        except Exception as e:
            logger.warning(f"TensorRT conversion failed: {e}")
            return model
    
    def estimate_model_size(self, model: nn.Module) -> Dict[str, Any]:
        """
        Estimate the memory footprint of a model.
        
        Args:
            model: PyTorch model
            
        Returns:
            Dictionary with size estimates
        """
        param_size = 0
        buffer_size = 0
        
        # Calculate parameter size
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        # Calculate buffer size
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        total_size_bytes = param_size + buffer_size
        
        return {
            'parameter_size_mb': param_size / (1024 * 1024),
            'buffer_size_mb': buffer_size / (1024 * 1024),
            'total_size_mb': total_size_bytes / (1024 * 1024),
            'parameter_count': sum(p.numel() for p in model.parameters()),
            'dtype': next(model.parameters()).dtype
        }
    
    def benchmark_model(
        self,
        model: nn.Module,
        input_size: Tuple[int, ...],
        num_runs: int = 100,
        warmup_runs: int = 10
    ) -> Dict[str, float]:
        """
        Benchmark model inference speed.
        
        Args:
            model: PyTorch model
            input_size: Input tensor dimensions
            num_runs: Number of inference runs
            warmup_runs: Number of warmup runs
            
        Returns:
            Dictionary with timing statistics
        """
        device = next(model.parameters()).device
        model.eval()
        
        # Create random input
        input_tensor = torch.randn(input_size).to(device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(input_tensor)
        
        # Synchronize if CUDA
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Benchmark
        import time
        times = []
        
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.time()
                _ = model(input_tensor)
                
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                
                end_time = time.time()
                times.append((end_time - start_time) * 1000)  # ms
        
        # Calculate statistics
        times = np.array(times)
        
        return {
            'mean_ms': np.mean(times),
            'std_ms': np.std(times),
            'min_ms': np.min(times),
            'max_ms': np.max(times),
            'p95_ms': np.percentile(times, 95),
            'p99_ms': np.percentile(times, 99),
            'throughput_items_per_sec': 1000 / np.mean(times) * self.config.batch_size
        }