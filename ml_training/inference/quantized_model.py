"""
Quantized Model for Efficient Inference.

This module provides tools for model quantization to reduce memory footprint
and improve inference speed with minimal accuracy loss.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union, List
import logging
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


class QuantizedModel:
    """
    Wrapper for quantized models with various quantization techniques.
    
    Supports:
    - Post-training quantization (dynamic and static)
    - Quantization-aware training (QAT)
    - Mixed precision inference
    - FP16, INT8, INT4 quantization
    """
    
    def __init__(
        self,
        model: nn.Module,
        quantize_dtype: str = 'int8',
        per_channel: bool = True,
        per_tensor: bool = False
    ):
        """
        Initialize quantized model.
        
        Args:
            model: Original PyTorch model
            quantize_dtype: Quantization dtype ('fp16', 'int8', 'int4')
            per_channel: Per-channel quantization for weights
            per_tensor: Per-tensor quantization for activations
        """
        self.original_model = model
        self.quantize_dtype = quantize_dtype
        self.per_channel = per_channel
        self.per_tensor = per_tensor
        self.quantized_model = None
        
        logger.info(f"Initializing quantization with dtype: {quantize_dtype}")
    
    def apply_post_training_quantization(
        self,
        calibration_loader: Optional[torch.utils.data.DataLoader] = None,
        num_calibration_batches: int = 100
    ) -> nn.Module:
        """
        Apply post-training quantization.
        
        Args:
            calibration_loader: DataLoader for calibration (required for static quantization)
            num_calibration_batches: Number of batches for calibration
            
        Returns:
            Quantized model
        """
        if self.quantize_dtype == 'fp16':
            return self._quantize_fp16()
        elif self.quantize_dtype == 'int8':
            return self._quantize_int8(calibration_loader, num_calibration_batches)
        elif self.quantize_dtype == 'int4':
            return self._quantize_int4()
        else:
            raise ValueError(f"Unsupported quantization dtype: {self.quantize_dtype}")
    
    def _quantize_fp16(self) -> nn.Module:
        """
        Convert model to half precision (FP16).
        
        Returns:
            FP16 model
        """
        logger.info("Converting model to FP16")
        model_fp16 = self.original_model.half()
        
        # Move to appropriate device
        if torch.cuda.is_available():
            model_fp16 = model_fp16.cuda()
        
        self.quantized_model = model_fp16
        return model_fp16
    
    def _quantize_int8(
        self,
        calibration_loader: Optional[torch.utils.data.DataLoader],
        num_calibration_batches: int
    ) -> nn.Module:
        """
        Apply INT8 quantization using PyTorch's built-in quantization.
        
        Args:
            calibration_loader: DataLoader for calibration
            num_calibration_batches: Number of batches for calibration
            
        Returns:
            INT8 quantized model
        """
        logger.info("Applying INT8 quantization")
        
        # Set quantization configuration
        if self.per_channel:
            qconfig = torch.quantization.get_default_qconfig('fbgemm')
        else:
            qconfig = torch.quantization.get_default_qconfig('qnnpack')
        
        # Prepare model for quantization
        model_to_quantize = self.original_model
        model_to_quantize.eval()
        model_to_quantize.qconfig = qconfig
        
        # Fuse modules where possible
        model_to_quantize = torch.quantization.fuse_modules(
            model_to_quantize,
            [['conv1', 'bn1', 'relu1']]  # Adjust based on model architecture
        )
        
        # Prepare for quantization
        model_prepared = torch.quantization.prepare(model_to_quantize, inplace=False)
        
        # Calibrate with representative data
        if calibration_loader:
            logger.info(f"Calibrating with {num_calibration_batches} batches")
            with torch.no_grad():
                for i, batch in enumerate(calibration_loader):
                    if i >= num_calibration_batches:
                        break
                    
                    # Handle different batch formats
                    if isinstance(batch, dict):
                        inputs = batch['input_ids']
                    elif isinstance(batch, (list, tuple)):
                        inputs = batch[0]
                    else:
                        inputs = batch
                    
                    # Run calibration
                    model_prepared(inputs)
        
        # Convert to quantized model
        quantized_model = torch.quantization.convert(model_prepared, inplace=False)
        
        self.quantized_model = quantized_model
        return quantized_model
    
    def _quantize_int4(self) -> nn.Module:
        """
        Apply INT4 quantization using custom implementation.
        
        Note: INT4 quantization requires custom kernels or libraries like bitsandbytes.
        
        Returns:
            INT4 quantized model
        """
        logger.warning("INT4 quantization requires additional libraries (bitsandbytes)")
        
        try:
            import bitsandbytes as bnb
            
            # Replace linear layers with 4-bit versions
            for name, module in self.original_model.named_children():
                if isinstance(module, nn.Linear):
                    setattr(
                        self.original_model,
                        name,
                        bnb.nn.Linear4bit(
                            module.in_features,
                            module.out_features,
                            bias=module.bias is not None
                        )
                    )
            
            self.quantized_model = self.original_model
            return self.original_model
            
        except ImportError:
            logger.error("bitsandbytes not installed. Falling back to INT8 quantization")
            return self._quantize_int8(None, 0)
    
    def apply_quantization_aware_training(
        self,
        train_loader: torch.utils.data.DataLoader,
        num_epochs: int = 3,
        learning_rate: float = 1e-5
    ) -> nn.Module:
        """
        Apply Quantization-Aware Training (QAT).
        
        Args:
            train_loader: DataLoader for training
            num_epochs: Number of QAT epochs
            learning_rate: Learning rate for QAT
            
        Returns:
            Quantized model
        """
        logger.info("Applying Quantization-Aware Training")
        
        # Set quantization configuration
        qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        model_qat = self.original_model
        model_qat.qconfig = qconfig
        
        # Prepare for QAT
        model_qat = torch.quantization.prepare_qat(model_qat, inplace=False)
        
        # Training setup
        optimizer = torch.optim.Adam(model_qat.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # QAT training loop
        model_qat.train()
        for epoch in range(num_epochs):
            total_loss = 0.0
            for batch in train_loader:
                # Handle different batch formats
                if isinstance(batch, dict):
                    inputs = batch['input_ids']
                    labels = batch['labels']
                elif isinstance(batch, (list, tuple)):
                    inputs, labels = batch[0], batch[1]
                else:
                    inputs, labels = batch, None
                
                optimizer.zero_grad()
                outputs = model_qat(inputs)
                
                if labels is not None:
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
            
            logger.info(f"QAT Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}")
        
        # Convert to quantized model
        quantized_model = torch.quantization.convert(model_qat, inplace=False)
        
        self.quantized_model = quantized_model
        return quantized_model
    
    def dynamic_quantization(self) -> nn.Module:
        """
        Apply dynamic quantization (weights only, not activations).
        
        Returns:
            Dynamically quantized model
        """
        logger.info("Applying dynamic quantization")
        
        quantized_model = torch.quantization.quantize_dynamic(
            self.original_model,
            {nn.Linear, nn.LSTM, nn.GRU},
            dtype=torch.qint8
        )
        
        self.quantized_model = quantized_model
        return quantized_model
    
    def get_size_reduction(self) -> float:
        """
        Calculate model size reduction after quantization.
        
        Returns:
            Size reduction ratio (original_size / quantized_size)
        """
        if self.quantized_model is None:
            logger.warning("Quantized model not available")
            return 1.0
        
        # Calculate parameter sizes
        original_size = sum(p.numel() * p.element_size() for p in self.original_model.parameters())
        quantized_size = sum(p.numel() * p.element_size() for p in self.quantized_model.parameters())
        
        return original_size / quantized_size if quantized_size > 0 else 1.0
    
    def get_accuracy_drop(
        self,
        test_loader: torch.utils.data.DataLoader,
        original_accuracy: float
    ) -> float:
        """
        Calculate accuracy drop after quantization.
        
        Args:
            test_loader: DataLoader for testing
            original_accuracy: Original model accuracy
            
        Returns:
            Accuracy drop percentage
        """
        if self.quantized_model is None:
            logger.warning("Quantized model not available")
            return 0.0
        
        # Evaluate quantized model
        self.quantized_model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in test_loader:
                # Handle different batch formats
                if isinstance(batch, dict):
                    inputs = batch['input_ids']
                    labels = batch['labels']
                elif isinstance(batch, (list, tuple)):
                    inputs, labels = batch[0], batch[1]
                else:
                    inputs, labels = batch, None
                
                outputs = self.quantized_model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        quantized_accuracy = correct / total
        accuracy_drop = original_accuracy - quantized_accuracy
        
        logger.info(f"Original accuracy: {original_accuracy:.4f}")
        logger.info(f"Quantized accuracy: {quantized_accuracy:.4f}")
        logger.info(f"Accuracy drop: {accuracy_drop:.4f} ({accuracy_drop * 100:.2f}%)")
        
        return accuracy_drop


class MixedPrecisionInference:
    """
    Mixed precision inference for optimal performance.
    
    Uses FP16 for compute and FP32 for accumulation to balance
    speed and precision.
    """
    
    def __init__(self, model: nn.Module):
        """
        Initialize mixed precision inference.
        
        Args:
            model: PyTorch model
        """
        self.model = model
        self.model.eval()
        
        # Enable automatic mixed precision
        if torch.cuda.is_available():
            self.amp_enabled = True
        else:
            self.amp_enabled = False
            logger.warning("Mixed precision requires CUDA, using FP32")
    
    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Run inference with mixed precision.
        
        Args:
            inputs: Input tensor
            
        Returns:
            Output tensor
        """
        if self.amp_enabled:
            with torch.cuda.amp.autocast():
                outputs = self.model(inputs)
        else:
            outputs = self.model(inputs)
        
        return outputs
    
    def predict_batch(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Run batch inference with mixed precision.
        
        Args:
            inputs: List of input tensors
            
        Returns:
            List of output tensors
        """
        outputs = []
        
        for inp in inputs:
            if self.amp_enabled:
                with torch.cuda.amp.autocast():
                    out = self.model(inp)
            else:
                out = self.model(inp)
            outputs.append(out)
        
        return outputs
    
    def get_performance_improvement(
        self,
        test_inputs: List[torch.Tensor],
        num_runs: int = 100
    ) -> Dict[str, float]:
        """
        Measure performance improvement with mixed precision.
        
        Args:
            test_inputs: Test inputs for benchmarking
            num_runs: Number of runs for averaging
            
        Returns:
            Performance metrics
        """
        import time
        
        # Measure FP32 performance
        self.amp_enabled = False
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        fp32_times = []
        for _ in range(num_runs):
            start = time.time()
            for inp in test_inputs:
                _ = self.predict(inp)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            fp32_times.append(time.time() - start)
        
        # Measure mixed precision performance
        self.amp_enabled = True
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        amp_times = []
        for _ in range(num_runs):
            start = time.time()
            for inp in test_inputs:
                _ = self.predict(inp)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            amp_times.append(time.time() - start)
        
        fp32_avg = np.mean(fp32_times)
        amp_avg = np.mean(amp_times)
        
        return {
            'fp32_avg_time_ms': fp32_avg * 1000,
            'amp_avg_time_ms': amp_avg * 1000,
            'speedup': fp32_avg / amp_avg,
            'improvement_percent': ((fp32_avg - amp_avg) / fp32_avg) * 100
        }


def quantize_model(
    model: nn.Module,
    method: str = 'dynamic',
    calibration_loader: Optional[torch.utils.data.DataLoader] = None,
    **kwargs
) -> nn.Module:
    """
    Convenience function to quantize a model.
    
    Args:
        model: PyTorch model
        method: Quantization method ('dynamic', 'static', 'fp16', 'qat')
        calibration_loader: DataLoader for calibration (required for static)
        **kwargs: Additional arguments for quantization
        
    Returns:
        Quantized model
    """
    quantizer = QuantizedModel(model, **kwargs)
    
    if method == 'dynamic':
        return quantizer.dynamic_quantization()
    elif method == 'static':
        return quantizer.apply_post_training_quantization(calibration_loader)
    elif method == 'fp16':
        return quantizer.apply_post_training_quantization()
    elif method == 'qat':
        return quantizer.apply_quantization_aware_training(calibration_loader)
    else:
        raise ValueError(f"Unsupported quantization method: {method}")