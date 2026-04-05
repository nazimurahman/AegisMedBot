"""
Model Server for Production Inference.

This module provides a production-ready model server with:
- Model loading and management
- Batch inference
- Async processing
- GPU support
- Model versioning
- Health checks
- Metrics collection
"""

import asyncio
import logging
import time
import threading
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import asyncio
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    """Model status enumeration."""
    LOADING = "loading"
    LOADED = "loaded"
    ERROR = "error"
    UNLOADED = "unloaded"


@dataclass
class InferenceRequest:
    """Inference request data structure."""
    id: str
    inputs: List[Any]
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class InferenceResponse:
    """Inference response data structure."""
    id: str
    outputs: List[Any]
    processing_time_ms: float
    model_version: str
    error: Optional[str] = None
    metadata: Dict[str, Any] = None


class ModelServer:
    """
    Production model server with advanced features.
    
    Features:
    - Async inference
    - Batch processing
    - Model versioning
    - Automatic batching
    - GPU acceleration
    - Health monitoring
    - Metrics collection
    """
    
    def __init__(
        self,
        model_path: Union[str, Path],
        model_class: type,
        device: str = None,
        batch_size: int = 32,
        max_queue_size: int = 1000,
        max_workers: int = 4,
        auto_batch: bool = True,
        auto_batch_timeout: float = 0.1
    ):
        """
        Initialize model server.
        
        Args:
            model_path: Path to the model checkpoint
            model_class: Model class to instantiate
            device: Device to run inference on
            batch_size: Maximum batch size for inference
            max_queue_size: Maximum size of request queue
            max_workers: Number of worker threads
            auto_batch: Whether to automatically batch requests
            auto_batch_timeout: Timeout for batch collection
        """
        self.model_path = Path(model_path)
        self.model_class = model_class
        self.batch_size = batch_size
        self.max_queue_size = max_queue_size
        self.max_workers = max_workers
        self.auto_batch = auto_batch
        self.auto_batch_timeout = auto_batch_timeout
        
        # Setup device
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model state
        self.model = None
        self.model_version = None
        self.status = ModelStatus.UNLOADED
        self.load_time = None
        
        # Request queue
        self.request_queue = asyncio.Queue(maxsize=max_queue_size)
        
        # Executor for CPU inference
        if self.device.type == 'cpu':
            self.executor = ThreadPoolExecutor(max_workers=max_workers)
        else:
            self.executor = None
        
        # Metrics
        self.metrics = {
            'total_requests': 0,
            'total_batches': 0,
            'total_processing_time': 0,
            'errors': 0,
            'batch_sizes': []
        }
        
        # Start background processor
        self.running = False
        self.processor_task = None
        
        logger.info(f"ModelServer initialized with device: {self.device}")
        logger.info(f"Model path: {self.model_path}")
    
    def load_model(self, model_version: str = None):
        """
        Load model from checkpoint.
        
        Args:
            model_version: Specific model version to load
        """
        self.status = ModelStatus.LOADING
        logger.info("Loading model...")
        
        try:
            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Determine model version
            if model_version:
                self.model_version = model_version
            elif 'metadata' in checkpoint and 'version' in checkpoint['metadata']:
                self.model_version = checkpoint['metadata']['version']
            else:
                self.model_version = '1.0.0'
            
            # Instantiate model
            if 'config' in checkpoint:
                self.model = self.model_class(**checkpoint['config'])
            else:
                self.model = self.model_class()
            
            # Load weights
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            # Move to device and set to eval mode
            self.model.to(self.device)
            self.model.eval()
            
            self.status = ModelStatus.LOADED
            self.load_time = time.time()
            
            logger.info(f"Model loaded successfully (version: {self.model_version})")
            
        except Exception as e:
            self.status = ModelStatus.ERROR
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def unload_model(self):
        """Unload model from memory."""
        if self.model is not None:
            self.model = None
            self.status = ModelStatus.UNLOADED
            logger.info("Model unloaded")
    
    async def predict(
        self,
        inputs: List[Any],
        request_id: str = None,
        timeout: float = 30.0
    ) -> InferenceResponse:
        """
        Run inference on inputs.
        
        Args:
            inputs: List of inputs for inference
            request_id: Optional request identifier
            timeout: Request timeout in seconds
            
        Returns:
            InferenceResponse object
        """
        if self.status != ModelStatus.LOADED:
            raise RuntimeError(f"Model not loaded (status: {self.status})")
        
        if request_id is None:
            request_id = f"req_{int(time.time() * 1000)}_{id(inputs)}"
        
        # Create request
        request = InferenceRequest(
            id=request_id,
            inputs=inputs,
            metadata={'timestamp': time.time()}
        )
        
        if self.auto_batch:
            # Queue for automatic batching
            await self.request_queue.put(request)
            
            # Create future for response
            future = asyncio.Future()
            request.metadata['future'] = future
            
            # Wait for response
            try:
                response = await asyncio.wait_for(future, timeout=timeout)
                return response
            except asyncio.TimeoutError:
                request.metadata.get('future', future).cancel()
                raise TimeoutError(f"Request {request_id} timed out after {timeout}s")
        else:
            # Process immediately
            return await self._process_single(request)
    
    async def _process_single(self, request: InferenceRequest) -> InferenceResponse:
        """
        Process a single request.
        
        Args:
            request: Inference request
            
        Returns:
            Inference response
        """
        start_time = time.time()
        
        try:
            # Run inference
            if self.device.type == 'cuda':
                outputs = await self._inference_gpu(request.inputs)
            else:
                outputs = await self._inference_cpu(request.inputs)
            
            processing_time = (time.time() - start_time) * 1000
            
            # Update metrics
            self.metrics['total_requests'] += 1
            self.metrics['total_processing_time'] += processing_time
            
            return InferenceResponse(
                id=request.id,
                outputs=outputs,
                processing_time_ms=processing_time,
                model_version=self.model_version,
                metadata=request.metadata
            )
            
        except Exception as e:
            self.metrics['errors'] += 1
            logger.error(f"Inference error for request {request.id}: {str(e)}")
            
            return InferenceResponse(
                id=request.id,
                outputs=[],
                processing_time_ms=(time.time() - start_time) * 1000,
                model_version=self.model_version,
                error=str(e),
                metadata=request.metadata
            )
    
    async def _process_batch(self, requests: List[InferenceRequest]) -> List[InferenceResponse]:
        """
        Process a batch of requests.
        
        Args:
            requests: List of inference requests
            
        Returns:
            List of inference responses
        """
        start_time = time.time()
        
        # Collect all inputs
        all_inputs = []
        for req in requests:
            all_inputs.extend(req.inputs)
        
        try:
            # Run batch inference
            if self.device.type == 'cuda':
                outputs = await self._inference_gpu(all_inputs)
            else:
                outputs = await self._inference_cpu(all_inputs)
            
            processing_time = (time.time() - start_time) * 1000
            
            # Distribute outputs back to requests
            responses = []
            idx = 0
            for req in requests:
                num_inputs = len(req.inputs)
                req_outputs = outputs[idx:idx + num_inputs]
                idx += num_inputs
                
                responses.append(InferenceResponse(
                    id=req.id,
                    outputs=req_outputs,
                    processing_time_ms=processing_time,
                    model_version=self.model_version,
                    metadata=req.metadata
                ))
            
            # Update metrics
            self.metrics['total_requests'] += len(requests)
            self.metrics['total_batches'] += 1
            self.metrics['total_processing_time'] += processing_time
            self.metrics['batch_sizes'].append(len(requests))
            
            return responses
            
        except Exception as e:
            self.metrics['errors'] += len(requests)
            logger.error(f"Batch inference error: {str(e)}")
            
            # Return error responses
            return [
                InferenceResponse(
                    id=req.id,
                    outputs=[],
                    processing_time_ms=(time.time() - start_time) * 1000,
                    model_version=self.model_version,
                    error=str(e),
                    metadata=req.metadata
                )
                for req in requests
            ]
    
    async def _inference_gpu(self, inputs: List[Any]) -> List[Any]:
        """
        Run inference on GPU.
        
        Args:
            inputs: List of inputs
            
        Returns:
            List of outputs
        """
        # Convert inputs to tensors
        if isinstance(inputs[0], torch.Tensor):
            input_tensors = inputs
        elif isinstance(inputs[0], np.ndarray):
            input_tensors = [torch.from_numpy(x) for x in inputs]
        else:
            input_tensors = [torch.tensor(x) for x in inputs]
        
        # Move to device
        input_tensors = [x.to(self.device) for x in input_tensors]
        
        # Batch processing
        outputs = []
        for i in range(0, len(input_tensors), self.batch_size):
            batch = input_tensors[i:i + self.batch_size]
            
            # Stack batch if possible
            if len(batch) > 1 and all(b.shape == batch[0].shape for b in batch):
                batch_tensor = torch.stack(batch)
                with torch.no_grad():
                    batch_output = self.model(batch_tensor)
                
                # Handle different output types
                if isinstance(batch_output, tuple):
                    batch_output = batch_output[0]
                
                outputs.extend(batch_output.cpu().numpy())
            else:
                # Process individually
                for item in batch:
                    with torch.no_grad():
                        item_output = self.model(item.unsqueeze(0))
                        if isinstance(item_output, tuple):
                            item_output = item_output[0]
                        outputs.append(item_output.squeeze(0).cpu().numpy())
        
        return outputs
    
    async def _inference_cpu(self, inputs: List[Any]) -> List[Any]:
        """
        Run inference on CPU with thread pool.
        
        Args:
            inputs: List of inputs
            
        Returns:
            List of outputs
        """
        # Run inference in thread pool
        loop = asyncio.get_event_loop()
        
        def inference_task():
            with torch.no_grad():
                # Convert inputs
                if isinstance(inputs[0], torch.Tensor):
                    input_tensors = inputs
                else:
                    input_tensors = [torch.tensor(x) for x in inputs]
                
                # Process in batches
                outputs = []
                for i in range(0, len(input_tensors), self.batch_size):
                    batch = input_tensors[i:i + self.batch_size]
                    
                    if len(batch) > 1 and all(b.shape == batch[0].shape for b in batch):
                        batch_tensor = torch.stack(batch)
                        batch_output = self.model(batch_tensor)
                        if isinstance(batch_output, tuple):
                            batch_output = batch_output[0]
                        outputs.extend(batch_output.numpy())
                    else:
                        for item in batch:
                            item_output = self.model(item.unsqueeze(0))
                            if isinstance(item_output, tuple):
                                item_output = item_output[0]
                            outputs.append(item_output.squeeze(0).numpy())
                
                return outputs
        
        return await loop.run_in_executor(self.executor, inference_task)
    
    async def _batch_processor(self):
        """
        Background task for automatic batching.
        """
        while self.running:
            try:
                # Collect batch
                batch = []
                try:
                    # Get first request
                    first = await asyncio.wait_for(
                        self.request_queue.get(),
                        timeout=self.auto_batch_timeout
                    )
                    batch.append(first)
                    
                    # Collect more requests
                    while len(batch) < self.batch_size:
                        try:
                            req = self.request_queue.get_nowait()
                            batch.append(req)
                        except asyncio.QueueEmpty:
                            break
                            
                except asyncio.TimeoutError:
                    continue
                
                # Process batch
                responses = await self._process_batch(batch)
                
                # Send responses back to futures
                for response in responses:
                    future = response.metadata.get('future')
                    if future and not future.done():
                        future.set_result(response)
                        
            except Exception as e:
                logger.error(f"Batch processor error: {str(e)}")
    
    async def start(self):
        """Start the model server."""
        if self.auto_batch:
            self.running = True
            self.processor_task = asyncio.create_task(self._batch_processor())
            logger.info("Model server started with automatic batching")
    
    async def stop(self):
        """Stop the model server."""
        if self.auto_batch:
            self.running = False
            if self.processor_task:
                self.processor_task.cancel()
                try:
                    await self.processor_task
                except asyncio.CancelledError:
                    pass
            logger.info("Model server stopped")
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get server metrics.
        
        Returns:
            Dictionary with metrics
        """
        avg_batch_size = (
            sum(self.metrics['batch_sizes']) / len(self.metrics['batch_sizes'])
            if self.metrics['batch_sizes'] else 0
        )
        
        avg_processing_time = (
            self.metrics['total_processing_time'] / self.metrics['total_requests']
            if self.metrics['total_requests'] > 0 else 0
        )
        
        return {
            'model_version': self.model_version,
            'status': self.status.value,
            'total_requests': self.metrics['total_requests'],
            'total_batches': self.metrics['total_batches'],
            'errors': self.metrics['errors'],
            'avg_processing_time_ms': avg_processing_time,
            'avg_batch_size': avg_batch_size,
            'queue_size': self.request_queue.qsize(),
            'device': str(self.device),
            'load_time': self.load_time
        }
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check.
        
        Returns:
            Health status dictionary
        """
        return {
            'status': self.status.value,
            'healthy': self.status == ModelStatus.LOADED,
            'model_version': self.model_version,
            'device': str(self.device)
        }