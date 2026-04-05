"""
Model Registry for managing and tracking ML models in the MedIntel platform.

This module provides a centralized registry for all machine learning models used across
the hospital intelligence system. It handles model versioning, metadata tracking,
and provides a unified interface for model loading and saving.

The registry pattern ensures that models are properly tracked, versioned, and can be
reproduced across different environments and deployments.
"""

import os
import json
import hashlib
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import torch
import pickle
import yaml

logger = logging.getLogger(__name__)

class ModelRegistry:
    """
    Central registry for managing ML models in the MedIntel platform.
    
    This class provides:
    - Model registration and tracking
    - Version management
    - Model metadata storage
    - Model loading and saving
    - Model search and filtering
    - Experiment tracking support
    
    The registry maintains a catalog of all trained models with their:
    - Unique identifiers
    - Version numbers
    - Training metadata
    - Performance metrics
    - File paths
    - Creation timestamps
    """
    
    def __init__(self, registry_path: str = "./model_registry"):
        """
        Initialize the model registry.
        
        Args:
            registry_path: Directory path where model metadata and files will be stored.
                         Defaults to "./model_registry" in the current working directory.
        """
        self.registry_path = Path(registry_path)
        self.metadata_path = self.registry_path / "metadata.json"
        self.models_path = self.registry_path / "models"
        
        # Create registry directories if they don't exist
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.models_path.mkdir(parents=True, exist_ok=True)
        
        # Load existing registry metadata or create new one
        self.metadata = self._load_metadata()
        
        logger.info(f"ModelRegistry initialized at {self.registry_path}")
        
    def _load_metadata(self) -> Dict[str, Any]:
        """
        Load existing registry metadata from disk.
        
        Returns:
            Dictionary containing all registered models with their metadata.
            If metadata file doesn't exist, returns an empty dictionary.
        """
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, 'r') as f:
                    metadata = json.load(f)
                logger.info(f"Loaded registry metadata with {len(metadata)} models")
                return metadata
            except Exception as e:
                logger.error(f"Failed to load metadata: {e}")
                return {}
        return {}
    
    def _save_metadata(self) -> None:
        """
        Save current registry metadata to disk.
        """
        try:
            with open(self.metadata_path, 'w') as f:
                json.dump(self.metadata, f, indent=2, default=str)
            logger.debug("Registry metadata saved successfully")
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
            raise
    
    def register_model(
        self,
        model: torch.nn.Module,
        model_name: str,
        version: str = None,
        metadata: Dict[str, Any] = None,
        metrics: Dict[str, float] = None,
        training_config: Dict[str, Any] = None
    ) -> str:
        """
        Register a trained model in the registry.
        
        Args:
            model: The PyTorch model instance to register
            model_name: Name identifier for the model type (e.g., "clinical_risk", "mortality")
            version: Optional version string. If not provided, auto-incremented
            metadata: Additional metadata about the model (architecture, features, etc.)
            metrics: Performance metrics (accuracy, f1, etc.)
            training_config: Configuration used for training
        
        Returns:
            model_id: Unique identifier for the registered model
        """
        # Generate model ID using timestamp and model name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Auto-generate version if not provided
        if version is None:
            existing_versions = [
                m["version"] for m in self.metadata.values() 
                if m["model_name"] == model_name
            ]
            version_num = 1
            while f"v{version_num}" in existing_versions:
                version_num += 1
            version = f"v{version_num}"
        
        model_id = f"{model_name}_{version}_{timestamp}"
        
        # Create model metadata
        model_metadata = {
            "model_id": model_id,
            "model_name": model_name,
            "version": version,
            "created_at": datetime.now().isoformat(),
            "metadata": metadata or {},
            "metrics": metrics or {},
            "training_config": training_config or {},
            "file_path": str(self.models_path / f"{model_id}.pt"),
            "model_architecture": model.__class__.__name__,
            "state_dict_hash": self._compute_state_dict_hash(model.state_dict())
        }
        
        # Save model to disk
        model_path = self.models_path / f"{model_id}.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'metadata': model_metadata,
            'config': training_config
        }, model_path)
        
        # Store metadata in registry
        self.metadata[model_id] = model_metadata
        self._save_metadata()
        
        logger.info(f"Registered model: {model_id}")
        return model_id
    
    def _compute_state_dict_hash(self, state_dict: Dict) -> str:
        """
        Compute a hash of the model's state dictionary for integrity checking.
        
        Args:
            state_dict: PyTorch model state dictionary
        
        Returns:
            SHA256 hash of the state dict as string
        """
        # Convert state dict to bytes for hashing
        state_bytes = pickle.dumps(state_dict)
        return hashlib.sha256(state_bytes).hexdigest()
    
    def load_model(
        self,
        model_id: str = None,
        model_name: str = None,
        version: str = None,
        model_class: type = None
    ) -> torch.nn.Module:
        """
        Load a model from the registry.
        
        Args:
            model_id: Direct identifier of the model to load
            model_name: Name of model to load (used with version)
            version: Version of model to load
            model_class: Class to instantiate before loading weights
        
        Returns:
            Loaded PyTorch model instance
        
        Raises:
            ValueError: If model not found or ambiguous selection
        """
        # Find model based on provided criteria
        if model_id is not None:
            # Load by direct ID
            if model_id not in self.metadata:
                raise ValueError(f"Model with ID {model_id} not found in registry")
            model_metadata = self.metadata[model_id]
        
        elif model_name is not None:
            # Find by name and optional version
            candidates = [
                m for m in self.metadata.values() 
                if m["model_name"] == model_name
            ]
            
            if version is not None:
                candidates = [c for c in candidates if c["version"] == version]
            
            if not candidates:
                raise ValueError(f"No models found for name={model_name}, version={version}")
            
            # Sort by creation time and take latest
            candidates.sort(key=lambda x: x["created_at"], reverse=True)
            model_metadata = candidates[0]
        else:
            raise ValueError("Either model_id or model_name must be provided")
        
        # Instantiate model if class provided
        if model_class is None:
            raise ValueError("model_class must be provided for loading")
        
        model = model_class()
        
        # Load weights
        model_path = Path(model_metadata["file_path"])
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info(f"Loaded model: {model_metadata['model_id']}")
        return model
    
    def list_models(
        self,
        model_name: str = None,
        version: str = None,
        min_metric_value: float = None,
        metric_name: str = None
    ) -> List[Dict[str, Any]]:
        """
        List models in the registry with optional filtering.
        
        Args:
            model_name: Filter by model name
            version: Filter by version
            min_metric_value: Minimum metric value for filtering
            metric_name: Metric name to filter on (requires min_metric_value)
        
        Returns:
            List of model metadata dictionaries matching criteria
        """
        models = list(self.metadata.values())
        
        # Apply filters
        if model_name:
            models = [m for m in models if m["model_name"] == model_name]
        
        if version:
            models = [m for m in models if m["version"] == version]
        
        if min_metric_value is not None and metric_name:
            models = [
                m for m in models 
                if m["metrics"].get(metric_name, 0) >= min_metric_value
            ]
        
        # Sort by creation date (newest first)
        models.sort(key=lambda x: x["created_at"], reverse=True)
        
        return models
    
    def delete_model(self, model_id: str) -> bool:
        """
        Delete a model from the registry.
        
        Args:
            model_id: Identifier of model to delete
        
        Returns:
            True if deletion successful, False otherwise
        """
        if model_id not in self.metadata:
            logger.warning(f"Model {model_id} not found in registry")
            return False
        
        # Delete model file
        model_path = Path(self.metadata[model_id]["file_path"])
        if model_path.exists():
            model_path.unlink()
        
        # Remove from metadata
        del self.metadata[model_id]
        self._save_metadata()
        
        logger.info(f"Deleted model: {model_id}")
        return True
    
    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific model.
        
        Args:
            model_id: Identifier of the model
        
        Returns:
            Dictionary containing model metadata
        """
        if model_id not in self.metadata:
            raise ValueError(f"Model {model_id} not found")
        
        return self.metadata[model_id].copy()
    
    def update_model_metrics(
        self,
        model_id: str,
        metrics: Dict[str, float]
    ) -> None:
        """
        Update performance metrics for an existing model.
        
        Args:
            model_id: Identifier of the model
            metrics: Dictionary of metric names and values to update
        """
        if model_id not in self.metadata:
            raise ValueError(f"Model {model_id} not found")
        
        self.metadata[model_id]["metrics"].update(metrics)
        self._save_metadata()
        
        logger.info(f"Updated metrics for model: {model_id}")
    
    def export_model(
        self,
        model_id: str,
        export_format: str = "onnx",
        export_path: Path = None
    ) -> Path:
        """
        Export model to different formats for deployment.
        
        Args:
            model_id: Identifier of the model to export
            export_format: Format to export (onnx, torchscript, etc.)
            export_path: Path to save exported model
        
        Returns:
            Path to exported model file
        """
        # Load the model first
        model = self.load_model(model_id=model_id)
        model.eval()
        
        if export_path is None:
            export_path = self.models_path / f"{model_id}.{export_format}"
        
        # Export based on format
        if export_format == "onnx":
            # Create dummy input for tracing
            dummy_input = torch.randn(1, 768)  # Adjust based on model input size
            torch.onnx.export(
                model,
                dummy_input,
                export_path,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
            )
        
        elif export_format == "torchscript":
            scripted_model = torch.jit.script(model)
            scripted_model.save(export_path)
        
        else:
            raise ValueError(f"Unsupported export format: {export_format}")
        
        logger.info(f"Exported model to {export_path}")
        return export_path