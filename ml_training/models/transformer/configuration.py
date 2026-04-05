"""
Configuration management for medical transformer models.
Handles model configurations, hyperparameters, and training settings.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field, asdict
from transformers import BertConfig, AutoConfig
import logging

logger = logging.getLogger(__name__)

@dataclass
class MedicalModelConfig:
    """
    Configuration for medical transformer models.
    Extends HuggingFace config with medical-specific parameters.
    """
    
    # Model architecture
    model_type: str = "medical_bert"
    pretrained_model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    
    # Vocabulary and tokenization
    vocab_size: int = 30522
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    
    # Medical-specific features
    use_clinical_embeddings: bool = True
    use_medical_attention: bool = True
    clinical_vocab_size: int = 10000
    medical_attention_heads: int = 8
    
    # Task-specific settings
    num_labels: Optional[int] = None
    num_classes: Optional[int] = None
    num_qa_tokens: Optional[int] = None
    
    # Training hyperparameters
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_grad_norm: float = 1.0
    batch_size: int = 16
    eval_batch_size: int = 32
    num_epochs: int = 3
    
    # Optimization
    use_fp16: bool = False
    gradient_accumulation_steps: int = 1
    optimizer_type: str = "adamw"
    scheduler_type: str = "linear"
    
    # Regularization
    dropout_rate: float = 0.1
    attention_dropout: float = 0.1
    layer_norm_eps: float = 1e-12
    
    # Data processing
    max_seq_length: int = 512
    doc_stride: int = 128
    max_query_length: int = 64
    
    # Logging and saving
    save_steps: int = 1000
    eval_steps: int = 500
    logging_steps: int = 100
    save_total_limit: int = 3
    
    # System settings
    device: str = "cuda"
    num_workers: int = 4
    seed: int = 42
    
    def __post_init__(self):
        """Post-initialization processing."""
        # Set default number of labels if not specified
        if self.num_labels is None and self.num_classes is not None:
            self.num_labels = self.num_classes
    
    def to_huggingface_config(self) -> BertConfig:
        """
        Convert to HuggingFace BERT configuration.
        
        Returns:
            BertConfig object for HuggingFace models
        """
        return BertConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            layer_norm_eps=self.layer_norm_eps,
            num_labels=self.num_labels
        )
    
    def save(self, save_path: Union[str, Path]):
        """
        Save configuration to file.
        
        Args:
            save_path: Path to save configuration
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as JSON
        with open(save_path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
        
        logger.info(f"Configuration saved to {save_path}")
    
    @classmethod
    def load(cls, load_path: Union[str, Path]) -> 'MedicalModelConfig':
        """
        Load configuration from file.
        
        Args:
            load_path: Path to configuration file
            
        Returns:
            Loaded configuration object
        """
        load_path = Path(load_path)
        
        if load_path.suffix == '.json':
            with open(load_path, 'r') as f:
                config_dict = json.load(f)
        elif load_path.suffix in ['.yaml', '.yml']:
            with open(load_path, 'r') as f:
                config_dict = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported file format: {load_path.suffix}")
        
        return cls(**config_dict)
    
    def update(self, updates: Dict[str, Any]):
        """
        Update configuration with new values.
        
        Args:
            updates: Dictionary of updates to apply
        """
        for key, value in updates.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logger.debug(f"Updated {key} = {value}")
            else:
                logger.warning(f"Unknown configuration key: {key}")


@dataclass
class TrainingConfig:
    """
    Training-specific configuration.
    Contains settings for the training process.
    """
    
    # Data
    train_data_path: Optional[Path] = None
    eval_data_path: Optional[Path] = None
    test_data_path: Optional[Path] = None
    
    # Output
    output_dir: Path = Path("./outputs")
    model_name: str = "medical_bert"
    checkpoint_dir: Path = Path("./checkpoints")
    
    # Training
    num_epochs: int = 3
    batch_size: int = 16
    eval_batch_size: int = 32
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    
    # Validation
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 100
    
    # Early stopping
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001
    
    # Mixed precision
    use_fp16: bool = False
    use_bf16: bool = False
    
    # Distributed training
    local_rank: int = -1
    world_size: int = 1
    gradient_accumulation_steps: int = 1
    
    # Logging
    use_wandb: bool = False
    wandb_project: str = "medical-transformer"
    wandb_run_name: Optional[str] = None
    use_tensorboard: bool = True
    
    # Resuming
    resume_from_checkpoint: Optional[Path] = None
    
    def save(self, save_path: Union[str, Path]):
        """
        Save training configuration.
        
        Args:
            save_path: Path to save configuration
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
        
        logger.info(f"Training config saved to {save_path}")
    
    @classmethod
    def load(cls, load_path: Union[str, Path]) -> 'TrainingConfig':
        """
        Load training configuration from file.
        
        Args:
            load_path: Path to configuration file
            
        Returns:
            Loaded training configuration
        """
        load_path = Path(load_path)
        
        with open(load_path, 'r') as f:
            if load_path.suffix == '.json':
                config_dict = json.load(f)
            elif load_path.suffix in ['.yaml', '.yml']:
                config_dict = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported file format: {load_path.suffix}")
        
        return cls(**config_dict)


@dataclass
class ModelOptimizationConfig:
    """
    Configuration for model optimization and inference.
    """
    
    # Quantization
    use_quantization: bool = False
    quantization_type: str = "dynamic"  # dynamic, static, qat
    quantization_bits: int = 8
    
    # Pruning
    use_pruning: bool = False
    pruning_method: str = "magnitude"  # magnitude, l1, random
    pruning_sparsity: float = 0.3
    
    # Distillation
    use_distillation: bool = False
    teacher_model_name: Optional[str] = None
    distillation_temperature: float = 3.0
    distillation_alpha: float = 0.5
    
    # ONNX export
    export_onnx: bool = False
    onnx_opset_version: int = 14
    onnx_input_names: List[str] = field(default_factory=lambda: ["input_ids", "attention_mask"])
    onnx_output_names: List[str] = field(default_factory=lambda: ["logits"])
    
    # TensorRT
    use_tensorrt: bool = False
    tensorrt_precision: str = "fp16"  # fp32, fp16, int8
    
    # Inference
    batch_size: int = 1
    max_length: int = 512
    use_cache: bool = True
    
    def save(self, save_path: Union[str, Path]):
        """
        Save optimization configuration.
        
        Args:
            save_path: Path to save configuration
        """
        save_path = Path(save_path)
        with open(save_path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, load_path: Union[str, Path]) -> 'ModelOptimizationConfig':
        """
        Load optimization configuration from file.
        
        Args:
            load_path: Path to configuration file
            
        Returns:
            Loaded optimization configuration
        """
        load_path = Path(load_path)
        
        with open(load_path, 'r') as f:
            if load_path.suffix == '.json':
                config_dict = json.load(f)
            else:
                config_dict = yaml.safe_load(f)
        
        return cls(**config_dict)


class ConfigManager:
    """
    Central configuration manager for all model configurations.
    Handles loading, saving, and merging configurations.
    """
    
    def __init__(self, base_config_path: Optional[Path] = None):
        """
        Initialize configuration manager.
        
        Args:
            base_config_path: Path to base configuration file
        """
        self.base_config = None
        self.model_config = None
        self.training_config = None
        self.optimization_config = None
        
        if base_config_path:
            self.load_all(base_config_path)
    
    def load_all(self, config_path: Path):
        """
        Load all configurations from a base path.
        
        Args:
            config_path: Path to directory containing config files
        """
        config_path = Path(config_path)
        
        # Load model config
        model_config_path = config_path / "model_config.json"
        if model_config_path.exists():
            self.model_config = MedicalModelConfig.load(model_config_path)
        
        # Load training config
        training_config_path = config_path / "training_config.json"
        if training_config_path.exists():
            self.training_config = TrainingConfig.load(training_config_path)
        
        # Load optimization config
        optimization_config_path = config_path / "optimization_config.json"
        if optimization_config_path.exists():
            self.optimization_config = ModelOptimizationConfig.load(optimization_config_path)
        
        logger.info(f"Loaded configurations from {config_path}")
    
    def save_all(self, save_path: Path):
        """
        Save all configurations to a directory.
        
        Args:
            save_path: Path to directory for saving configs
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        if self.model_config:
            self.model_config.save(save_path / "model_config.json")
        
        if self.training_config:
            self.training_config.save(save_path / "training_config.json")
        
        if self.optimization_config:
            self.optimization_config.save(save_path / "optimization_config.json")
        
        logger.info(f"Saved configurations to {save_path}")
    
    def get_model_config(self) -> MedicalModelConfig:
        """Get model configuration."""
        if self.model_config is None:
            self.model_config = MedicalModelConfig()
        return self.model_config
    
    def get_training_config(self) -> TrainingConfig:
        """Get training configuration."""
        if self.training_config is None:
            self.training_config = TrainingConfig()
        return self.training_config
    
    def get_optimization_config(self) -> ModelOptimizationConfig:
        """Get optimization configuration."""
        if self.optimization_config is None:
            self.optimization_config = ModelOptimizationConfig()
        return self.optimization_config
    
    def merge_with_args(self, args: Dict[str, Any]):
        """
        Merge configuration with command line arguments.
        
        Args:
            args: Dictionary of command line arguments
        """
        # Update model config
        if self.model_config:
            model_updates = {
                k: v for k, v in args.items()
                if hasattr(self.model_config, k)
            }
            self.model_config.update(model_updates)
        
        # Update training config
        if self.training_config:
            training_updates = {
                k: v for k, v in args.items()
                if hasattr(self.training_config, k)
            }
            self.training_config.update(training_updates)
        
        # Update optimization config
        if self.optimization_config:
            opt_updates = {
                k: v for k, v in args.items()
                if hasattr(self.optimization_config, k)
            }
            self.optimization_config.update(opt_updates)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert all configurations to dictionary.
        
        Returns:
            Dictionary containing all configurations
        """
        return {
            'model_config': asdict(self.model_config) if self.model_config else None,
            'training_config': asdict(self.training_config) if self.training_config else None,
            'optimization_config': asdict(self.optimization_config) if self.optimization_config else None
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ConfigManager':
        """
        Create configuration manager from dictionary.
        
        Args:
            config_dict: Dictionary containing configurations
            
        Returns:
            ConfigManager instance
        """
        manager = cls()
        
        if config_dict.get('model_config'):
            manager.model_config = MedicalModelConfig(**config_dict['model_config'])
        
        if config_dict.get('training_config'):
            manager.training_config = TrainingConfig(**config_dict['training_config'])
        
        if config_dict.get('optimization_config'):
            manager.optimization_config = ModelOptimizationConfig(**config_dict['optimization_config'])
        
        return manager


# Default configuration for quick setup
DEFAULT_CONFIG = MedicalModelConfig(
    model_type="medical_bert",
    pretrained_model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    num_labels=2,
    learning_rate=2e-5,
    batch_size=16,
    num_epochs=3
)

# Example usage
if __name__ == "__main__":
    # Create configuration
    config = MedicalModelConfig()
    print(f"Model config: {config.model_type}")
    print(f"Hidden size: {config.hidden_size}")
    
    # Save and load
    config.save("test_config.json")
    loaded_config = MedicalModelConfig.load("test_config.json")
    print(f"Loaded config: {loaded_config.model_type}")
    
    # Create training config
    train_config = TrainingConfig()
    train_config.num_epochs = 5
    train_config.batch_size = 32
    print(f"Training config: {train_config.num_epochs} epochs")
    
    # Use config manager
    manager = ConfigManager()
    manager.model_config = config
    manager.training_config = train_config
    manager.save_all("configs/")
    
    # Load from directory
    loaded_manager = ConfigManager("configs/")
    print(f"Loaded manager model: {loaded_manager.model_config.model_type}")