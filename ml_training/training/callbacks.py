"""
Training Callbacks for Advanced Monitoring and Control.

This module provides callback classes that allow injecting custom behavior
at various stages of the training process. Callbacks enable:
- Early stopping based on custom conditions
- Model checkpointing with versioning
- Learning rate monitoring and adjustment
- Metrics logging to multiple backends
- Gradient monitoring and visualization
- Model pruning and quantization during training
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
from datetime import datetime
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)

class Callback:
    """
    Base class for all training callbacks.
    
    Callbacks provide hooks into the training process at various stages:
    - on_train_begin: Called when training starts
    - on_train_end: Called when training ends
    - on_epoch_begin: Called at the start of each epoch
    - on_epoch_end: Called at the end of each epoch
    - on_batch_begin: Called at the start of each batch
    - on_batch_end: Called at the end of each batch
    - on_validation_begin: Called before validation
    - on_validation_end: Called after validation
    """
    
    def on_train_begin(self, trainer):
        """Called when training begins."""
        pass
    
    def on_train_end(self, trainer):
        """Called when training ends."""
        pass
    
    def on_epoch_begin(self, trainer, epoch):
        """Called at the start of an epoch."""
        pass
    
    def on_epoch_end(self, trainer, epoch, metrics):
        """Called at the end of an epoch."""
        pass
    
    def on_batch_begin(self, trainer, batch):
        """Called at the start of a batch."""
        pass
    
    def on_batch_end(self, trainer, batch, loss, metrics):
        """Called at the end of a batch."""
        pass
    
    def on_validation_begin(self, trainer):
        """Called before validation."""
        pass
    
    def on_validation_end(self, trainer, metrics):
        """Called after validation."""
        pass


class EarlyStoppingCallback(Callback):
    """
    Early stopping callback that monitors validation metrics.
    
    Stops training when a monitored metric stops improving for a specified
    number of epochs. Supports both minimization and maximization of metrics.
    """
    
    def __init__(
        self,
        monitor: str = 'val_loss',
        mode: str = 'min',
        patience: int = 3,
        min_delta: float = 0.001,
        restore_best_weights: bool = True
    ):
        """
        Initialize early stopping callback.
        
        Args:
            monitor: Metric to monitor (e.g., 'val_loss', 'val_accuracy')
            mode: 'min' for minimizing metric, 'max' for maximizing
            patience: Number of epochs with no improvement to wait
            min_delta: Minimum change to qualify as improvement
            restore_best_weights: Whether to restore best model weights
        """
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        
        self.best_value = None
        self.counter = 0
        self.best_epoch = 0
        self.best_weights = None
        
        logger.info(f"Early stopping configured: monitor={monitor}, mode={mode}, patience={patience}")
    
    def on_epoch_end(self, trainer, epoch, metrics):
        """
        Check for improvement at epoch end.
        
        Args:
            trainer: Trainer instance
            epoch: Current epoch number
            metrics: Dictionary of current epoch metrics
        """
        current_value = metrics.get(self.monitor)
        
        if current_value is None:
            logger.warning(f"Metric '{self.monitor}' not found in metrics")
            return
        
        # Check if this is the best value
        is_better = self._is_better(current_value)
        
        if is_better:
            # Improvement detected
            self.best_value = current_value
            self.counter = 0
            self.best_epoch = epoch
            
            if self.restore_best_weights:
                self.best_weights = {
                    k: v.clone() for k, v in trainer.model.state_dict().items()
                }
            
            logger.info(f"Improved {self.monitor} to {current_value:.4f}")
        else:
            # No improvement
            self.counter += 1
            logger.info(
                f"No improvement in {self.monitor} for {self.counter} epochs. "
                f"Best: {self.best_value:.4f}"
            )
            
            if self.counter >= self.patience:
                logger.info(f"Early stopping triggered after {epoch} epochs")
                trainer.should_stop = True
                
                if self.restore_best_weights and self.best_weights is not None:
                    logger.info("Restoring best model weights")
                    trainer.model.load_state_dict(self.best_weights)
    
    def _is_better(self, current_value: float) -> bool:
        """
        Check if current value is better than best value.
        
        Args:
            current_value: Current metric value
            
        Returns:
            True if current value is better
        """
        if self.best_value is None:
            return True
        
        if self.mode == 'min':
            return current_value < self.best_value - self.min_delta
        else:  # mode == 'max'
            return current_value > self.best_value + self.min_delta


class ModelCheckpointCallback(Callback):
    """
    Model checkpointing callback with version control.
    
    Saves model checkpoints at specified intervals with configurable
    naming and storage management. Supports saving best models and
    periodic checkpoints.
    """
    
    def __init__(
        self,
        checkpoint_dir: str,
        save_best: bool = True,
        save_last: bool = True,
        save_interval: int = 1,
        monitor: str = 'val_loss',
        mode: str = 'min',
        max_keep: int = 5
    ):
        """
        Initialize checkpoint callback.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            save_best: Whether to save best model
            save_last: Whether to save last model
            save_interval: Save checkpoint every N epochs
            monitor: Metric to monitor for best model
            mode: 'min' or 'max' for monitored metric
            max_keep: Maximum number of checkpoints to keep
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_best = save_best
        self.save_last = save_last
        self.save_interval = save_interval
        self.monitor = monitor
        self.mode = mode
        self.max_keep = max_keep
        
        self.best_value = None
        self.checkpoints = []
        
        logger.info(f"Model checkpoint callback initialized: {self.checkpoint_dir}")
    
    def on_epoch_end(self, trainer, epoch, metrics):
        """
        Save checkpoint at epoch end if conditions met.
        
        Args:
            trainer: Trainer instance
            epoch: Current epoch number
            metrics: Dictionary of epoch metrics
        """
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': trainer.model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'scheduler_state_dict': trainer.scheduler.state_dict() if trainer.scheduler else None,
            'metrics': metrics,
            'config': trainer.config
        }
        
        # Save periodic checkpoint
        if epoch % self.save_interval == 0:
            filename = f"checkpoint_epoch_{epoch}.pt"
            path = self.checkpoint_dir / filename
            torch.save(checkpoint_data, path)
            self.checkpoints.append(path)
            logger.info(f"Saved checkpoint: {path}")
        
        # Save best model
        if self.save_best:
            current_value = metrics.get(self.monitor)
            if current_value is not None:
                is_best = self._is_better(current_value)
                
                if is_best:
                    self.best_value = current_value
                    path = self.checkpoint_dir / "best_model.pt"
                    torch.save(checkpoint_data, path)
                    logger.info(f"New best model saved: {self.monitor}={current_value:.4f}")
        
        # Save last model
        if self.save_last:
            path = self.checkpoint_dir / "last_model.pt"
            torch.save(checkpoint_data, path)
            logger.debug(f"Saved last model checkpoint")
        
        # Cleanup old checkpoints
        self._cleanup_checkpoints()
    
    def _is_better(self, current_value: float) -> bool:
        """
        Check if current value is better than best value.
        
        Args:
            current_value: Current metric value
            
        Returns:
            True if current value is better
        """
        if self.best_value is None:
            return True
        
        if self.mode == 'min':
            return current_value < self.best_value
        else:
            return current_value > self.best_value
    
    def _cleanup_checkpoints(self):
        """
        Remove old checkpoints to maintain max_keep limit.
        """
        if len(self.checkpoints) > self.max_keep:
            to_remove = self.checkpoints[:-self.max_keep]
            for path in to_remove:
                if path.exists():
                    path.unlink()
                    logger.debug(f"Removed old checkpoint: {path}")
            self.checkpoints = self.checkpoints[-self.max_keep:]


class MetricsLoggerCallback(Callback):
    """
    Comprehensive metrics logging callback.
    
    Logs training metrics to multiple backends including:
    - Console logging
    - TensorBoard
    - JSON file
    - CSV file
    - Custom callbacks
    """
    
    def __init__(
        self,
        log_dir: str = './logs',
        log_interval: int = 10,
        use_tensorboard: bool = True,
        save_json: bool = True,
        save_csv: bool = True,
        custom_loggers: List[Callable] = None
    ):
        """
        Initialize metrics logger callback.
        
        Args:
            log_dir: Directory to save logs
            log_interval: Log every N batches
            use_tensorboard: Enable TensorBoard logging
            save_json: Save metrics to JSON file
            save_csv: Save metrics to CSV file
            custom_loggers: List of custom logging functions
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_interval = log_interval
        self.use_tensorboard = use_tensorboard
        self.save_json = save_json
        self.save_csv = save_csv
        self.custom_loggers = custom_loggers or []
        
        self.metrics_history = []
        
        # Initialize TensorBoard
        if use_tensorboard:
            self.writer = SummaryWriter(log_dir=self.log_dir / 'tensorboard')
        
        logger.info(f"Metrics logger initialized: {self.log_dir}")
    
    def on_batch_end(self, trainer, batch, loss, metrics):
        """
        Log batch-level metrics.
        
        Args:
            trainer: Trainer instance
            batch: Batch data
            loss: Loss value
            metrics: Dictionary of batch metrics
        """
        if trainer.current_step % self.log_interval == 0:
            # Console logging
            log_str = f"Step {trainer.current_step}: loss={loss:.4f}"
            for key, value in metrics.items():
                log_str += f", {key}={value:.4f}"
            logger.info(log_str)
            
            # TensorBoard logging
            if self.use_tensorboard:
                self.writer.add_scalar('batch/loss', loss, trainer.current_step)
                for key, value in metrics.items():
                    self.writer.add_scalar(f'batch/{key}', value, trainer.current_step)
            
            # Custom loggers
            for logger_func in self.custom_loggers:
                logger_func(trainer.current_step, loss, metrics)
    
    def on_epoch_end(self, trainer, epoch, metrics):
        """
        Log epoch-level metrics.
        
        Args:
            trainer: Trainer instance
            epoch: Current epoch number
            metrics: Dictionary of epoch metrics
        """
        # Store metrics
        epoch_metrics = {
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            **metrics
        }
        self.metrics_history.append(epoch_metrics)
        
        # Console logging
        log_str = f"Epoch {epoch}: "
        for key, value in metrics.items():
            log_str += f"{key}={value:.4f} "
        logger.info(log_str)
        
        # TensorBoard logging
        if self.use_tensorboard:
            for key, value in metrics.items():
                self.writer.add_scalar(f'epoch/{key}', value, epoch)
        
        # Save to file
        if self.save_json:
            self._save_json()
        
        if self.save_csv:
            self._save_csv()
    
    def on_train_end(self, trainer):
        """
        Final logging at training end.
        
        Args:
            trainer: Trainer instance
        """
        if self.use_tensorboard:
            self.writer.close()
        
        # Save final metrics
        self._save_json()
        self._save_csv()
        
        logger.info(f"Training completed. Metrics saved to {self.log_dir}")
    
    def _save_json(self):
        """
        Save metrics to JSON file.
        """
        json_path = self.log_dir / 'metrics.json'
        with open(json_path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
    
    def _save_csv(self):
        """
        Save metrics to CSV file.
        """
        import pandas as pd
        csv_path = self.log_dir / 'metrics.csv'
        df = pd.DataFrame(self.metrics_history)
        df.to_csv(csv_path, index=False)


class GradientMonitorCallback(Callback):
    """
    Gradient monitoring callback for debugging and analysis.
    
    Logs gradient statistics including:
    - Gradient norm
    - Gradient distribution statistics
    - Gradient sparsity
    - Per-layer gradient norms
    """
    
    def __init__(
        self,
        log_interval: int = 100,
        track_histograms: bool = False,
        log_gradient_norms: bool = True
    ):
        """
        Initialize gradient monitor callback.
        
        Args:
            log_interval: Log gradient stats every N batches
            track_histograms: Whether to track gradient histograms (slower)
            log_gradient_norms: Whether to log gradient norms
        """
        self.log_interval = log_interval
        self.track_histograms = track_histograms
        self.log_gradient_norms = log_gradient_norms
    
    def on_batch_end(self, trainer, batch, loss, metrics):
        """
        Monitor gradients after backward pass.
        
        Args:
            trainer: Trainer instance
            batch: Batch data
            loss: Loss value
            metrics: Batch metrics dictionary to update
        """
        if trainer.current_step % self.log_interval != 0:
            return
        
        # Collect gradient statistics
        grad_norms = []
        layer_grads = {}
        
        for name, param in trainer.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_norms.append(grad_norm)
                
                if self.log_gradient_norms:
                    layer_grads[name] = grad_norm
                
                if self.track_histograms:
                    # Track histogram in TensorBoard
                    if trainer.writer:
                        trainer.writer.add_histogram(
                            f'gradients/{name}',
                            param.grad,
                            trainer.current_step
                        )
        
        # Add gradient metrics
        if grad_norms:
            metrics['grad_norm_mean'] = np.mean(grad_norms)
            metrics['grad_norm_std'] = np.std(grad_norms)
            metrics['grad_norm_max'] = np.max(grad_norms)
            metrics['grad_norm_min'] = np.min(grad_norms)
            
            # Log layer-wise gradients if enabled
            if self.log_gradient_norms and trainer.writer:
                for name, norm in layer_grads.items():
                    trainer.writer.add_scalar(
                        f'grad_norm/{name}',
                        norm,
                        trainer.current_step
                    )


class LearningRateMonitorCallback(Callback):
    """
    Learning rate monitoring callback.
    
    Tracks learning rate changes and logs them to configured backends.
    """
    
    def __init__(self, log_interval: int = 1):
        """
        Initialize learning rate monitor.
        
        Args:
            log_interval: Log learning rate every N steps
        """
        self.log_interval = log_interval
    
    def on_batch_end(self, trainer, batch, loss, metrics):
        """
        Log learning rate at batch end.
        
        Args:
            trainer: Trainer instance
            batch: Batch data
            loss: Loss value
            metrics: Batch metrics dictionary to update
        """
        if trainer.current_step % self.log_interval != 0:
            return
        
        if trainer.scheduler:
            current_lr = trainer.scheduler.get_last_lr()[0]
            metrics['learning_rate'] = current_lr
            
            # Log to TensorBoard
            if trainer.writer:
                trainer.writer.add_scalar(
                    'learning_rate',
                    current_lr,
                    trainer.current_step
                )
    
    def on_epoch_end(self, trainer, epoch, metrics):
        """
        Log learning rate at epoch end.
        
        Args:
            trainer: Trainer instance
            epoch: Current epoch
            metrics: Epoch metrics dictionary to update
        """
        if trainer.scheduler:
            current_lr = trainer.scheduler.get_last_lr()[0]
            metrics['learning_rate'] = current_lr


class ModelSummaryCallback(Callback):
    """
    Model summary callback for debugging.
    
    Prints model architecture and parameter counts at the start of training.
    """
    
    def __init__(self, print_every_n_epochs: int = 5):
        """
        Initialize model summary callback.
        
        Args:
            print_every_n_epochs: Print summary every N epochs
        """
        self.print_every_n_epochs = print_every_n_epochs
    
    def on_train_begin(self, trainer):
        """
        Print model summary at training start.
        
        Args:
            trainer: Trainer instance
        """
        self._print_model_summary(trainer)
    
    def on_epoch_begin(self, trainer, epoch):
        """
        Print periodic model summary.
        
        Args:
            trainer: Trainer instance
            epoch: Current epoch
        """
        if epoch % self.print_every_n_epochs == 0:
            self._print_model_summary(trainer)
    
    def _print_model_summary(self, trainer):
        """
        Print model architecture and parameter counts.
        
        Args:
            trainer: Trainer instance
        """
        total_params = sum(p.numel() for p in trainer.model.parameters())
        trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        
        logger.info("=" * 60)
        logger.info("Model Summary:")
        logger.info(f"Architecture: {trainer.model.__class__.__name__}")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Trainable percentage: {100 * trainable_params / total_params:.2f}%")
        logger.info("=" * 60)


# Factory function to create callback collection
def create_default_callbacks(
    checkpoint_dir: str = './checkpoints',
    log_dir: str = './logs',
    early_stopping_patience: int = 3
) -> List[Callback]:
    """
    Create a default set of callbacks for training.
    
    Args:
        checkpoint_dir: Directory for model checkpoints
        log_dir: Directory for logs
        early_stopping_patience: Patience for early stopping
        
    Returns:
        List of callback instances
    """
    callbacks = [
        ModelSummaryCallback(),
        ModelCheckpointCallback(
            checkpoint_dir=checkpoint_dir,
            save_best=True,
            save_last=True,
            save_interval=1,
            monitor='val_loss',
            mode='min'
        ),
        MetricsLoggerCallback(
            log_dir=log_dir,
            log_interval=10,
            use_tensorboard=True,
            save_json=True,
            save_csv=True
        ),
        EarlyStoppingCallback(
            monitor='val_loss',
            mode='min',
            patience=early_stopping_patience,
            restore_best_weights=True
        ),
        LearningRateMonitorCallback(log_interval=10),
        GradientMonitorCallback(log_interval=50, track_histograms=False)
    ]
    
    return callbacks