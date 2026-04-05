"""
Advanced Model Trainer for Medical Transformer Models.

This module provides a comprehensive training pipeline for medical domain models
with support for:
- Mixed precision training
- Gradient accumulation
- Learning rate scheduling
- Early stopping
- Model checkpointing
- Distributed training
- Hyperparameter optimization
- Validation and testing
- Metrics tracking
- TensorBoard logging
"""

import os
import math
import logging
from typing import Dict, Any, Optional, List, Callable, Union
from datetime import datetime
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm
import wandb

logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Advanced trainer for medical transformer models with comprehensive features.
    
    This trainer handles:
    - Training loop with progress tracking
    - Validation at specified intervals
    - Early stopping based on validation metrics
    - Model checkpointing (best and last)
    - Learning rate scheduling
    - Mixed precision training (FP16)
    - Gradient clipping
    - Metrics logging (TensorBoard, WandB)
    - Distributed training support
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optimizer,
        scheduler: _LRScheduler = None,
        criterion: nn.Module = None,
        config: Dict[str, Any] = None,
        device: str = None
    ):
        """
        Initialize the trainer with all necessary components.
        
        Args:
            model: PyTorch model to train
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            optimizer: Optimizer for parameter updates
            scheduler: Learning rate scheduler
            criterion: Loss function (defaults to CrossEntropyLoss)
            config: Training configuration dictionary
            device: Device to use for training (cuda/cpu)
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion or nn.CrossEntropyLoss()
        
        # Configuration defaults
        self.config = {
            'num_epochs': 10,
            'gradient_accumulation_steps': 1,
            'max_grad_norm': 1.0,
            'checkpoint_dir': './checkpoints',
            'log_interval': 10,
            'eval_interval': 100,
            'save_interval': 1000,
            'early_stopping_patience': 3,
            'early_stopping_threshold': 0.001,
            'mixed_precision': True,
            'use_wandb': False,
            'use_tensorboard': True,
            'seed': 42,
            'save_best_model': True,
            'save_last_model': True
        }
        
        # Update with provided config
        if config:
            self.config.update(config)
        
        # Setup device
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Setup directories
        self.checkpoint_dir = Path(self.config['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize gradient scaler for mixed precision
        self.scaler = GradScaler() if self.config['mixed_precision'] else None
        
        # Initialize tracking variables
        self.current_epoch = 0
        self.current_step = 0
        self.best_val_loss = float('inf')
        self.best_val_accuracy = 0.0
        self.early_stopping_counter = 0
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        # Setup logging
        self._setup_logging()
        
        logger.info(f"Trainer initialized with config: {self.config}")
        logger.info(f"Using device: {self.device}")
        logger.info(f"Total training steps: {len(train_loader) * self.config['num_epochs']}")
    
    def _setup_logging(self):
        """
        Setup logging with TensorBoard and Weights & Biases if enabled.
        """
        if self.config['use_tensorboard']:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=self.checkpoint_dir / 'logs')
        
        if self.config['use_wandb']:
            wandb.init(
                project=self.config.get('wandb_project', 'medintel'),
                config=self.config,
                name=self.config.get('run_name', datetime.now().strftime("%Y%m%d_%H%M%S"))
            )
    
    def train(self) -> Dict[str, Any]:
        """
        Main training loop with validation and checkpointing.
        
        Returns:
            Dictionary containing training results and best metrics
        """
        logger.info("Starting training...")
        start_time = datetime.now()
        
        for epoch in range(self.config['num_epochs']):
            self.current_epoch = epoch + 1
            logger.info(f"\n{'='*60}")
            logger.info(f"Epoch {self.current_epoch}/{self.config['num_epochs']}")
            logger.info(f"{'='*60}")
            
            # Training phase
            train_loss, train_acc = self._train_epoch()
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            
            # Validation phase
            val_loss, val_acc = self._validate()
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # Log metrics
            self._log_metrics(train_loss, train_acc, val_loss, val_acc)
            
            # Update learning rate scheduler
            if self.scheduler:
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]
                logger.info(f"Learning rate: {current_lr:.6f}")
            
            # Check for improvement
            is_best = val_loss < self.best_val_loss - self.config['early_stopping_threshold']
            if is_best:
                self.best_val_loss = val_loss
                self.best_val_accuracy = val_acc
                self.early_stopping_counter = 0
                logger.info(f"🎉 New best model! Validation loss: {val_loss:.4f}")
                
                if self.config['save_best_model']:
                    self._save_checkpoint(is_best=True)
            else:
                self.early_stopping_counter += 1
                logger.info(f"No improvement. Early stopping counter: {self.early_stopping_counter}")
            
            # Save checkpoint at epoch end
            if self.config['save_last_model']:
                self._save_checkpoint(is_best=False)
            
            # Early stopping check
            if self.early_stopping_counter >= self.config['early_stopping_patience']:
                logger.info(f"Early stopping triggered after {self.current_epoch} epochs")
                break
        
        # Training complete
        training_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"\n{'='*60}")
        logger.info(f"Training completed in {training_time:.2f} seconds")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        logger.info(f"Best validation accuracy: {self.best_val_accuracy:.4f}")
        
        # Return results
        results = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'best_val_loss': self.best_val_loss,
            'best_val_accuracy': self.best_val_accuracy,
            'training_time': training_time,
            'total_epochs': self.current_epoch
        }
        
        if self.config['use_wandb']:
            wandb.finish()
        
        return results
    
    def _train_epoch(self) -> tuple:
        """
        Train for one epoch.
        
        Returns:
            Tuple of (average_loss, accuracy) for the epoch
        """
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        # Create progress bar
        pbar = tqdm(
            self.train_loader,
            desc=f"Training Epoch {self.current_epoch}",
            leave=False
        )
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(self.device)
            
            # Forward pass with mixed precision
            if self.scaler:
                with autocast():
                    outputs = self.model(**inputs)
                    loss = self.criterion(outputs, labels)
                    loss = loss / self.config['gradient_accumulation_steps']
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
            else:
                outputs = self.model(**inputs)
                loss = self.criterion(outputs, labels)
                loss = loss / self.config['gradient_accumulation_steps']
                loss.backward()
            
            # Update weights after accumulation
            if (batch_idx + 1) % self.config['gradient_accumulation_steps'] == 0:
                # Gradient clipping
                if self.config['max_grad_norm'] > 0:
                    if self.scaler:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config['max_grad_norm']
                        )
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config['max_grad_norm']
                        )
                
                # Optimizer step
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                self.current_step += 1
            
            # Calculate metrics
            _, predicted = torch.max(outputs, 1)
            batch_correct = (predicted == labels).sum().item()
            batch_size = labels.size(0)
            
            total_loss += loss.item() * self.config['gradient_accumulation_steps']
            total_correct += batch_correct
            total_samples += batch_size
            
            # Update progress bar
            current_loss = total_loss / (batch_idx + 1)
            current_acc = total_correct / total_samples
            
            pbar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'acc': f'{current_acc:.4f}'
            })
            
            # Log at intervals
            if self.current_step % self.config['log_interval'] == 0:
                self._log_step_metrics(current_loss, current_acc, 'train')
        
        # Calculate epoch averages
        avg_loss = total_loss / len(self.train_loader)
        avg_acc = total_correct / total_samples
        
        return avg_loss, avg_acc
    
    def _validate(self) -> tuple:
        """
        Validate the model on validation set.
        
        Returns:
            Tuple of (average_loss, accuracy) for validation
        """
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        logger.info("Running validation...")
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation", leave=False):
                # Move to device
                inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'labels'}
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(**inputs)
                loss = self.criterion(outputs, labels)
                
                # Calculate metrics
                _, predicted = torch.max(outputs, 1)
                batch_correct = (predicted == labels).sum().item()
                batch_size = labels.size(0)
                
                total_loss += loss.item()
                total_correct += batch_correct
                total_samples += batch_size
        
        avg_loss = total_loss / len(self.val_loader)
        avg_acc = total_correct / total_samples
        
        return avg_loss, avg_acc
    
    def _log_metrics(
        self,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float
    ):
        """
        Log metrics to configured logging systems.
        
        Args:
            train_loss: Training loss for the epoch
            train_acc: Training accuracy for the epoch
            val_loss: Validation loss for the epoch
            val_acc: Validation accuracy for the epoch
        """
        logger.info(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # TensorBoard logging
        if self.config['use_tensorboard']:
            self.writer.add_scalar('Loss/train', train_loss, self.current_epoch)
            self.writer.add_scalar('Loss/val', val_loss, self.current_epoch)
            self.writer.add_scalar('Accuracy/train', train_acc, self.current_epoch)
            self.writer.add_scalar('Accuracy/val', val_acc, self.current_epoch)
            
            # Log learning rate
            if self.scheduler:
                lr = self.scheduler.get_last_lr()[0]
                self.writer.add_scalar('Learning_rate', lr, self.current_epoch)
        
        # Weights & Biases logging
        if self.config['use_wandb']:
            wandb.log({
                'epoch': self.current_epoch,
                'train_loss': train_loss,
                'train_accuracy': train_acc,
                'val_loss': val_loss,
                'val_accuracy': val_acc,
                'learning_rate': self.scheduler.get_last_lr()[0] if self.scheduler else 0
            })
    
    def _log_step_metrics(self, loss: float, accuracy: float, mode: str):
        """
        Log metrics at step level for finer granularity.
        
        Args:
            loss: Current loss value
            accuracy: Current accuracy value
            mode: 'train' or 'val'
        """
        if self.config['use_tensorboard']:
            self.writer.add_scalar(f'{mode}/loss_step', loss, self.current_step)
            self.writer.add_scalar(f'{mode}/accuracy_step', accuracy, self.current_step)
    
    def _save_checkpoint(self, is_best: bool = False):
        """
        Save model checkpoint.
        
        Args:
            is_best: If True, saves as best model. If False, saves as last model.
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'step': self.current_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'best_val_accuracy': self.best_val_accuracy,
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        
        # Save as best model
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model to {best_path}")
        
        # Save as last model
        if self.config['save_last_model']:
            last_path = self.checkpoint_dir / 'last_model.pt'
            torch.save(checkpoint, last_path)
            logger.info(f"Saved last model to {last_path}")
        
        # Save periodic checkpoint
        if self.current_epoch % self.config['save_interval'] == 0:
            periodic_path = self.checkpoint_dir / f'checkpoint_epoch_{self.current_epoch}.pt'
            torch.save(checkpoint, periodic_path)
            logger.info(f"Saved periodic checkpoint to {periodic_path}")
    
    def load_checkpoint(self, checkpoint_path: str, load_optimizer: bool = True):
        """
        Load training checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            load_optimizer: Whether to load optimizer and scheduler states
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if load_optimizer:
            # Load optimizer state
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load scheduler state
            if checkpoint['scheduler_state_dict'] and self.scheduler:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            # Load training state
            self.current_epoch = checkpoint['epoch']
            self.current_step = checkpoint['step']
            self.best_val_loss = checkpoint['best_val_loss']
            self.best_val_accuracy = checkpoint['best_val_accuracy']
            self.train_losses = checkpoint['train_losses']
            self.val_losses = checkpoint['val_losses']
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")