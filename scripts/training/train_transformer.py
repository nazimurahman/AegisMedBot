"""
Transformer model training script for AegisMedBot.

This script trains a transformer-based model for medical text understanding,
clinical prediction, and patient risk assessment.

The training pipeline includes:
- Data loading and preprocessing for medical text
- Model configuration with LoRA for efficient fine-tuning
- Training loop with gradient accumulation and mixed precision
- Validation and evaluation metrics
- Model checkpointing and export

Usage:
    python scripts/training/train_transformer.py --config config.yaml
    python scripts/training/train_transformer.py --model-name microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext
"""

import sys
import os
import argparse
import logging
import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
import random
import numpy as np

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import PyTorch and transformer libraries
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    get_linear_schedule_with_warmup
)
from datasets import Dataset as HFDataset
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from peft import LoraConfig, get_peft_model, TaskType

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility across different libraries.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

class MedicalTextDataset(Dataset):
    """
    Custom dataset for medical text classification and prediction.
    
    This dataset handles medical text data including clinical notes,
    patient summaries, and other medical documentation.
    """
    
    def __init__(
        self,
        texts: list,
        labels: list,
        tokenizer,
        max_length: int = 512,
        is_training: bool = True
    ):
        """
        Initialize the medical text dataset.
        
        Args:
            texts: List of text strings
            labels: List of labels (can be None for inference)
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            is_training: Whether this is training data
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_training = is_training
        
    def __len__(self) -> int:
        """Return the total number of samples."""
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary containing tokenized inputs and labels
        """
        text = str(self.texts[idx])
        
        # Tokenize the text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Remove batch dimension
        item = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }
        
        # Add labels if available
        if self.labels is not None and self.is_training:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return item

class MedicalDataProcessor:
    """
    Process medical data for transformer model training.
    
    This class handles loading, cleaning, and preparing medical text data
    for training and evaluation.
    """
    
    def __init__(self, data_path: Optional[Path] = None):
        """
        Initialize the data processor.
        
        Args:
            data_path: Path to data directory
        """
        self.data_path = data_path
        
    def load_data(self, task_type: str = 'classification') -> Dict[str, Any]:
        """
        Load and preprocess medical data.
        
        Args:
            task_type: Type of task ('classification', 'qa', 'ner')
            
        Returns:
            Dictionary containing processed data
        """
        logger.info(f"Loading data for task: {task_type}")
        
        # For demonstration, create synthetic medical QA data
        # In production, this would load from actual datasets
        
        if task_type == 'classification':
            # Create synthetic classification data
            medical_conditions = [
                'hypertension', 'diabetes', 'asthma', 'copd', 
                'heart_failure', 'pneumonia', 'uti', 'stroke'
            ]
            
            texts = []
            labels = []
            
            # Generate synthetic clinical notes
            for i in range(1000):
                # Randomly select condition
                condition_idx = random.randint(0, len(medical_conditions) - 1)
                condition = medical_conditions[condition_idx]
                
                # Generate synthetic text
                text_templates = [
                    f"Patient presents with symptoms consistent with {condition}. "
                    f"Blood pressure elevated, heart rate normal. "
                    f"History of {condition} in family. "
                    f"Prescribed medication for management.",
                    
                    f"Follow-up visit for {condition}. "
                    f"Patient reports improvement with current treatment. "
                    f"Vital signs stable. "
                    f"Continue current medication regimen.",
                    
                    f"Emergency admission for acute exacerbation of {condition}. "
                    f"Patient in distress, requiring immediate intervention. "
                    f"Admitted to ICU for monitoring and treatment."
                ]
                
                text = random.choice(text_templates)
                texts.append(text)
                labels.append(condition_idx)
            
            return {
                'texts': texts,
                'labels': labels,
                'num_classes': len(medical_conditions),
                'class_names': medical_conditions
            }
            
        elif task_type == 'qa':
            # Create synthetic question answering data
            questions = [
                "What is the recommended treatment for hypertension?",
                "What are the symptoms of diabetes?",
                "How is pneumonia diagnosed?",
                "What medications are used for asthma?",
                "What are the risk factors for heart disease?"
            ]
            
            contexts = [
                "Hypertension treatment includes lifestyle modifications and medications such as ACE inhibitors, ARBs, and calcium channel blockers.",
                "Diabetes symptoms include increased thirst, frequent urination, fatigue, and blurred vision.",
                "Pneumonia diagnosis involves chest X-ray, blood tests, and sputum culture to identify the causative organism.",
                "Asthma medications include bronchodilators for quick relief and inhaled corticosteroids for long-term control.",
                "Heart disease risk factors include high blood pressure, high cholesterol, smoking, diabetes, and family history."
            ]
            
            answers = [
                "ACE inhibitors, ARBs, and calcium channel blockers",
                "Increased thirst, frequent urination, fatigue, and blurred vision",
                "Chest X-ray, blood tests, and sputum culture",
                "Bronchodilators and inhaled corticosteroids",
                "High blood pressure, high cholesterol, smoking, diabetes, and family history"
            ]
            
            return {
                'questions': questions,
                'contexts': contexts,
                'answers': answers
            }
        
        else:
            raise ValueError(f"Unsupported task type: {task_type}")

class MedicalTransformerTrainer:
    """
    Trainer for medical transformer models with advanced features.
    
    Supports:
    - LoRA fine-tuning for parameter efficiency
    - Mixed precision training
    - Gradient accumulation
    - Early stopping
    - Model checkpointing
    """
    
    def __init__(
        self,
        model_name: str,
        task_type: str = 'classification',
        num_labels: int = 2,
        use_lora: bool = True,
        device: Optional[str] = None
    ):
        """
        Initialize the trainer.
        
        Args:
            model_name: HuggingFace model name
            task_type: Type of task ('classification', 'qa', 'ner')
            num_labels: Number of output classes
            use_lora: Whether to use LoRA fine-tuning
            device: Device to use ('cuda', 'cpu')
        """
        self.model_name = model_name
        self.task_type = task_type
        self.num_labels = num_labels
        self.use_lora = use_lora
        
        # Determine device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Ensure padding token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model based on task type
        self.model = self._load_model()
        
        # Apply LoRA if requested
        if use_lora:
            self.model = self._apply_lora()
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize gradient scaler for mixed precision
        self.scaler = GradScaler()
        
        # Training history
        self.train_history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_f1': []
        }
    
    def _load_model(self):
        """
        Load the appropriate model based on task type.
        
        Returns:
            Loaded model
        """
        logger.info(f"Loading model: {self.model_name}")
        
        if self.task_type == 'classification':
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=self.num_labels,
                ignore_mismatched_sizes=True
            )
        elif self.task_type == 'qa':
            model = AutoModelForQuestionAnswering.from_pretrained(
                self.model_name
            )
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")
        
        return model
    
    def _apply_lora(self):
        """
        Apply LoRA (Low-Rank Adaptation) to the model.
        
        LoRA reduces the number of trainable parameters by adding
        low-rank matrices to the attention layers.
        
        Returns:
            Model with LoRA adapters
        """
        logger.info("Applying LoRA fine-tuning configuration")
        
        # Configure LoRA
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS if self.task_type == 'classification' else TaskType.QUESTION_ANS,
            r=16,  # Rank of low-rank matrices
            lora_alpha=32,  # Scaling factor
            target_modules=['query', 'value'],  # Modules to apply LoRA to
            lora_dropout=0.1,  # Dropout rate
            bias='none'  # Don't train bias terms
        )
        
        # Apply LoRA to model
        model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        
        logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,}")
        logger.info(f"Percentage of trainable parameters: {100 * trainable_params / total_params:.2f}%")
        
        return model
    
    def train(
        self,
        train_dataset: MedicalTextDataset,
        val_dataset: Optional[MedicalTextDataset] = None,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        num_epochs: int = 3,
        warmup_steps: int = 500,
        weight_decay: float = 0.01,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        use_mixed_precision: bool = True,
        early_stopping_patience: int = 3,
        checkpoint_dir: Optional[Path] = None
    ):
        """
        Train the transformer model.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            num_epochs: Number of training epochs
            warmup_steps: Number of warmup steps for scheduler
            weight_decay: Weight decay for optimizer
            gradient_accumulation_steps: Steps to accumulate gradients
            max_grad_norm: Maximum gradient norm for clipping
            use_mixed_precision: Whether to use mixed precision training
            early_stopping_patience: Patience for early stopping
            checkpoint_dir: Directory to save checkpoints
        """
        logger.info("Starting training...")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Learning rate: {learning_rate}")
        logger.info(f"Number of epochs: {num_epochs}")
        logger.info(f"Mixed precision: {use_mixed_precision}")
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = None
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )
        
        # Initialize optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Calculate total training steps
        total_steps = len(train_loader) * num_epochs // gradient_accumulation_steps
        
        # Initialize scheduler
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Training loop
        global_step = 0
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        
        for epoch in range(num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
            logger.info("-" * 50)
            
            # Training phase
            self.model.train()
            total_train_loss = 0
            num_train_batches = 0
            
            for batch_idx, batch in enumerate(train_loader):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass with mixed precision
                if use_mixed_precision:
                    with autocast():
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels
                        )
                        loss = outputs.loss / gradient_accumulation_steps
                else:
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss / gradient_accumulation_steps
                
                # Backward pass
                if use_mixed_precision:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # Update weights after accumulating gradients
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    # Clip gradients
                    if use_mixed_precision:
                        self.scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                        optimizer.step()
                    
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                
                total_train_loss += loss.item() * gradient_accumulation_steps
                num_train_batches += 1
                
                # Log progress
                if (batch_idx + 1) % 10 == 0:
                    avg_loss = total_train_loss / num_train_batches
                    logger.info(f"Batch {batch_idx + 1}/{len(train_loader)} - Loss: {avg_loss:.4f}")
            
            # Calculate average training loss for epoch
            avg_train_loss = total_train_loss / num_train_batches
            self.train_history['train_loss'].append(avg_train_loss)
            logger.info(f"Average training loss: {avg_train_loss:.4f}")
            
            # Validation phase
            if val_loader:
                val_loss, val_accuracy, val_f1 = self.evaluate(val_loader, use_mixed_precision)
                self.train_history['val_loss'].append(val_loss)
                self.train_history['val_accuracy'].append(val_accuracy)
                self.train_history['val_f1'].append(val_f1)
                
                logger.info(f"Validation loss: {val_loss:.4f}")
                logger.info(f"Validation accuracy: {val_accuracy:.4f}")
                logger.info(f"Validation F1 score: {val_f1:.4f}")
                
                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_without_improvement = 0
                    
                    # Save best model
                    if checkpoint_dir:
                        self.save_checkpoint(checkpoint_dir / 'best_model')
                        logger.info(f"Saved best model to {checkpoint_dir / 'best_model'}")
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= early_stopping_patience:
                        logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                        break
            
            # Save checkpoint at end of epoch
            if checkpoint_dir:
                self.save_checkpoint(checkpoint_dir / f'checkpoint_epoch_{epoch + 1}')
        
        logger.info("Training completed!")
        return self.train_history
    
    def evaluate(
        self,
        dataloader: DataLoader,
        use_mixed_precision: bool = True
    ) -> Tuple[float, float, float]:
        """
        Evaluate the model on validation data.
        
        Args:
            dataloader: Validation data loader
            use_mixed_precision: Whether to use mixed precision
            
        Returns:
            Tuple of (loss, accuracy, f1_score)
        """
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                if use_mixed_precision:
                    with autocast():
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels
                        )
                else:
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                
                total_loss += outputs.loss.item()
                
                # Get predictions
                predictions = torch.argmax(outputs.logits, dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(dataloader)
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted'
        )
        
        return avg_loss, accuracy, f1
    
    def predict(self, texts: list) -> np.ndarray:
        """
        Make predictions on new texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            Array of predictions
        """
        self.model.eval()
        
        # Tokenize texts
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors='pt'
        )
        
        # Move to device
        input_ids = encodings['input_ids'].to(self.device)
        attention_mask = encodings['attention_mask'].to(self.device)
        
        # Make predictions
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            predictions = torch.softmax(outputs.logits, dim=-1)
        
        return predictions.cpu().numpy()
    
    def save_checkpoint(self, path: Path):
        """
        Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        if self.use_lora:
            self.model.save_pretrained(path)
        else:
            self.model.save_pretrained(path)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(path)
        
        # Save training configuration
        config = {
            'model_name': self.model_name,
            'task_type': self.task_type,
            'num_labels': self.num_labels,
            'use_lora': self.use_lora,
            'train_history': self.train_history,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(path / 'training_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: Path):
        """
        Load model checkpoint.
        
        Args:
            path: Path to checkpoint
        """
        path = Path(path)
        
        # Load model
        if self.use_lora:
            from peft import PeftModel
            base_model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=self.num_labels
            )
            self.model = PeftModel.from_pretrained(base_model, path)
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(path)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        
        # Move to device
        self.model.to(self.device)
        
        logger.info(f"Checkpoint loaded from {path}")

def main():
    """
    Main execution function for training.
    
    Parses command line arguments and runs the training pipeline.
    """
    parser = argparse.ArgumentParser(
        description='Train transformer model for medical text understanding'
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default='microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
        help='HuggingFace model name'
    )
    parser.add_argument(
        '--task',
        type=str,
        choices=['classification', 'qa', 'ner'],
        default='classification',
        help='Type of task to train for'
    )
    parser.add_argument(
        '--num-labels',
        type=int,
        default=2,
        help='Number of output labels for classification'
    )
    parser.add_argument(
        '--no-lora',
        action='store_true',
        help='Disable LoRA fine-tuning'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Training batch size'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=3,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=2e-5,
        help='Learning rate'
    )
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='./checkpoints',
        help='Directory to save checkpoints'
    )
    
    args = parser.parse_args()
    
    # Load configuration from file if provided
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
            for key, value in config.items():
                setattr(args, key, value)
    
    logger.info("=" * 60)
    logger.info("Medical Transformer Training")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Task: {args.task}")
    logger.info(f"Use LoRA: {not args.no_lora}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info("=" * 60)
    
    # Initialize data processor
    processor = MedicalDataProcessor()
    
    # Load and prepare data
    data = processor.load_data(task_type=args.task)
    
    if args.task == 'classification':
        # Prepare datasets
        texts = data['texts']
        labels = data['labels']
        num_labels = data['num_classes']
        
        # Split data into train and validation
        split_idx = int(0.8 * len(texts))
        train_texts = texts[:split_idx]
        train_labels = labels[:split_idx]
        val_texts = texts[split_idx:]
        val_labels = labels[split_idx:]
        
        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Create datasets
        train_dataset = MedicalTextDataset(
            texts=train_texts,
            labels=train_labels,
            tokenizer=tokenizer,
            max_length=512,
            is_training=True
        )
        
        val_dataset = MedicalTextDataset(
            texts=val_texts,
            labels=val_labels,
            tokenizer=tokenizer,
            max_length=512,
            is_training=True
        )
        
        # Initialize trainer
        trainer = MedicalTransformerTrainer(
            model_name=args.model_name,
            task_type=args.task,
            num_labels=num_labels,
            use_lora=not args.no_lora
        )
        
        # Train model
        checkpoint_dir = Path(args.checkpoint_dir) / datetime.now().strftime('%Y%m%d_%H%M%S')
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        history = trainer.train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            num_epochs=args.epochs,
            checkpoint_dir=checkpoint_dir
        )
        
        # Save final model
        trainer.save_checkpoint(checkpoint_dir / 'final_model')
        
        logger.info("=" * 60)
        logger.info("Training completed successfully!")
        logger.info(f"Best validation loss: {min(history['val_loss']):.4f}")
        logger.info(f"Best validation accuracy: {max(history['val_accuracy']):.4f}")
        logger.info(f"Model saved to: {checkpoint_dir}")
        logger.info("=" * 60)
        
    else:
        logger.warning(f"Task type {args.task} not yet fully implemented in this script")
        logger.info("Please use classification task for demonstration")

if __name__ == "__main__":
    main()