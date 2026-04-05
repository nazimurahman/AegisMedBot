"""
LSTM model training script for time-series patient data.

This script trains an LSTM-based model for patient risk prediction,
clinical deterioration forecasting, and resource utilization prediction.

The training pipeline includes:
- Time-series data preprocessing
- LSTM model architecture with attention mechanism
- Training with early stopping and learning rate scheduling
- Evaluation metrics for time-series prediction
- Model export for inference

Usage:
    python scripts/training/train_lstm.py --data-path ./data/patient_vitals.csv
    python scripts/training/train_lstm.py --config config.yaml
"""

import sys
import os
import argparse
import logging
import json
import yaml
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple, List
import random
import numpy as np
import pandas as pd

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import PyTorch libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('lstm_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
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

class TimeSeriesDataset(Dataset):
    """
    Dataset for time-series patient data.
    
    Creates sliding window sequences for LSTM training.
    """
    
    def __init__(
        self,
        data: np.ndarray,
        targets: np.ndarray,
        sequence_length: int = 24,
        prediction_horizon: int = 6
    ):
        """
        Initialize the time-series dataset.
        
        Args:
            data: Feature matrix (samples, features)
            targets: Target values
            sequence_length: Number of time steps to look back
            prediction_horizon: Number of steps to predict ahead
        """
        self.data = data
        self.targets = targets
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        
        # Calculate number of valid sequences
        self.num_sequences = len(data) - sequence_length - prediction_horizon + 1
        
    def __len__(self) -> int:
        """Return the number of sequences in the dataset."""
        return self.num_sequences
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sequence and its target.
        
        Args:
            idx: Index of the sequence
            
        Returns:
            Tuple of (sequence tensor, target tensor)
        """
        # Extract sequence
        start_idx = idx
        end_idx = start_idx + self.sequence_length
        sequence = self.data[start_idx:end_idx]
        
        # Extract target (prediction_horizon steps after sequence)
        target_idx = end_idx + self.prediction_horizon - 1
        target = self.targets[target_idx]
        
        # Convert to tensors
        sequence_tensor = torch.FloatTensor(sequence)
        target_tensor = torch.FloatTensor([target])
        
        return sequence_tensor, target_tensor

class PatientRiskDataset(Dataset):
    """
    Dataset for patient risk prediction using demographic and clinical features.
    """
    
    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        patient_ids: Optional[List[str]] = None
    ):
        """
        Initialize the patient risk dataset.
        
        Args:
            features: Feature matrix (patients, features)
            labels: Risk labels (0-1 for binary classification)
            patient_ids: Optional patient identifiers
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels) if labels is not None else None
        self.patient_ids = patient_ids
        
    def __len__(self) -> int:
        """Return the number of patients in the dataset."""
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single patient's features and label.
        
        Args:
            idx: Index of the patient
            
        Returns:
            Dictionary containing features and label
        """
        item = {'features': self.features[idx]}
        if self.labels is not None:
            item['label'] = self.labels[idx]
        return item

class AttentionLSTM(nn.Module):
    """
    LSTM model with attention mechanism for time-series prediction.
    
    This model combines LSTM layers with attention to focus on
    important time steps in the sequence.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        output_size: int = 1,
        dropout: float = 0.2,
        bidirectional: bool = True,
        use_attention: bool = True
    ):
        """
        Initialize the Attention LSTM model.
        
        Args:
            input_size: Number of input features
            hidden_size: Number of hidden units in LSTM
            num_layers: Number of LSTM layers
            output_size: Size of output (1 for regression)
            dropout: Dropout rate
            bidirectional: Whether to use bidirectional LSTM
            use_attention: Whether to use attention mechanism
        """
        super(AttentionLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        
        # Calculate LSTM output dimension
        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Attention mechanism
        if use_attention:
            self.attention_weights = nn.Linear(lstm_output_size, 1)
        
        # Fully connected output layer
        self.fc = nn.Linear(lstm_output_size, output_size)
        
        # Activation for output
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor (batch_size, sequence_length, input_size)
            
        Returns:
            Output tensor (batch_size, output_size)
        """
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        # lstm_out shape: (batch_size, seq_len, hidden_size * num_directions)
        
        if self.use_attention:
            # Calculate attention weights
            attention_scores = self.attention_weights(lstm_out)
            # attention_scores shape: (batch_size, seq_len, 1)
            
            # Apply softmax to get attention weights
            attention_weights = torch.softmax(attention_scores, dim=1)
            
            # Apply attention to LSTM outputs
            weighted_output = torch.sum(attention_weights * lstm_out, dim=1)
            # weighted_output shape: (batch_size, hidden_size * num_directions)
        else:
            # Use last hidden state
            if self.bidirectional:
                # Concatenate forward and backward hidden states
                weighted_output = torch.cat((hidden[-2], hidden[-1]), dim=1)
            else:
                weighted_output = hidden[-1]
        
        # Apply dropout
        weighted_output = self.dropout(weighted_output)
        
        # Fully connected layer
        output = self.fc(weighted_output)
        
        # Apply sigmoid for binary classification
        output = self.sigmoid(output)
        
        return output

class PatientRiskLSTM(nn.Module):
    """
    LSTM model for patient risk prediction using clinical time-series data.
    """
    
    def __init__(
        self,
        feature_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        """
        Initialize the patient risk LSTM model.
        
        Args:
            feature_size: Number of input features
            hidden_size: Number of hidden units
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super(PatientRiskLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=feature_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        
        # Batch normalization
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Output layers
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor (batch_size, sequence_length, feature_size)
            
        Returns:
            Risk score (batch_size, 1)
        """
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))
        
        # Take the last output
        out = out[:, -1, :]  # (batch_size, hidden_size)
        
        # Batch normalization
        out = self.batch_norm(out)
        
        # Fully connected layers
        out = self.dropout(self.relu(self.fc1(out)))
        out = self.dropout(self.relu(self.fc2(out)))
        out = self.fc3(out)
        
        # Apply sigmoid for probability
        out = self.sigmoid(out)
        
        return out

class LSTMTrainer:
    """
    Trainer for LSTM models with advanced features.
    
    Supports:
    - Early stopping
    - Learning rate scheduling
    - Gradient clipping
    - Model checkpointing
    - Training history tracking
    """
    
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 0.001,
        device: Optional[str] = None
    ):
        """
        Initialize the LSTM trainer.
        
        Args:
            model: PyTorch model to train
            learning_rate: Initial learning rate
            device: Device to use for training
        """
        self.model = model
        
        # Determine device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Initialize scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Loss function for regression
        self.criterion = nn.MSELoss()
        
        # Training history
        self.train_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': []
        }
        
    def train_epoch(
        self,
        train_loader: DataLoader,
        gradient_clip: float = 1.0
    ) -> float:
        """
        Train the model for one epoch.
        
        Args:
            train_loader: DataLoader for training data
            gradient_clip: Maximum gradient norm for clipping
            
        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            # Move data to device
            data = data.to(self.device)
            targets = targets.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)
            
            # Update weights
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    
    def validate(
        self,
        val_loader: DataLoader
    ) -> Tuple[float, float, float, float]:
        """
        Validate the model on validation data.
        
        Args:
            val_loader: DataLoader for validation data
            
        Returns:
            Tuple of (loss, mse, mae, r2)
        """
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, targets in val_loader:
                # Move data to device
                data = data.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                all_predictions.extend(outputs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # Convert to numpy arrays
        predictions = np.array(all_predictions).flatten()
        targets = np.array(all_targets).flatten()
        
        # Calculate metrics
        avg_loss = total_loss / len(val_loader)
        mse = mean_squared_error(targets, predictions)
        mae = mean_absolute_error(targets, predictions)
        r2 = r2_score(targets, predictions)
        
        return avg_loss, mse, mae, r2
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 100,
        early_stopping_patience: int = 10,
        checkpoint_dir: Optional[Path] = None
    ) -> Dict[str, List[float]]:
        """
        Train the model with early stopping.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            num_epochs: Maximum number of epochs
            early_stopping_patience: Patience for early stopping
            checkpoint_dir: Directory to save checkpoints
            
        Returns:
            Training history dictionary
        """
        logger.info("Starting LSTM training...")
        logger.info(f"Device: {self.device}")
        logger.info(f"Number of epochs: {num_epochs}")
        logger.info(f"Early stopping patience: {early_stopping_patience}")
        
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        
        for epoch in range(num_epochs):
            # Training phase
            train_loss = self.train_epoch(train_loader)
            
            # Validation phase
            val_loss, mse, mae, r2 = self.validate(val_loader)
            
            # Update learning rate scheduler
            self.scheduler.step(val_loss)
            
            # Record learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.train_history['train_loss'].append(train_loss)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['learning_rates'].append(current_lr)
            
            # Log progress
            logger.info(
                f"Epoch {epoch + 1}/{num_epochs} - "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"MSE: {mse:.4f}, "
                f"MAE: {mae:.4f}, "
                f"R2: {r2:.4f}, "
                f"LR: {current_lr:.6f}"
            )
            
            # Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                
                # Save best model
                if checkpoint_dir:
                    self.save_checkpoint(checkpoint_dir / 'best_model')
                    logger.info(f"Saved best model to {checkpoint_dir / 'best_model'}")
            else:
                epochs_without_improvement += 1
                
                # Early stopping
                if epochs_without_improvement >= early_stopping_patience:
                    logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break
        
        logger.info("Training completed!")
        return self.train_history
    
    def save_checkpoint(self, path: Path):
        """
        Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save model state
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_history': self.train_history
        }, path / 'model_checkpoint.pt')
        
        logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: Path):
        """
        Load model checkpoint.
        
        Args:
            path: Path to checkpoint
        """
        path = Path(path)
        checkpoint = torch.load(path / 'model_checkpoint.pt', map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_history = checkpoint['train_history']
        
        logger.info(f"Checkpoint loaded from {path}")

def generate_synthetic_patient_data(
    num_patients: int = 1000,
    sequence_length: int = 24,
    feature_count: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic patient time-series data for demonstration.
    
    Args:
        num_patients: Number of patient sequences to generate
        sequence_length: Length of each sequence
        feature_count: Number of features per time step
        
    Returns:
        Tuple of (features, targets)
    """
    logger.info(f"Generating {num_patients} synthetic patient sequences...")
    
    features = []
    targets = []
    
    for patient in range(num_patients):
        # Generate base trend (improving, stable, or deteriorating)
        trend = random.choice(['improving', 'stable', 'deteriorating'])
        
        # Initialize sequence
        sequence = np.zeros((sequence_length, feature_count))
        
        for t in range(sequence_length):
            # Base values with random noise
            base_values = np.random.normal(0, 1, feature_count)
            
            # Apply trend
            if trend == 'improving':
                trend_factor = -0.05 * t  # Decreasing over time
            elif trend == 'deteriorating':
                trend_factor = 0.05 * t  # Increasing over time
            else:
                trend_factor = 0
            
            # Add trend to features
            sequence[t] = base_values + trend_factor + np.random.normal(0, 0.1, feature_count)
        
        # Generate target (risk score based on trend)
        if trend == 'improving':
            risk = np.random.uniform(0, 0.3)
        elif trend == 'stable':
            risk = np.random.uniform(0.2, 0.6)
        else:
            risk = np.random.uniform(0.5, 1.0)
        
        features.append(sequence)
        targets.append(risk)
    
    # Convert to numpy arrays
    features = np.array(features)
    targets = np.array(targets)
    
    logger.info(f"Generated data shape: features {features.shape}, targets {targets.shape}")
    
    return features, targets

def prepare_time_series_data(
    features: np.ndarray,
    targets: np.ndarray,
    test_size: float = 0.2,
    val_size: float = 0.1
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Prepare data loaders for time-series training.
    
    Args:
        features: Feature matrix
        targets: Target values
        test_size: Proportion for test set
        val_size: Proportion for validation set
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Split into train+val and test
    indices = np.arange(len(features))
    train_val_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=42
    )
    
    # Split train+val into train and val
    val_proportion = val_size / (1 - test_size)
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=val_proportion, random_state=42
    )
    
    # Create datasets
    train_dataset = TensorDataset(
        torch.FloatTensor(features[train_idx]),
        torch.FloatTensor(targets[train_idx])
    )
    
    val_dataset = TensorDataset(
        torch.FloatTensor(features[val_idx]),
        torch.FloatTensor(targets[val_idx])
    )
    
    test_dataset = TensorDataset(
        torch.FloatTensor(features[test_idx]),
        torch.FloatTensor(targets[test_idx])
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader

def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """
    Evaluate the trained model on test data.
    
    Args:
        model: Trained PyTorch model
        test_loader: DataLoader for test data
        device: Device to use for evaluation
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    model = model.to(device)
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, targets in test_loader:
            data = data.to(device)
            targets = targets.to(device)
            
            outputs = model(data)
            
            all_predictions.extend(outputs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Convert to numpy arrays
    predictions = np.array(all_predictions).flatten()
    targets = np.array(all_targets).flatten()
    
    # Calculate metrics
    mse = mean_squared_error(targets, predictions)
    mae = mean_absolute_error(targets, predictions)
    r2 = r2_score(targets, predictions)
    
    # Calculate additional metrics
    mape = np.mean(np.abs((targets - predictions) / targets)) * 100
    
    metrics = {
        'mse': mse,
        'mae': mae,
        'r2': r2,
        'mape': mape,
        'rmse': np.sqrt(mse)
    }
    
    return metrics

def main():
    """
    Main execution function for LSTM training.
    
    Parses command line arguments and runs the training pipeline.
    """
    parser = argparse.ArgumentParser(
        description='Train LSTM model for patient risk prediction'
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--data-path',
        type=str,
        help='Path to patient data CSV file'
    )
    parser.add_argument(
        '--hidden-size',
        type=int,
        default=128,
        help='Hidden size for LSTM layers'
    )
    parser.add_argument(
        '--num-layers',
        type=int,
        default=2,
        help='Number of LSTM layers'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.001,
        help='Learning rate'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size'
    )
    parser.add_argument(
        '--sequence-length',
        type=int,
        default=24,
        help='Length of input sequences'
    )
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='./lstm_checkpoints',
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
    logger.info("LSTM Model Training for Patient Risk Prediction")
    logger.info("=" * 60)
    logger.info(f"Hidden size: {args.hidden_size}")
    logger.info(f"Number of layers: {args.num_layers}")
    logger.info(f"Sequence length: {args.sequence_length}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info("=" * 60)
    
    # Generate or load data
    if args.data_path and Path(args.data_path).exists():
        logger.info(f"Loading data from {args.data_path}")
        # In production, load actual patient data
        # df = pd.read_csv(args.data_path)
        # Process data here
        pass
    else:
        logger.info("Generating synthetic patient data for demonstration")
        features, targets = generate_synthetic_patient_data(
            num_patients=5000,
            sequence_length=args.sequence_length,
            feature_count=15
        )
    
    # Prepare data loaders
    train_loader, val_loader, test_loader = prepare_time_series_data(
        features=features,
        targets=targets,
        test_size=0.2,
        val_size=0.1
    )
    
    # Get feature size from data
    feature_size = features.shape[2]
    logger.info(f"Feature size: {feature_size}")
    
    # Initialize model
    model = AttentionLSTM(
        input_size=feature_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        output_size=1,
        dropout=0.2,
        bidirectional=True,
        use_attention=True
    )
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Initialize trainer
    trainer = LSTMTrainer(
        model=model,
        learning_rate=args.learning_rate
    )
    
    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir) / datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Train model
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        early_stopping_patience=10,
        checkpoint_dir=checkpoint_dir
    )
    
    # Evaluate on test set
    logger.info("Evaluating model on test set...")
    test_metrics = evaluate_model(model, test_loader, trainer.device)
    
    logger.info("=" * 60)
    logger.info("Test Set Evaluation Results:")
    logger.info(f"  MSE: {test_metrics['mse']:.4f}")
    logger.info(f"  RMSE: {test_metrics['rmse']:.4f}")
    logger.info(f"  MAE: {test_metrics['mae']:.4f}")
    logger.info(f"  R2 Score: {test_metrics['r2']:.4f}")
    logger.info(f"  MAPE: {test_metrics['mape']:.2f}%")
    logger.info("=" * 60)
    
    # Save final model
    trainer.save_checkpoint(checkpoint_dir / 'final_model')
    
    # Save training history
    history_df = pd.DataFrame(history)
    history_df.to_csv(checkpoint_dir / 'training_history.csv', index=False)
    
    # Save evaluation metrics
    with open(checkpoint_dir / 'test_metrics.json', 'w') as f:
        json.dump(test_metrics, f, indent=2)
    
    logger.info(f"All artifacts saved to: {checkpoint_dir}")
    logger.info("LSTM training completed successfully!")

if __name__ == "__main__":
    main()