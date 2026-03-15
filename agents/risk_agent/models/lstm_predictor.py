"""
LSTM-based prediction model for patient risk assessment.

This module implements a multi-layer LSTM network with:
- Bidirectional LSTM layers for temporal pattern capture
- Attention mechanism for important time steps
- Dropout for regularization
- Skip connections for gradient flow
- Multi-task learning for multiple risk predictions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import math
import logging

logger = logging.getLogger(__name__)

class LSTMPredictor(nn.Module):
    """
    Multi-layer LSTM network for patient risk prediction.
    
    Architecture:
    - Input layer with feature normalization
    - Multiple bidirectional LSTM layers
    - Temporal attention mechanism
    - Fully connected layers for each prediction task
    - Output layer with task-specific heads
    
    Supports:
    - Mortality prediction at multiple time horizons
    - Complication risk assessment
    - ICU admission prediction
    - Length of stay estimation
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize LSTM predictor.
        
        Args:
            config: Configuration dictionary with parameters:
                - input_size: Number of input features
                - hidden_size: LSTM hidden dimension
                - num_layers: Number of LSTM layers
                - output_size: Number of output predictions
                - dropout: Dropout rate (default: 0.3)
                - bidirectional: Use bidirectional LSTM (default: True)
                - use_attention: Enable attention mechanism (default: True)
                - attention_size: Attention layer size (default: 128)
        """
        super(LSTMPredictor, self).__init__()
        
        # Store configuration
        self.config = config
        self.input_size = config.get("input_size", 128)
        self.hidden_size = config.get("hidden_size", 256)
        self.num_layers = config.get("num_layers", 2)
        self.output_size = config.get("output_size", 15)  # 3 horizons * 5 tasks
        self.dropout_rate = config.get("dropout", 0.3)
        self.bidirectional = config.get("bidirectional", True)
        self.use_attention = config.get("use_attention", True)
        self.attention_size = config.get("attention_size", 128)
        
        # Calculate directions multiplier
        self.num_directions = 2 if self.bidirectional else 1
        
        # Input normalization layer
        self.input_norm = nn.LayerNorm(self.input_size)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout_rate if self.num_layers > 1 else 0,
            bidirectional=self.bidirectional
        )
        
        # Attention mechanism
        if self.use_attention:
            self.attention = TemporalAttention(
                hidden_size=self.hidden_size * self.num_directions,
                attention_size=self.attention_size
            )
        
        # Dropout for regularization
        self.dropout = nn.Dropout(self.dropout_rate)
        
        # Task-specific prediction heads
        # Each head handles different prediction tasks
        self.mortality_head = nn.Sequential(
            nn.Linear(self.hidden_size * self.num_directions, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(128, 3)  # 3 mortality horizons
        )
        
        self.complication_head = nn.Sequential(
            nn.Linear(self.hidden_size * self.num_directions, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(128, 5)  # 5 complication types
        )
        
        self.icu_head = nn.Sequential(
            nn.Linear(self.hidden_size * self.num_directions, 64),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(64, 1)  # ICU admission probability
        )
        
        self.los_head = nn.Sequential(
            nn.Linear(self.hidden_size * self.num_directions, 64),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(64, 1)  # Length of stay (regression)
        )
        
        # Residual connection for the final representation
        self.residual_proj = nn.Linear(
            self.hidden_size * self.num_directions,
            self.hidden_size * self.num_directions
        )
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(f"LSTM Predictor initialized with config: {config}")
        logger.info(f"Total parameters: {sum(p.numel() for p in self.parameters())}")
    
    def _initialize_weights(self):
        """Initialize model weights using Xavier initialization."""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
        
        # Initialize LSTM weights specially
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                # Initialize forget gate bias to 1 for better gradient flow
                n = param.size(0)
                start, end = n // 4, n // 2
                param.data[start:end].fill_(1.0)
    
    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape [batch_size, sequence_length, input_size]
            hidden: Optional initial hidden state
            
        Returns:
            Output tensor of shape [batch_size, output_size] containing
            concatenated predictions from all task heads
        """
        batch_size, seq_len, _ = x.shape
        
        # Normalize input
        x = self.input_norm(x)
        
        # Apply dropout to input
        x = self.dropout(x)
        
        # LSTM forward pass
        lstm_out, (hidden_state, cell_state) = self.lstm(x, hidden)
        
        # Apply attention if enabled
        if self.use_attention:
            context, attention_weights = self.attention(lstm_out)
            final_repr = context
        else:
            # Use last hidden state
            if self.bidirectional:
                # Concatenate forward and backward last states
                h_forward = hidden_state[-2, :, :]  # Last forward layer
                h_backward = hidden_state[-1, :, :]  # Last backward layer
                final_repr = torch.cat([h_forward, h_backward], dim=1)
            else:
                final_repr = hidden_state[-1, :, :]
        
        # Apply residual connection
        residual = self.residual_proj(final_repr)
        final_repr = final_repr + 0.1 * residual  # Small residual contribution
        
        # Apply dropout to final representation
        final_repr = self.dropout(final_repr)
        
        # Generate predictions from each task head
        mortality_pred = self.mortality_head(final_repr)  # [batch, 3]
        complication_pred = self.complication_head(final_repr)  # [batch, 5]
        icu_pred = self.icu_head(final_repr)  # [batch, 1]
        los_pred = self.los_head(final_repr)  # [batch, 1]
        
        # Concatenate all predictions
        # Order: [mortality_24h, mortality_48h, mortality_168h,
        #         sepsis, cardiac_arrest, resp_failure, aki, bleeding,
        #         icu_admission, los]
        output = torch.cat([
            mortality_pred,
            complication_pred,
            icu_pred,
            los_pred
        ], dim=1)
        
        return output
    
    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get attention weights for interpretability.
        
        Args:
            x: Input tensor
            
        Returns:
            Attention weights over time steps
        """
        if not self.use_attention:
            return None
        
        batch_size, seq_len, _ = x.shape
        
        # Normalize input
        x = self.input_norm(x)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Get attention weights
        _, attention_weights = self.attention(lstm_out)
        
        return attention_weights
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self.config.copy()
    
    def save_pretrained(self, path: str):
        """Save model weights and configuration."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config
        }, path)
    
    @classmethod
    def from_pretrained(cls, path: str) -> 'LSTMPredictor':
        """Load model from pretrained weights."""
        checkpoint = torch.load(path)
        model = cls(checkpoint['config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        return model


class TemporalAttention(nn.Module):
    """
    Temporal attention mechanism for LSTM outputs.
    
    Computes attention weights over time steps to identify
    important moments in the patient's clinical timeline.
    """
    
    def __init__(self, hidden_size: int, attention_size: int = 128):
        """
        Initialize temporal attention.
        
        Args:
            hidden_size: Size of LSTM hidden states
            attention_size: Size of attention layer
        """
        super(TemporalAttention, self).__init__()
        
        self.hidden_size = hidden_size
        self.attention_size = attention_size
        
        # Attention mechanism
        self.W = nn.Linear(hidden_size, attention_size, bias=False)
        self.v = nn.Linear(attention_size, 1, bias=False)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.v.weight)
    
    def forward(self, lstm_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute attention-weighted context vector.
        
        Args:
            lstm_output: LSTM outputs of shape [batch_size, seq_len, hidden_size]
            
        Returns:
            - Context vector: [batch_size, hidden_size]
            - Attention weights: [batch_size, seq_len]
        """
        # Compute attention scores
        # [batch_size, seq_len, attention_size]
        attention_hidden = torch.tanh(self.W(lstm_output))
        
        # [batch_size, seq_len, 1]
        attention_scores = self.v(attention_hidden)
        
        # [batch_size, seq_len]
        attention_weights = F.softmax(attention_scores.squeeze(-1), dim=1)
        
        # Compute context vector as weighted sum
        # [batch_size, hidden_size]
        context = torch.sum(lstm_output * attention_weights.unsqueeze(-1), dim=1)
        
        return context, attention_weights


class LSTMCellWithSkip(nn.Module):
    """
    LSTM cell with skip connections for better gradient flow.
    Used internally for custom LSTM implementation.
    """
    
    def __init__(self, input_size: int, hidden_size: int):
        super(LSTMCellWithSkip, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Input gate
        self.W_ii = nn.Linear(input_size, hidden_size, bias=True)
        self.W_hi = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Forget gate
        self.W_if = nn.Linear(input_size, hidden_size, bias=True)
        self.W_hf = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Cell gate
        self.W_ig = nn.Linear(input_size, hidden_size, bias=True)
        self.W_hg = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Output gate
        self.W_io = nn.Linear(input_size, hidden_size, bias=True)
        self.W_ho = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Skip connection projection
        self.skip_proj = nn.Linear(input_size, hidden_size, bias=False)
        
    def forward(
        self,
        x: torch.Tensor,
        state: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through LSTM cell.
        
        Args:
            x: Input tensor [batch_size, input_size]
            state: Tuple of (hidden_state, cell_state)
            
        Returns:
            - Output: [batch_size, hidden_size]
            - New state: (new_hidden, new_cell)
        """
        h_prev, c_prev = state
        
        # Gates computation
        i = torch.sigmoid(self.W_ii(x) + self.W_hi(h_prev))
        f = torch.sigmoid(self.W_if(x) + self.W_hf(h_prev))
        g = torch.tanh(self.W_ig(x) + self.W_hg(h_prev))
        o = torch.sigmoid(self.W_io(x) + self.W_ho(h_prev))
        
        # Cell update
        c_new = f * c_prev + i * g
        
        # Hidden update with skip connection
        h_new = o * torch.tanh(c_new)
        
        # Add skip connection
        h_new = h_new + self.skip_proj(x)
        
        return h_new, (h_new, c_new)