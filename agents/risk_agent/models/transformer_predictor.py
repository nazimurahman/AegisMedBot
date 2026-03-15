"""
Transformer-based prediction model for patient risk assessment.

This module implements a transformer architecture with:
- Multi-head self-attention for temporal dependencies
- Positional encoding for sequence order
- Layer normalization and residual connections
- Task-specific output heads
- Interpretable attention weights
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, Any, Optional, Tuple, List
import copy
import logging

logger = logging.getLogger(__name__)

class TransformerPredictor(nn.Module):
    """
    Transformer-based model for patient risk prediction.
    
    Architecture:
    - Input embedding with linear projection
    - Positional encoding
    - Multiple transformer encoder layers
    - Multi-head self-attention
    - Feed-forward networks
    - Layer normalization
    - Task-specific output heads
    
    Captures long-range dependencies in patient timelines
    and provides attention-based interpretability.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize transformer predictor.
        
        Args:
            config: Configuration dictionary with parameters:
                - input_dim: Number of input features
                - d_model: Model dimension (default: 512)
                - nhead: Number of attention heads (default: 8)
                - num_layers: Number of transformer layers (default: 6)
                - dim_feedforward: FFN dimension (default: 2048)
                - output_dim: Number of output predictions (default: 15)
                - dropout: Dropout rate (default: 0.1)
                - activation: Activation function ('relu' or 'gelu')
                - max_seq_length: Maximum sequence length (default: 512)
        """
        super(TransformerPredictor, self).__init__()
        
        # Store configuration
        self.config = config
        self.input_dim = config.get("input_dim", 128)
        self.d_model = config.get("d_model", 512)
        self.nhead = config.get("nhead", 8)
        self.num_layers = config.get("num_layers", 6)
        self.dim_feedforward = config.get("dim_feedforward", 2048)
        self.output_dim = config.get("output_dim", 15)
        self.dropout_rate = config.get("dropout", 0.1)
        self.activation = config.get("activation", "gelu")
        self.max_seq_length = config.get("max_seq_length", 512)
        
        # Validate dimensions
        assert self.d_model % self.nhead == 0, \
            f"d_model ({self.d_model}) must be divisible by nhead ({self.nhead})"
        
        # Input projection
        self.input_projection = nn.Linear(self.input_dim, self.d_model)
        self.input_norm = nn.LayerNorm(self.d_model)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            d_model=self.d_model,
            max_len=self.max_seq_length,
            dropout=self.dropout_rate
        )
        
        # Transformer encoder layers
        encoder_layer = TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout_rate,
            activation=self.activation
        )
        
        self.transformer_encoder = nn.ModuleList([
            copy.deepcopy(encoder_layer) for _ in range(self.num_layers)
        ])
        
        # Output aggregation
        self.output_aggregation = MultiHeadAggregation(
            d_model=self.d_model,
            nhead=self.nhead,
            dropout=self.dropout_rate
        )
        
        # Task-specific output heads (same as LSTM for consistency)
        self.mortality_head = nn.Sequential(
            nn.Linear(self.d_model, 128),
            nn.GELU() if self.activation == "gelu" else nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(128, 3)  # 3 mortality horizons
        )
        
        self.complication_head = nn.Sequential(
            nn.Linear(self.d_model, 128),
            nn.GELU() if self.activation == "gelu" else nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(128, 5)  # 5 complication types
        )
        
        self.icu_head = nn.Sequential(
            nn.Linear(self.d_model, 64),
            nn.GELU() if self.activation == "gelu" else nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(64, 1)  # ICU admission probability
        )
        
        self.los_head = nn.Sequential(
            nn.Linear(self.d_model, 64),
            nn.GELU() if self.activation == "gelu" else nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(64, 1)  # Length of stay (regression)
        )
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(f"Transformer Predictor initialized with config: {config}")
        logger.info(f"Total parameters: {sum(p.numel() for p in self.parameters())}")
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through transformer.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            mask: Optional attention mask [batch_size, seq_len]
                  True values indicate positions to mask
            
        Returns:
            Output tensor [batch_size, output_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # Input projection
        x = self.input_projection(x)  # [batch, seq_len, d_model]
        x = self.input_norm(x)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Prepare attention mask
        if mask is not None:
            # Convert boolean mask to attention mask
            # True values -> -inf (mask out), False -> 0
            attention_mask = mask.float().masked_fill(mask, float('-inf'))
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]
        else:
            attention_mask = None
        
        # Apply transformer layers
        attention_weights = []
        for layer in self.transformer_encoder:
            x, layer_attention = layer(x, attention_mask)
            attention_weights.append(layer_attention)
        
        # Aggregate outputs across sequence
        context = self.output_aggregation(x)  # [batch, d_model]
        
        # Generate predictions from each task head
        mortality_pred = self.mortality_head(context)  # [batch, 3]
        complication_pred = self.complication_head(context)  # [batch, 5]
        icu_pred = self.icu_head(context)  # [batch, 1]
        los_pred = self.los_head(context)  # [batch, 1]
        
        # Concatenate all predictions
        output = torch.cat([
            mortality_pred,
            complication_pred,
            icu_pred,
            los_pred
        ], dim=1)
        
        return output
    
    def get_attention_weights(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Get attention weights from all layers for interpretability.
        
        Args:
            x: Input tensor
            
        Returns:
            List of attention weight tensors from each layer
        """
        batch_size, seq_len, _ = x.shape
        
        # Input projection
        x = self.input_projection(x)
        x = self.input_norm(x)
        x = self.positional_encoding(x)
        
        # Collect attention weights
        attention_weights = []
        for layer in self.transformer_encoder:
            x, layer_attention = layer(x, return_attention=True)
            attention_weights.append(layer_attention)
        
        return attention_weights
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self.config.copy()


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for transformer.
    
    Adds information about position in sequence using
    sine and cosine functions of different frequencies.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Initialize positional encoding.
        
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
            dropout: Dropout rate
        """
        super(PositionalEncoding, self).__init__()
        
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Compute div_term using exponential for numerical stability
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model)
        )
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            Output with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerEncoderLayer(nn.Module):
    """
    Single transformer encoder layer.
    
    Consists of:
    - Multi-head self-attention
    - Feed-forward network
    - Layer normalization
    - Residual connections
    """
    
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu"
    ):
        """
        Initialize encoder layer.
        
        Args:
            d_model: Model dimension
            nhead: Number of attention heads
            dim_feedforward: FFN dimension
            dropout: Dropout rate
            activation: Activation function
        """
        super(TransformerEncoderLayer, self).__init__()
        
        # Multi-head attention
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        
        # Feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        # Activation
        self.activation = getattr(F, activation)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through encoder layer.
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
            mask: Attention mask
            return_attention: Whether to return attention weights
            
        Returns:
            - Output tensor
            - Attention weights (if return_attention=True)
        """
        # Self-attention with residual
        attn_out, attn_weights = self.self_attn(x, x, x, mask)
        x = x + self.dropout1(attn_out)
        x = self.norm1(x)
        
        # Feed-forward with residual
        ff_out = self.linear2(self.dropout2(self.activation(self.linear1(x))))
        x = x + self.dropout3(ff_out)
        x = self.norm2(x)
        
        if return_attention:
            return x, attn_weights
        return x, None


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism.
    
    Computes attention across multiple heads and
    concatenates results for richer representations.
    """
    
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        """
        Initialize multi-head attention.
        
        Args:
            d_model: Model dimension
            nhead: Number of attention heads
            dropout: Dropout rate
        """
        super(MultiHeadAttention, self).__init__()
        
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scale = self.head_dim ** -0.5
        
        # Linear projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute multi-head attention.
        
        Args:
            query: Query tensor [batch, q_len, d_model]
            key: Key tensor [batch, k_len, d_model]
            value: Value tensor [batch, v_len, d_model]
            mask: Attention mask [batch, 1, 1, k_len] or [batch, 1, q_len, k_len]
            
        Returns:
            - Output tensor [batch, q_len, d_model]
            - Attention weights [batch, nhead, q_len, k_len]
        """
        batch_size = query.size(0)
        
        # Linear projections and reshape
        q = self.q_proj(query).view(batch_size, -1, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, -1, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, -1, self.nhead, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [batch, nhead, q_len, k_len]
        
        # Apply mask
        if mask is not None:
            scores = scores + mask
        
        # Attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, v)  # [batch, nhead, q_len, head_dim]
        
        # Reshape and project
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        output = self.out_proj(context)
        
        return output, attn_weights


class MultiHeadAggregation(nn.Module):
    """
    Multi-head aggregation of transformer outputs.
    
    Combines information across sequence using multiple
    attention heads for robust feature aggregation.
    """
    
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        """
        Initialize aggregation module.
        
        Args:
            d_model: Model dimension
            nhead: Number of attention heads
            dropout: Dropout rate
        """
        super(MultiHeadAggregation, self).__init__()
        
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        
        # Learnable query for attention pooling
        self.query = nn.Parameter(torch.randn(1, nhead, 1, self.head_dim))
        
        # Key projection
        self.k_proj = nn.Linear(d_model, d_model)
        
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize query
        nn.init.xavier_uniform_(self.query)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Aggregate sequence using attention pooling.
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
            
        Returns:
            Aggregated context [batch, d_model]
        """
        batch_size, seq_len, _ = x.shape
        
        # Project keys
        k = self.k_proj(x).view(batch_size, seq_len, self.nhead, self.head_dim)
        k = k.transpose(1, 2)  # [batch, nhead, seq_len, head_dim]
        
        # Expand query to batch
        q = self.query.expand(batch_size, -1, -1, -1)  # [batch, nhead, 1, head_dim]
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.squeeze(2)  # [batch, nhead, seq_len]
        
        # Attention weights
        attn_weights = F.softmax(scores, dim=-1)  # [batch, nhead, seq_len]
        attn_weights = self.dropout(attn_weights)
        
        # Weighted sum
        context = torch.sum(
            attn_weights.unsqueeze(-1) * x.unsqueeze(1),
            dim=2
        )  # [batch, nhead, d_model]
        
        # Combine heads
        context = context.mean(dim=1)  # [batch, d_model]
        
        # Output projection
        output = self.out_proj(context)
        
        return output