"""
Patient LSTM Model
Implements LSTM-based models for patient trajectory prediction and clinical time series analysis.
Handles sequential patient data including vitals, lab results, and clinical events.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any
import math
import logging

logger = logging.getLogger(__name__)

class PatientLSTM(nn.Module):
    """
    LSTM model for patient trajectory prediction.
    Processes sequential clinical data and predicts patient outcomes.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        output_dim: int = 1,
        dropout: float = 0.2,
        bidirectional: bool = True,
        use_attention: bool = True
    ):
        """
        Initialize Patient LSTM model.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden state dimension
            num_layers: Number of LSTM layers
            output_dim: Output dimension (number of predictions)
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional LSTM
            use_attention: Whether to use attention mechanism
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        
        # Determine direction multiplier
        self.direction_multiplier = 2 if bidirectional else 1
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Attention mechanism
        if use_attention:
            self.attention = PatientAttention(hidden_dim * self.direction_multiplier)
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_dim * self.direction_multiplier, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(hidden_dim * self.direction_multiplier)
        
        logger.info(f"Initialized PatientLSTM with hidden_dim={hidden_dim}, "
                   f"num_layers={num_layers}, bidirectional={bidirectional}")
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through LSTM.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            mask: Optional mask for padding (batch, seq_len)
            hidden: Optional initial hidden state
            
        Returns:
            Tuple of (predictions, last_hidden_state)
        """
        batch_size = x.size(0)
        
        # Apply mask if provided (set masked positions to zero)
        if mask is not None:
            x = x * mask.unsqueeze(-1).float()
        
        # Pass through LSTM
        lstm_out, (hidden_state, cell_state) = self.lstm(x, hidden)
        
        # Apply attention if enabled
        if self.use_attention:
            # Compute attention weights
            attention_weights = self.attention(lstm_out, mask)
            # Apply attention to LSTM outputs
            context_vector = torch.sum(lstm_out * attention_weights.unsqueeze(-1), dim=1)
            # Combine with LSTM outputs
            last_output = context_vector
        else:
            # Use last hidden state
            if self.bidirectional:
                # Concatenate forward and backward last hidden states
                last_output = torch.cat([hidden_state[-2], hidden_state[-1]], dim=1)
            else:
                last_output = hidden_state[-1]
        
        # Apply layer normalization
        last_output = self.layer_norm(last_output)
        
        # Apply dropout
        last_output = self.dropout(last_output)
        
        # Generate predictions
        predictions = self.output_layers(last_output)
        
        return predictions, hidden_state
    
    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get attention weights for interpretability.
        
        Args:
            x: Input tensor
            
        Returns:
            Attention weights tensor
        """
        if not self.use_attention:
            raise ValueError("Attention is not enabled in this model")
        
        lstm_out, _ = self.lstm(x)
        attention_weights = self.attention(lstm_out)
        return attention_weights


class PatientAttention(nn.Module):
    """
    Attention mechanism for patient time series data.
    Allows model to focus on important time steps.
    """
    
    def __init__(self, hidden_dim: int):
        """
        Initialize attention mechanism.
        
        Args:
            hidden_dim: Hidden dimension of LSTM outputs
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Attention parameters
        self.attention_weights = nn.Linear(hidden_dim, 1)
        self.temperature = nn.Parameter(torch.ones(1))
        
    def forward(
        self,
        lstm_outputs: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute attention weights.
        
        Args:
            lstm_outputs: LSTM outputs of shape (batch, seq_len, hidden_dim)
            mask: Optional mask for padding
            
        Returns:
            Attention weights of shape (batch, seq_len)
        """
        # Compute attention scores
        scores = self.attention_weights(lstm_outputs).squeeze(-1)
        
        # Apply temperature scaling
        scores = scores / (self.temperature.abs() + 1e-8)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get weights
        attention_weights = F.softmax(scores, dim=1)
        
        return attention_weights


class HierarchicalPatientLSTM(nn.Module):
    """
    Hierarchical LSTM for patient data with multiple time scales.
    Processes daily, weekly, and monthly patterns in patient trajectories.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_hierarchical_levels: int = 3,
        dropout: float = 0.2
    ):
        """
        Initialize hierarchical patient LSTM.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension for each level
            num_hierarchical_levels: Number of hierarchical levels
            dropout: Dropout probability
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_levels = num_hierarchical_levels
        self.dropout = dropout
        
        # Encoder for each hierarchical level
        self.encoders = nn.ModuleList()
        for level in range(num_hierarchical_levels):
            encoder = nn.LSTM(
                input_size=input_dim if level == 0 else hidden_dim,
                hidden_size=hidden_dim,
                num_layers=1,
                batch_first=True,
                dropout=dropout
            )
            self.encoders.append(encoder)
        
        # Aggregation layers
        self.aggregators = nn.ModuleList()
        for level in range(num_hierarchical_levels - 1):
            aggregator = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self.aggregators.append(aggregator)
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, 1)
        
        logger.info(f"Initialized HierarchicalPatientLSTM with {num_hierarchical_levels} levels")
    
    def forward(
        self,
        x: torch.Tensor,
        time_scales: Optional[List[int]] = None
    ) -> torch.Tensor:
        """
        Forward pass through hierarchical LSTM.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            time_scales: List of time scales for each level
            
        Returns:
            Predictions tensor
        """
        batch_size = x.size(0)
        current_input = x
        
        # Process each hierarchical level
        level_outputs = []
        
        for level_idx, encoder in enumerate(self.encoders):
            # Encode at this level
            encoded, _ = encoder(current_input)
            
            # Aggregate if not the last level
            if level_idx < self.num_levels - 1:
                # Apply aggregation
                aggregated = self.aggregators[level_idx](encoded[:, -1, :])
                level_outputs.append(aggregated)
                
                # Prepare input for next level
                if time_scales and level_idx < len(time_scales):
                    # Downsample based on time scale
                    scale = time_scales[level_idx]
                    if scale > 1:
                        # Average pooling for downsampling
                        seq_len = current_input.size(1)
                        num_pools = seq_len // scale
                        if num_pools > 0:
                            pooled = current_input[:, :num_pools * scale, :]
                            pooled = pooled.view(batch_size, num_pools, scale, -1)
                            current_input = pooled.mean(dim=2)
                        else:
                            current_input = current_input[:, :1, :]
                    else:
                        current_input = encoded[:, -1:, :]
                else:
                    current_input = encoded[:, -1:, :]
            else:
                # Last level - use full sequence
                level_outputs.append(encoded)
        
        # Combine all level outputs
        combined = torch.cat(level_outputs, dim=1)
        
        # Final prediction
        predictions = self.output_layer(combined.mean(dim=1))
        
        return predictions


class ClinicalEventLSTM(nn.Module):
    """
    LSTM model for predicting clinical events (ICU admission, readmission, etc.).
    Incorporates both time series data and static patient features.
    """
    
    def __init__(
        self,
        time_series_dim: int,
        static_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
        num_events: int = 5
    ):
        """
        Initialize clinical event prediction LSTM.
        
        Args:
            time_series_dim: Dimension of time series features
            static_dim: Dimension of static patient features
            hidden_dim: Hidden dimension for LSTM
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            num_events: Number of possible events to predict
        """
        super().__init__()
        
        self.time_series_dim = time_series_dim
        self.static_dim = static_dim
        self.hidden_dim = hidden_dim
        self.num_events = num_events
        
        # Static feature encoder
        self.static_encoder = nn.Sequential(
            nn.Linear(static_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Time series LSTM
        self.lstm = nn.LSTM(
            input_size=time_series_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Event prediction heads
        self.event_heads = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(num_events)
        ])
        
        # Time-to-event prediction
        self.time_to_event = nn.Linear(hidden_dim, 1)
        
        logger.info(f"Initialized ClinicalEventLSTM with {num_events} event heads")
    
    def forward(
        self,
        time_series: torch.Tensor,
        static_features: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for event prediction.
        
        Args:
            time_series: Time series data of shape (batch, seq_len, time_series_dim)
            static_features: Static features of shape (batch, static_dim)
            mask: Optional mask for padding
            
        Returns:
            Dictionary with event probabilities and time predictions
        """
        # Encode static features
        static_encoded = self.static_encoder(static_features)
        
        # Process time series with LSTM
        if mask is not None:
            time_series = time_series * mask.unsqueeze(-1).float()
        
        lstm_out, (hidden, cell) = self.lstm(time_series)
        
        # Use last hidden state
        time_series_encoded = hidden[-1]  # Use last layer
        
        # Combine static and time series features
        combined = torch.cat([time_series_encoded, static_encoded], dim=1)
        fused = self.fusion(combined)
        
        # Predict events
        event_logits = []
        for head in self.event_heads:
            event_logits.append(head(fused))
        
        event_probs = torch.cat([torch.sigmoid(logit) for logit in event_logits], dim=1)
        
        # Predict time to event
        time_to_event = F.softplus(self.time_to_event(fused))
        
        return {
            'event_probabilities': event_probs,
            'time_to_event': time_to_event.squeeze(-1)
        }


class PatientTrajectoryDecoder(nn.Module):
    """
    Decoder for generating patient trajectory sequences.
    Used for forecasting future patient states.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        """
        Initialize patient trajectory decoder.
        
        Args:
            hidden_dim: Hidden dimension from encoder
            output_dim: Output feature dimension
            num_layers: Number of LSTM layers
            dropout: Dropout probability
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # LSTM decoder
        self.lstm = nn.LSTM(
            input_size=output_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, output_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        encoder_hidden: torch.Tensor,
        target_length: int,
        teacher_forcing: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 0.5
    ) -> torch.Tensor:
        """
        Generate future trajectory.
        
        Args:
            encoder_hidden: Hidden state from encoder
            target_length: Number of time steps to generate
            teacher_forcing: Optional ground truth for teacher forcing
            teacher_forcing_ratio: Probability of using teacher forcing
            
        Returns:
            Generated sequence of shape (batch, target_length, output_dim)
        """
        batch_size = encoder_hidden.size(0)
        predictions = []
        
        # Initialize decoder input with zeros
        decoder_input = torch.zeros(batch_size, 1, self.output_dim).to(encoder_hidden.device)
        hidden = (encoder_hidden.unsqueeze(0), torch.zeros_like(encoder_hidden).unsqueeze(0))
        
        for t in range(target_length):
            # Forward pass through decoder
            decoder_out, hidden = self.lstm(decoder_input, hidden)
            decoder_out = self.dropout(decoder_out)
            
            # Generate prediction
            prediction = self.output_projection(decoder_out)
            predictions.append(prediction)
            
            # Determine next input
            if teacher_forcing is not None and t < teacher_forcing.size(1):
                if torch.rand(1).item() < teacher_forcing_ratio:
                    decoder_input = teacher_forcing[:, t:t+1, :]
                else:
                    decoder_input = prediction
            else:
                decoder_input = prediction
        
        # Concatenate predictions
        output = torch.cat(predictions, dim=1)
        
        return output


# Model registry for LSTM models
class LSTMModelRegistry:
    """
    Registry for LSTM models with easy loading functionality.
    """
    
    _models = {}
    
    @classmethod
    def register(cls, name: str, model_class):
        """Register a model class."""
        cls._models[name] = model_class
    
    @classmethod
    def get_model(cls, name: str, **kwargs) -> nn.Module:
        """
        Get a model by name.
        
        Args:
            name: Model name
            **kwargs: Additional arguments for model initialization
            
        Returns:
            Model instance
        """
        if name not in cls._models:
            raise ValueError(f"Model '{name}' not found in registry")
        
        return cls._models[name](**kwargs)
    
    @classmethod
    def list_models(cls) -> List[str]:
        """List available models."""
        return list(cls._models.keys())


# Register models
LSTMModelRegistry.register('patient_lstm', PatientLSTM)
LSTMModelRegistry.register('hierarchical_lstm', HierarchicalPatientLSTM)
LSTMModelRegistry.register('clinical_event_lstm', ClinicalEventLSTM)
LSTMModelRegistry.register('trajectory_decoder', PatientTrajectoryDecoder)


# Example usage
if __name__ == "__main__":
    # Test PatientLSTM
    model = PatientLSTM(
        input_dim=10,
        hidden_dim=128,
        output_dim=2,
        use_attention=True
    )
    
    # Create dummy input
    x = torch.randn(4, 50, 10)  # batch=4, seq_len=50, features=10
    mask = torch.ones(4, 50)
    
    # Forward pass
    predictions, hidden = model(x, mask)
    print(f"PatientLSTM predictions shape: {predictions.shape}")
    
    # Test ClinicalEventLSTM
    event_model = ClinicalEventLSTM(
        time_series_dim=10,
        static_dim=5,
        hidden_dim=128,
        num_events=3
    )
    
    static = torch.randn(4, 5)
    outputs = event_model(x, static, mask)
    print(f"Event probabilities shape: {outputs['event_probabilities'].shape}")
    print(f"Time to event shape: {outputs['time_to_event'].shape}")
    
    # Test HierarchicalPatientLSTM
    hierarchical_model = HierarchicalPatientLSTM(
        input_dim=10,
        hidden_dim=64,
        num_hierarchical_levels=3
    )
    
    hierarchical_output = hierarchical_model(x)
    print(f"Hierarchical model output shape: {hierarchical_output.shape}")
    
    # List available models
    print(f"Available LSTM models: {LSTMModelRegistry.list_models()}")