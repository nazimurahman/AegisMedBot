"""
Medical Transformer Model
Implements transformer-based models for medical NLP tasks including classification,
question answering, and sequence labeling with medical domain adaptations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List, Union
from transformers import (
    BertModel,
    BertConfig,
    BertPreTrainedModel,
    AutoModel,
    AutoConfig,
    PreTrainedModel
)
import math
import logging

logger = logging.getLogger(__name__)

class MedicalBertModel(BertPreTrainedModel):
    """
    Bert-based model with medical domain adaptations.
    Supports classification, token classification, and sequence labeling tasks.
    """
    
    def __init__(
        self,
        config: BertConfig,
        num_labels: Optional[int] = None,
        use_clinical_embeddings: bool = True,
        use_medical_attention: bool = True
    ):
        """
        Initialize medical BERT model.
        
        Args:
            config: BERT configuration
            num_labels: Number of output labels for classification
            use_clinical_embeddings: Whether to use clinical-specific embeddings
            use_medical_attention: Whether to use medical attention mechanism
        """
        super().__init__(config)
        self.config = config
        self.num_labels = num_labels or config.num_labels
        self.use_clinical_embeddings = use_clinical_embeddings
        self.use_medical_attention = use_medical_attention
        
        # Base BERT model
        self.bert = BertModel(config)
        
        # Clinical embedding layer (adds medical term embeddings)
        if use_clinical_embeddings:
            self.clinical_embedding = nn.Embedding(
                num_embeddings=10000,  # Medical term vocabulary size
                embedding_dim=config.hidden_size
            )
            self.clinical_dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Medical attention mechanism
        if use_medical_attention:
            self.medical_attention = MedicalAttentionLayer(config)
        
        # Task-specific layers
        if num_labels is not None:
            self.classifier = nn.Linear(config.hidden_size, num_labels)
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Initialize weights
        self.init_weights()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        clinical_tokens: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None
    ) -> Union[Tuple[torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Token indices for input text
            attention_mask: Mask for padding tokens
            token_type_ids: Segment IDs for token types
            position_ids: Position indices
            head_mask: Mask for attention heads
            inputs_embeds: Alternative to input_ids (precomputed embeddings)
            labels: Target labels for classification
            clinical_tokens: Clinical token identifiers for specialized embeddings
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return hidden states
            return_dict: Whether to return as dictionary
            
        Returns:
            Model outputs including loss, logits, and hidden states
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Get BERT outputs
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True
        )
        
        # Extract sequence output and pooled output
        sequence_output = outputs.last_hidden_state
        pooled_output = outputs.pooler_output
        
        # Add clinical embeddings if enabled
        if self.use_clinical_embeddings and clinical_tokens is not None:
            clinical_embeds = self.clinical_embedding(clinical_tokens)
            clinical_embeds = self.clinical_dropout(clinical_embeds)
            sequence_output = sequence_output + clinical_embeds
        
        # Apply medical attention if enabled
        if self.use_medical_attention:
            medical_output = self.medical_attention(
                sequence_output,
                attention_mask=attention_mask
            )
            sequence_output = sequence_output + medical_output
        
        # Apply classification head
        logits = None
        loss = None
        
        if self.num_labels is not None:
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
            
            # Calculate loss if labels provided
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        
        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': outputs.hidden_states,
            'attentions': outputs.attentions,
            'sequence_output': sequence_output,
            'pooled_output': pooled_output
        }


class MedicalAttentionLayer(nn.Module):
    """
    Medical-specific attention layer that incorporates clinical knowledge.
    Uses domain-specific attention patterns for medical text.
    """
    
    def __init__(self, config: BertConfig):
        """
        Initialize medical attention layer.
        
        Args:
            config: BERT configuration
        """
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        # Query, Key, Value projections
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        
        # Medical term bias (weights for medical terms)
        self.medical_bias = nn.Parameter(torch.zeros(self.num_attention_heads, 1, 1))
        
        # Output projection
        self.output = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transpose tensor for multi-head attention.
        
        Args:
            x: Input tensor of shape (batch, seq_len, hidden_size)
            
        Returns:
            Transposed tensor of shape (batch, num_heads, seq_len, head_size)
        """
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply medical attention to hidden states.
        
        Args:
            hidden_states: Input hidden states
            attention_mask: Mask for attention scores
            
        Returns:
            Attention output
        """
        # Prepare for attention
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        # Compute attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # Apply medical bias (emphasizes medical terms)
        attention_scores = attention_scores + self.medical_bias
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        # Softmax and dropout
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        
        # Apply attention to values
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        # Output projection
        attention_output = self.output(context_layer)
        attention_output = self.dropout(attention_output)
        
        # Residual connection with layer norm
        output = self.layer_norm(hidden_states + attention_output)
        
        return output


class MedicalClinicalBert(MedicalBertModel):
    """
    Medical BERT model specialized for clinical tasks.
    Includes additional features for clinical text processing.
    """
    
    def __init__(self, config: BertConfig, num_labels: int = None):
        """
        Initialize clinical BERT model.
        
        Args:
            config: BERT configuration
            num_labels: Number of output labels
        """
        super().__init__(config, num_labels, use_clinical_embeddings=True, use_medical_attention=True)
        
        # Add clinical token type embeddings
        self.clinical_token_type_embeddings = nn.Embedding(3, config.hidden_size)
        
        # Add temperature embedding for time-series data
        self.temperature_embedding = nn.Linear(1, config.hidden_size)
        
        logger.info("Initialized MedicalClinicalBert model")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        clinical_token_ids: Optional[torch.Tensor] = None,
        temperature_values: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with clinical-specific features.
        
        Args:
            input_ids: Token indices
            attention_mask: Attention mask
            token_type_ids: Token type IDs
            clinical_token_ids: Clinical token identifiers
            temperature_values: Temperature readings for time-series
            
        Returns:
            Model outputs
        """
        # Get base embeddings
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
            return_dict=True
        )
        
        embeddings = outputs.last_hidden_state
        
        # Add clinical token type embeddings if provided
        if clinical_token_ids is not None:
            clinical_embeds = self.clinical_token_type_embeddings(clinical_token_ids)
            embeddings = embeddings + clinical_embeds
        
        # Add temperature embeddings if provided
        if temperature_values is not None:
            temp_embeds = self.temperature_embedding(temperature_values.unsqueeze(-1))
            embeddings = embeddings + temp_embeds
        
        # Apply medical attention
        medical_output = self.medical_attention(embeddings, attention_mask=attention_mask)
        embeddings = embeddings + medical_output
        
        # Pool and classify
        pooled = embeddings[:, 0, :]  # Use CLS token
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled) if self.num_labels else None
        
        # Calculate loss if labels provided
        loss = None
        labels = kwargs.get('labels')
        if labels is not None and logits is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': outputs.hidden_states,
            'embeddings': embeddings
        }


class MedicalSequenceModel(nn.Module):
    """
    Transformer-based model for medical sequence tasks (time series, patient trajectories).
    Combines transformer with LSTM for temporal medical data.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        num_classes: int = 2
    ):
        """
        Initialize medical sequence model.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            dropout: Dropout probability
            num_classes: Number of output classes
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding for sequence
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        
        # Output layers
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for sequence data.
        
        Args:
            x: Input sequence of shape (batch, seq_len, input_dim)
            mask: Optional padding mask
            
        Returns:
            Output logits of shape (batch, num_classes)
        """
        # Project to hidden dimension
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply transformer
        if mask is not None:
            x = self.transformer(x, src_key_padding_mask=mask)
        else:
            x = self.transformer(x)
        
        # Apply LSTM
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use last hidden state (concatenate forward and backward)
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        
        # Apply dropout
        hidden = self.dropout(hidden)
        
        # Output projection
        logits = self.output_projection(hidden)
        
        return logits


class PositionalEncoding(nn.Module):
    """
    Positional encoding for sequence models.
    Adds information about position in sequence.
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """
        Initialize positional encoding.
        
        Args:
            d_model: Model dimension
            dropout: Dropout probability
            max_len: Maximum sequence length
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor of shape (seq_len, batch, d_model)
            
        Returns:
            Output with positional encoding added
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class MedicalMultiTaskModel(nn.Module):
    """
    Multi-task medical model that can perform multiple tasks simultaneously:
    - Classification
    - Sequence labeling
    - Question answering
    """
    
    def __init__(
        self,
        base_model: PreTrainedModel,
        num_classes: int,
        num_labels: int,
        num_qa_tokens: int
    ):
        """
        Initialize multi-task medical model.
        
        Args:
            base_model: Pre-trained transformer model
            num_classes: Number of classification classes
            num_labels: Number of sequence labeling labels
            num_qa_tokens: Number of QA answer tokens
        """
        super().__init__()
        
        self.base_model = base_model
        hidden_size = base_model.config.hidden_size
        
        # Task-specific heads
        self.classification_head = nn.Linear(hidden_size, num_classes)
        self.sequence_labeling_head = nn.Linear(hidden_size, num_labels)
        self.qa_head = nn.Linear(hidden_size, num_qa_tokens)
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        task_type: str,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for multi-task learning.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            task_type: Type of task ('classification', 'labeling', 'qa')
            labels: Optional labels for loss calculation
            
        Returns:
            Dictionary with outputs for the specified task
        """
        # Get base model outputs
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        last_hidden = outputs.last_hidden_state
        pooled = outputs.pooler_output
        
        # Apply task-specific head
        if task_type == 'classification':
            logits = self.classification_head(self.dropout(pooled))
        elif task_type == 'labeling':
            logits = self.sequence_labeling_head(last_hidden)
        elif task_type == 'qa':
            logits = self.qa_head(last_hidden)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            if task_type == 'classification':
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            elif task_type == 'labeling':
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            elif task_type == 'qa':
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels.float())
        
        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': outputs.hidden_states
        }


# Model registry for easy loading
class MedicalModelRegistry:
    """
    Registry for medical models with easy loading functionality.
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
MedicalModelRegistry.register('medical_bert', MedicalBertModel)
MedicalModelRegistry.register('clinical_bert', MedicalClinicalBert)
MedicalModelRegistry.register('medical_sequence', MedicalSequenceModel)
MedicalModelRegistry.register('multi_task', MedicalMultiTaskModel)


# Example usage
if __name__ == "__main__":
    # Test MedicalBertModel
    config = BertConfig.from_pretrained('bert-base-uncased')
    config.num_labels = 5
    model = MedicalBertModel(config, num_labels=5)
    
    # Create dummy input
    input_ids = torch.randint(0, 30000, (2, 128))
    attention_mask = torch.ones_like(input_ids)
    labels = torch.randint(0, 5, (2,))
    
    # Forward pass
    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    print(f"Model outputs: {outputs.keys() if isinstance(outputs, dict) else type(outputs)}")
    
    # Test MedicalSequenceModel
    seq_model = MedicalSequenceModel(input_dim=10, hidden_dim=128, num_classes=3)
    seq_input = torch.randn(4, 50, 10)  # batch=4, seq_len=50, features=10
    seq_output = seq_model(seq_input)
    print(f"Sequence model output shape: {seq_output.shape}")
    
    # List available models
    print(f"Available models: {MedicalModelRegistry.list_models()}")