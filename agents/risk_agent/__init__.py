"""
Risk Agent Module for AegisMedBot Hospital Intelligence Platform.

This module provides patient risk assessment capabilities including:
- Mortality prediction
- Complication risk assessment
- ICU admission prediction
- Length of stay estimation
- Real-time risk monitoring

The risk agent integrates with clinical data sources and uses
advanced ML models to provide actionable risk insights.
"""

from agents.risk_agent.risk_predictor import RiskPredictor
from agents.risk_agent.models.lstm_predictor import LSTMPredictor
from agents.risk_agent.models.transformer_predictor import TransformerPredictor
from agents.risk_agent.features import ClinicalFeatureExtractor

__version__ = "1.0.0"
__all__ = [
    "RiskPredictor",
    "LSTMPredictor", 
    "TransformerPredictor",
    "ClinicalFeatureExtractor"
]

# Package metadata
__author__ = "AegisMedBot Team"
__description__ = "Advanced patient risk prediction module for hospital intelligence"