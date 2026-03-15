"""
Operations Agent Module for AegisMedBot Hospital Intelligence Platform.
This module provides real-time hospital operational intelligence including
bed occupancy analysis, patient flow prediction, and resource optimization.
"""

from .operations_agent import OperationsAgent
from .bed_analyzer import BedAnalyzer
from .flow_predictor import FlowPredictor

__all__ = [
    'OperationsAgent',
    'BedAnalyzer', 
    'FlowPredictor'
]

__version__ = '1.0.0'
__author__ = 'AegisMedBot Team'