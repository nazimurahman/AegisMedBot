"""
Research Agent Module for Medical Literature Analysis
====================================================

This module provides comprehensive research capabilities for medical professionals:
- Literature search across medical databases (PubMed, arXiv, internal repositories)
- Paper summarization with extractive and abstractive methods
- Citation management and reference extraction
- Research question answering
- Evidence synthesis from multiple sources

The module implements a hierarchical agent architecture where the ResearchAssistant
coordinates specialized sub-components for retrieval and summarization.

Author: AegisMedBot Team
Version: 1.0.0
"""

from typing import Dict, List, Any, Optional, Union
import logging

# Configure module-level logger
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Import main components for easy access
from .research_assistant import ResearchAssistant
from .paper_summarizer import PaperSummarizer, SummarizationMethod
from .literature_retriever import LiteratureRetriever, SearchSource, SearchResult

# Define module exports - what gets imported with "from research_agent import *"
__all__ = [
    'ResearchAssistant',
    'PaperSummarizer',
    'LiteratureRetriever',
    'SummarizationMethod',
    'SearchSource',
    'SearchResult',
    'create_research_agent'
]

# Module metadata
__version__ = '1.0.0'
__author__ = 'AegisMedBot'
__description__ = 'Medical research and literature analysis agent'

def create_research_agent(config: Optional[Dict[str, Any]] = None) -> ResearchAssistant:
    """
    Factory function to create a configured research agent.
    
    This provides a simplified interface for creating a fully initialized
    research agent with default or custom configuration.
    
    Args:
        config: Optional configuration dictionary with the following keys:
            - model_name: Transformer model for summarization
            - device: CPU or CUDA device
            - cache_dir: Directory for caching retrieved papers
            - max_papers: Maximum papers to retrieve per query
            - api_keys: Dictionary of API keys for external services
            
    Returns:
        Configured ResearchAssistant instance
        
    Example:
        >>> agent = create_research_agent({
        ...     'model_name': 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract',
        ...     'device': 'cuda',
        ...     'max_papers': 10
        ... })
        >>> results = await agent.research("Latest treatments for melanoma")
    """
    logger.info("Creating research agent with config: %s", config)
    return ResearchAssistant(config or {})

# Initialize module-level state if needed
def _initialize_module():
    """Internal initialization function."""
    logger.debug("Initializing research_agent module")
    # Perform any module-level setup here
    # For example, check for required dependencies
    try:
        import torch
        logger.info("PyTorch version: %s available", torch.__version__)
    except ImportError as e:
        logger.warning("PyTorch not available: %s", str(e))
    
    try:
        import transformers
        logger.info("Transformers version: %s available", transformers.__version__)
    except ImportError as e:
        logger.warning("Transformers not available: %s", str(e))

# Call initialization when module loads
_initialize_module()