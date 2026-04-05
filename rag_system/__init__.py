"""
RAG System Module for AegisMedBot - Medical Intelligence Platform

This module provides the core Retrieval-Augmented Generation (RAG) capabilities
for the AegisMedBot hospital intelligence system. It handles the retrieval
of medical knowledge from various sources including literature, clinical
guidelines, and hospital policies to augment the AI agents' responses with
accurate, up-to-date medical information.

The RAG system is built on a multi-tier architecture that includes:
1. Vector storage for embeddings (Qdrant)
2. Multiple specialized retrievers for different medical content types
3. Document indexing and processing pipelines
4. Hybrid search combining semantic and keyword-based retrieval
5. Medical text preprocessing optimized for clinical terminology

Key Features:
- Semantic search over medical literature using transformer embeddings
- Clinical guidelines retrieval with specialized preprocessing
- Hybrid search combining vector similarity and keyword matching
- Support for multiple document formats and medical data sources
- Optimized for medical terminology and clinical language

Author: AegisMedBot Team
Version: 1.0.0
"""

# Import core modules to expose them at package level
# This allows users to import directly from rag_system
from .vector_store.embeddings import MedicalEmbeddingGenerator
from .vector_store.qdrant_manager import QdrantManager
from .vector_store.schema import DocumentSchema, VectorMetadata

from .retrievers.medical_retriever import MedicalRetriever
from .retrievers.clinical_retriever import ClinicalGuidelineRetriever
from .retrievers.hybrid_retriever import HybridRetriever

from .indexers.document_indexer import DocumentIndexer
from .indexers.medical_text_processor import MedicalTextProcessor

# Define what gets imported with "from rag_system import *"
__all__ = [
    # Vector Store Components
    'MedicalEmbeddingGenerator',
    'QdrantManager', 
    'DocumentSchema',
    'VectorMetadata',
    
    # Retrievers
    'MedicalRetriever',
    'ClinicalGuidelineRetriever',
    'HybridRetriever',
    
    # Indexers
    'DocumentIndexer',
    'MedicalTextProcessor'
]

# Package metadata
__version__ = '1.0.0'
__author__ = 'AegisMedBot Team'
__description__ = 'Medical RAG System for Hospital Intelligence Platform'