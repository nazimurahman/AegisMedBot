"""
Vector Store Schema Module

This module defines the data structures and schemas for managing vector embeddings
in the Qdrant vector database. It provides type-safe models for document storage,
retrieval, and metadata management for medical and clinical data.

The schema ensures consistency across all vector operations and enables efficient
search and retrieval of medical knowledge, clinical guidelines, and hospital policies.
"""

from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field, asdict
from uuid import uuid4
import json
import hashlib


class DocumentCategory(Enum):
    """
    Enumeration of document categories in the medical knowledge base.
    
    Categories help with filtering and specialized retrieval strategies.
    Each category corresponds to a different type of medical content.
    """
    LITERATURE = "literature"           # Research papers, medical journals
    CLINICAL_GUIDELINE = "guideline"    # Clinical practice guidelines
    HOSPITAL_POLICY = "policy"          # Internal hospital policies
    MEDICAL_TEXTBOOK = "textbook"       # Standard medical textbooks
    DRUG_INFORMATION = "drug_info"      # Pharmaceutical information
    PATIENT_EDUCATION = "patient_edu"   # Patient-facing materials
    OPERATIONAL = "operational"         # Hospital operations documents
    TRAINING = "training"               # Staff training materials


class DocumentSource(Enum):
    """
    Source origins for documents in the knowledge base.
    
    Tracks where each document originated to maintain provenance
    and enable source-specific filtering.
    """
    PUBMED = "pubmed"                   # PubMed Central/Medline
    INTERNAL = "internal"               # Hospital internal documents
    WHO = "who"                         # World Health Organization
    CDC = "cdc"                         # Centers for Disease Control
    FDA = "fda"                         # Food and Drug Administration
    SPECIALTY_SOCIETY = "specialty"     # Medical specialty societies
    TEXTBOOK = "textbook"               # Published medical textbooks
    CUSTOM = "custom"                   # Custom/curated documents


class RetrievalStrategy(Enum):
    """
    Available retrieval strategies for the RAG system.
    
    Different strategies optimize for different use cases:
    - DENSE: Vector similarity only (best for semantic search)
    - SPARSE: Keyword matching only (best for exact term matching)
    - HYBRID: Combination of dense and sparse (balanced approach)
    - CROSS_ENCODER: Re-ranking with cross-encoder models (highest accuracy)
    """
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"
    CROSS_ENCODER = "cross_encoder"


@dataclass
class DocumentMetadata:
    """
    Metadata structure for documents in the vector store.
    
    This class captures all relevant metadata about a document to enable
    filtering, sorting, and provenance tracking. All fields are optional
    to accommodate different document types and sources.
    
    Attributes:
        source: Original source of the document (e.g., PUBMED, INTERNAL)
        category: Type of document (e.g., literature, guideline)
        author: Author or organization responsible for content
        publication_date: Date when document was published
        access_date: Date when document was added to the system
        version: Version number for versioned documents
        department: Hospital department this document pertains to
        specialty: Medical specialty area (e.g., cardiology, oncology)
        confidence_score: Quality/confidence rating of the source (0-1)
        citation_count: Number of citations (for academic papers)
        doi: Digital Object Identifier for academic papers
        pmid: PubMed ID for indexed papers
        tags: List of keywords or tags for categorization
        language: Language of the document
        is_reviewed: Whether content has been clinically reviewed
        review_date: Date of clinical review if applicable
        reviewer: Name/ID of clinical reviewer
        expires_at: Expiration date for time-sensitive content
        custom_fields: Additional custom metadata fields
    """
    source: Optional[str] = None
    category: Optional[str] = None
    author: Optional[str] = None
    publication_date: Optional[str] = None
    access_date: Optional[str] = None
    version: Optional[str] = None
    department: Optional[str] = None
    specialty: Optional[str] = None
    confidence_score: Optional[float] = None
    citation_count: Optional[int] = None
    doi: Optional[str] = None
    pmid: Optional[str] = None
    tags: Optional[List[str]] = None
    language: Optional[str] = "en"
    is_reviewed: Optional[bool] = False
    review_date: Optional[str] = None
    reviewer: Optional[str] = None
    expires_at: Optional[str] = None
    custom_fields: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert metadata to dictionary for JSON serialization.
        
        Returns:
            Dictionary representation of all metadata fields
        """
        return asdict(self)
    
    def from_dict(self, data: Dict[str, Any]) -> 'DocumentMetadata':
        """
        Create metadata instance from dictionary.
        
        Args:
            data: Dictionary containing metadata fields
            
        Returns:
            DocumentMetadata instance populated from dictionary
        """
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self


@dataclass
class VectorDocument:
    """
    Complete document structure for vector storage.
    
    This class represents a document with its text content, embedding vector,
    and associated metadata. It serves as the primary data structure for
    all vector operations.
    
    Attributes:
        id: Unique identifier for the document (UUID)
        text: Original text content of the document
        embedding: Vector embedding of the text (list of floats)
        metadata: DocumentMetadata object with descriptive information
        text_hash: Hash of the text for deduplication
        created_at: Timestamp when document was added
        updated_at: Timestamp when document was last updated
        chunk_index: Index if document is part of a larger document chunk
        parent_id: ID of parent document if this is a chunk
    """
    id: str = field(default_factory=lambda: str(uuid4()))
    text: str = ""
    embedding: Optional[List[float]] = None
    metadata: DocumentMetadata = field(default_factory=DocumentMetadata)
    text_hash: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    chunk_index: Optional[int] = None
    parent_id: Optional[str] = None
    
    def __post_init__(self):
        """
        Post-initialization hook to compute text hash if not provided.
        
        The hash enables quick deduplication and change detection.
        """
        if self.text and not self.text_hash:
            self.text_hash = hashlib.sha256(self.text.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert document to dictionary for serialization.
        
        Returns:
            Dictionary containing all document fields
        """
        return {
            'id': self.id,
            'text': self.text,
            'embedding': self.embedding,
            'metadata': self.metadata.to_dict(),
            'text_hash': self.text_hash,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'chunk_index': self.chunk_index,
            'parent_id': self.parent_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VectorDocument':
        """
        Create document instance from dictionary.
        
        Args:
            data: Dictionary containing document fields
            
        Returns:
            VectorDocument instance populated from dictionary
        """
        metadata = DocumentMetadata()
        if 'metadata' in data and data['metadata']:
            metadata.from_dict(data['metadata'])
        
        return cls(
            id=data.get('id', str(uuid4())),
            text=data.get('text', ''),
            embedding=data.get('embedding'),
            metadata=metadata,
            text_hash=data.get('text_hash'),
            created_at=data.get('created_at', datetime.now().isoformat()),
            updated_at=data.get('updated_at', datetime.now().isoformat()),
            chunk_index=data.get('chunk_index'),
            parent_id=data.get('parent_id')
        )


@dataclass
class SearchResult:
    """
    Structure for search results from vector retrieval.
    
    This class encapsulates a retrieved document along with its relevance
    score and additional search-specific metadata.
    
    Attributes:
        document: The retrieved VectorDocument
        score: Similarity score from the search (0-1)
        rank: Position in the search results
        retrieval_method: Which strategy produced this result
        explanation: Optional explanation of why this document was retrieved
        highlight: Optional highlighted text snippet
    """
    document: VectorDocument
    score: float
    rank: int
    retrieval_method: str
    explanation: Optional[str] = None
    highlight: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert search result to dictionary.
        
        Returns:
            Dictionary representation suitable for API responses
        """
        return {
            'document': self.document.to_dict(),
            'score': self.score,
            'rank': self.rank,
            'retrieval_method': self.retrieval_method,
            'explanation': self.explanation,
            'highlight': self.highlight
        }


@dataclass
class SearchQuery:
    """
    Structured query for vector search operations.
    
    This class encapsulates all parameters needed to perform a search,
    supporting complex filtering and customization.
    
    Attributes:
        query_text: The original user query text
        query_vector: Optional pre-computed embedding vector
        top_k: Number of results to return
        filter_conditions: Dictionary of metadata filters
        min_score: Minimum similarity score threshold
        retrieval_strategy: Which retrieval method to use
        hybrid_alpha: Weight for hybrid search (0=sparse only, 1=dense only)
        rerank: Whether to rerank results with cross-encoder
        department_filter: Filter by specific hospital department
        specialty_filter: Filter by medical specialty
        date_range: Tuple of (start_date, end_date) for publication dates
        require_reviewed: Only return clinically reviewed documents
        categories: List of DocumentCategory values to include
        sources: List of DocumentSource values to include
    """
    query_text: str
    query_vector: Optional[List[float]] = None
    top_k: int = 10
    filter_conditions: Optional[Dict[str, Any]] = None
    min_score: float = 0.7
    retrieval_strategy: RetrievalStrategy = RetrievalStrategy.HYBRID
    hybrid_alpha: float = 0.5
    rerank: bool = False
    department_filter: Optional[str] = None
    specialty_filter: Optional[str] = None
    date_range: Optional[tuple] = None
    require_reviewed: bool = False
    categories: Optional[List[DocumentCategory]] = None
    sources: Optional[List[DocumentSource]] = None
    
    def validate(self) -> bool:
        """
        Validate query parameters before execution.
        
        Returns:
            True if query is valid, False otherwise
        """
        if not self.query_text and not self.query_vector:
            return False
        
        if self.top_k <= 0 or self.top_k > 100:
            return False
        
        if self.min_score < 0 or self.min_score > 1:
            return False
        
        if self.hybrid_alpha < 0 or self.hybrid_alpha > 1:
            return False
        
        return True
    
    def build_filter(self) -> Dict[str, Any]:
        """
        Build Qdrant filter from query parameters.
        
        Returns:
            Dictionary formatted for Qdrant filtering syntax
        """
        must_conditions = []
        
        # Department filter
        if self.department_filter:
            must_conditions.append({
                'key': 'metadata.department',
                'match': {'value': self.department_filter}
            })
        
        # Specialty filter
        if self.specialty_filter:
            must_conditions.append({
                'key': 'metadata.specialty',
                'match': {'value': self.specialty_filter}
            })
        
        # Categories filter
        if self.categories:
            category_values = [cat.value for cat in self.categories]
            must_conditions.append({
                'key': 'metadata.category',
                'match': {'any': category_values}
            })
        
        # Sources filter
        if self.sources:
            source_values = [src.value for src in self.sources]
            must_conditions.append({
                'key': 'metadata.source',
                'match': {'any': source_values}
            })
        
        # Reviewed filter
        if self.require_reviewed:
            must_conditions.append({
                'key': 'metadata.is_reviewed',
                'match': {'value': True}
            })
        
        # Date range filter
        if self.date_range:
            start_date, end_date = self.date_range
            must_conditions.append({
                'key': 'metadata.publication_date',
                'range': {
                    'gte': start_date,
                    'lte': end_date
                }
            })
        
        # Combine all conditions
        if must_conditions:
            return {'must': must_conditions}
        
        return None


class VectorStoreConfig:
    """
    Configuration class for the vector store.
    
    Centralizes all configuration parameters for the Qdrant vector database
    connection and behavior.
    
    Attributes:
        host: Qdrant server hostname
        port: Qdrant server port
        collection_name: Name of the collection to use
        vector_size: Dimension of embedding vectors
        distance_metric: Distance metric for similarity (cosine, dot, euclid)
        batch_size: Batch size for bulk operations
        replication_factor: Number of replicas for collection
        shard_number: Number of shards for collection
        write_timeout: Timeout for write operations in seconds
        read_timeout: Timeout for read operations in seconds
        retry_attempts: Number of retry attempts for failed operations
        enable_payload_index: Whether to create indexes on payload fields
        payload_index_fields: List of fields to index for faster filtering
    """
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "medical_knowledge",
        vector_size: int = 768,
        distance_metric: str = "cosine",
        batch_size: int = 100,
        replication_factor: int = 1,
        shard_number: int = 1,
        write_timeout: int = 30,
        read_timeout: int = 30,
        retry_attempts: int = 3,
        enable_payload_index: bool = True,
        payload_index_fields: List[str] = None
    ):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.distance_metric = distance_metric
        self.batch_size = batch_size
        self.replication_factor = replication_factor
        self.shard_number = shard_number
        self.write_timeout = write_timeout
        self.read_timeout = read_timeout
        self.retry_attempts = retry_attempts
        self.enable_payload_index = enable_payload_index
        self.payload_index_fields = payload_index_fields or [
            'metadata.source',
            'metadata.category',
            'metadata.specialty',
            'metadata.department'
        ]
    
    def get_qdrant_url(self) -> str:
        """
        Get the full Qdrant connection URL.
        
        Returns:
            Formatted URL string for Qdrant client
        """
        return f"http://{self.host}:{self.port}"
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dictionary of all configuration parameters
        """
        return {
            'host': self.host,
            'port': self.port,
            'collection_name': self.collection_name,
            'vector_size': self.vector_size,
            'distance_metric': self.distance_metric,
            'batch_size': self.batch_size,
            'replication_factor': self.replication_factor,
            'shard_number': self.shard_number,
            'write_timeout': self.write_timeout,
            'read_timeout': self.read_timeout,
            'retry_attempts': self.retry_attempts,
            'enable_payload_index': self.enable_payload_index,
            'payload_index_fields': self.payload_index_fields
        }