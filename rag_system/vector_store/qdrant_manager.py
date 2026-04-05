"""
Qdrant Vector Database Manager for Medical Knowledge Storage

This module provides a comprehensive interface for managing vector embeddings
in Qdrant, a high-performance vector similarity search engine. It handles
collection management, document insertion, similarity search, and hybrid search
operations optimized for medical content retrieval.

Qdrant is chosen as the vector database because:
- High performance for billion-scale vector search
- Support for rich metadata filtering
- Built-in support for hybrid search with keyword matching
- Excellent Python client with async support
- Production-ready with replication and sharding

The QdrantManager provides:
- Collection initialization and management
- Batch insertion of documents with embeddings
- Vector similarity search with filtering
- Hybrid search combining vector and keyword relevance
- Document updates and deletions
- Collection statistics and monitoring

Dependencies:
- qdrant-client: Python client for Qdrant
- numpy: For vector operations
- typing: For type hints
"""

import numpy as np
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import logging
from enum import Enum

# Import Qdrant client components
try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    from qdrant_client.http.models import Distance, VectorParams, PointStruct
    from qdrant_client.http.models import Filter, FieldCondition, MatchValue
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    logging.warning("Qdrant client not available. Install with: pip install qdrant-client")

# Configure logging
logger = logging.getLogger(__name__)

class SearchType(Enum):
    """
    Enumeration of available search types for vector retrieval.
    
    This defines the different strategies that can be used when searching
    the vector database:
    - VECTOR: Pure vector similarity search using cosine distance
    - KEYWORD: Keyword-based search using BM25 or text matching
    - HYBRID: Combination of vector and keyword search
    - FILTERED: Vector search with metadata filters
    """
    VECTOR = "vector"
    KEYWORD = "keyword"
    HYBRID = "hybrid"
    FILTERED = "filtered"

class DocumentStatus(Enum):
    """
    Document status for tracking in the vector store.
    
    These statuses help track the lifecycle of documents in the system:
    - PENDING: Document indexed but not yet searchable
    - ACTIVE: Document is searchable
    - ARCHIVED: Document is hidden from search
    - DELETED: Document is removed (soft delete)
    """
    PENDING = "pending"
    ACTIVE = "active"
    ARCHIVED = "archived"
    DELETED = "deleted"

class QdrantManager:
    """
    Manages vector storage and retrieval operations in Qdrant.
    
    This class provides a high-level interface for all vector database
    operations needed by the RAG system. It handles:
    1. Collection initialization and configuration
    2. Batch insertion of documents with embeddings
    3. Similarity search with filtering capabilities
    4. Hybrid search combining vector and text relevance
    5. Document updates and deletions
    6. Collection statistics and health monitoring
    
    The manager maintains a single collection for medical documents but
    uses payload fields to categorize and filter different document types
    (literature, guidelines, policies, etc.).
    
    Attributes:
        client: Qdrant client instance for database operations
        collection_name: Name of the vector collection
        vector_size: Dimension of the embedding vectors
        collection_initialized: Flag indicating if collection exists
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "medical_knowledge",
        vector_size: int = 768,
        recreate_collection: bool = False,
        prefer_grpc: bool = False
    ):
        """
        Initialize the Qdrant manager with connection parameters.
        
        Args:
            host: Qdrant server hostname or IP address
            port: Qdrant server port (HTTP: 6333, gRPC: 6334)
            collection_name: Name of the collection to use
            vector_size: Dimension of embedding vectors
            recreate_collection: Whether to delete and recreate if exists
            prefer_grpc: Use gRPC protocol for better performance
        
        Raises:
            ImportError: If qdrant-client is not installed
            ConnectionError: If unable to connect to Qdrant server
        """
        if not QDRANT_AVAILABLE:
            raise ImportError(
                "Qdrant client is not installed. "
                "Install with: pip install qdrant-client"
            )
        
        # Store configuration
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.host = host
        self.port = port
        
        try:
            # Initialize Qdrant client
            # Using HTTP protocol by default, gRPC for better performance
            self.client = QdrantClient(
                host=host,
                port=port,
                prefer_grpc=prefer_grpc
            )
            
            # Test connection by getting collection list
            self.client.get_collections()
            logger.info(f"Connected to Qdrant at {host}:{port}")
            
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {str(e)}")
            raise ConnectionError(f"Cannot connect to Qdrant server: {e}")
        
        # Initialize or recreate the collection
        self._init_collection(recreate=recreate_collection)
        
        # Flag to track initialization status
        self.collection_initialized = True
    
    def _init_collection(self, recreate: bool = False):
        """
        Initialize the vector collection in Qdrant.
        
        This method creates the collection if it doesn't exist. If recreate
        is True, it will delete the existing collection and create a new one.
        
        The collection is configured with:
        - Cosine distance for similarity measurement (good for normalized vectors)
        - Payload indexes for efficient filtering on key fields
        - Optimized configuration for production use
        
        Args:
            recreate: Whether to delete and recreate the collection
        
        Raises:
            Exception: If collection creation fails
        """
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_exists = any(
                col.name == self.collection_name 
                for col in collections
            )
            
            # Delete collection if recreating
            if recreate and collection_exists:
                logger.info(f"Deleting existing collection: {self.collection_name}")
                self.client.delete_collection(self.collection_name)
                collection_exists = False
            
            # Create collection if it doesn't exist
            if not collection_exists:
                logger.info(f"Creating collection: {self.collection_name}")
                
                # Define vector configuration
                # Using cosine distance which works well with normalized vectors
                vectors_config = VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE
                )
                
                # Create the collection
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=vectors_config
                )
                
                # Create payload indexes for efficient filtering
                # These fields will be frequently used in filters
                self._create_payload_indexes()
                
                logger.info(f"Collection created successfully")
            else:
                logger.info(f"Collection already exists: {self.collection_name}")
                
        except Exception as e:
            logger.error(f"Failed to initialize collection: {str(e)}")
            raise
    
    def _create_payload_indexes(self):
        """
        Create indexes on payload fields for faster filtering.
        
        Indexes are created on fields that will be used in search filters
        to improve query performance. This is crucial for production
        workloads with large collections.
        
        Fields indexed:
        - source: Document source (pubmed, guidelines, etc.)
        - category: Document category (clinical, operational, etc.)
        - status: Document status (active, archived, etc.)
        - department: Medical department (cardiology, oncology, etc.)
        - year: Publication year for time-based filtering
        """
        try:
            # Define fields to index
            indexed_fields = [
                ("source", models.PayloadSchemaType.KEYWORD),
                ("category", models.PayloadSchemaType.KEYWORD),
                ("status", models.PayloadSchemaType.KEYWORD),
                ("department", models.PayloadSchemaType.KEYWORD),
                ("year", models.PayloadSchemaType.INTEGER),
                ("author", models.PayloadSchemaType.KEYWORD)
            ]
            
            # Create index for each field
            for field_name, field_type in indexed_fields:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=field_type
                )
                logger.debug(f"Created index on field: {field_name}")
                
            logger.info(f"Payload indexes created successfully")
            
        except Exception as e:
            logger.warning(f"Some indexes already exist: {str(e)}")
    
    def insert_documents(
        self,
        documents: List[Dict[str, Any]],
        embeddings: List[np.ndarray],
        batch_size: int = 100
    ) -> List[str]:
        """
        Insert documents with their embeddings into the vector store.
        
        Args:
            documents: List of document dictionaries with metadata.
                      Each document must contain at least 'text' field.
                      Optional fields: 'source', 'category', 'metadata', etc.
            embeddings: List of embedding vectors corresponding to documents
            batch_size: Number of documents to insert in a single batch
        
        Returns:
            List of document IDs that were inserted
        
        This method handles batch insertion of documents for efficiency.
        Each document is stored as a point with:
        - ID: Unique identifier (from document or auto-generated)
        - Vector: The embedding vector
        - Payload: All metadata including text content and filters
        
        Example document structure:
        {
            'id': 'doc_123',
            'text': 'Clinical text content...',
            'source': 'pubmed',
            'category': 'clinical_guideline',
            'department': 'cardiology',
            'year': 2023,
            'metadata': {'authors': [...], 'title': '...'}
        }
        """
        if len(documents) != len(embeddings):
            raise ValueError(
                f"Number of documents ({len(documents)}) and embeddings "
                f"({len(embeddings)}) must match"
            )
        
        if not documents:
            logger.warning("No documents to insert")
            return []
        
        inserted_ids = []
        points = []
        
        for idx, (doc, emb) in enumerate(zip(documents, embeddings)):
            # Generate ID if not provided
            doc_id = doc.get('id', f"doc_{idx}_{datetime.now().timestamp()}")
            
            # Prepare payload with required fields
            payload = {
                'text': doc.get('text', ''),
                'source': doc.get('source', 'unknown'),
                'category': doc.get('category', 'general'),
                'status': DocumentStatus.ACTIVE.value,
                'created_at': datetime.now().isoformat(),
                'metadata': doc.get('metadata', {})
            }
            
            # Add optional fields if present
            if 'department' in doc:
                payload['department'] = doc['department']
            if 'year' in doc:
                payload['year'] = doc['year']
            if 'author' in doc:
                payload['author'] = doc['author']
            if 'title' in doc:
                payload['title'] = doc['title']
            
            # Create point structure for Qdrant
            point = PointStruct(
                id=doc_id,
                vector=emb.tolist() if isinstance(emb, np.ndarray) else emb,
                payload=payload
            )
            points.append(point)
            inserted_ids.append(doc_id)
            
            # Insert in batches to avoid overwhelming the server
            if len(points) >= batch_size:
                self._upsert_points(points)
                logger.info(f"Inserted batch of {len(points)} documents")
                points = []
        
        # Insert remaining points
        if points:
            self._upsert_points(points)
            logger.info(f"Inserted final batch of {len(points)} documents")
        
        logger.info(f"Successfully inserted {len(inserted_ids)} documents")
        return inserted_ids
    
    def _upsert_points(self, points: List[PointStruct]):
        """
        Upsert points into the collection.
        
        Args:
            points: List of PointStruct objects to insert
        
        This internal method handles the actual insertion operation
        and catches any errors for logging.
        """
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
        except Exception as e:
            logger.error(f"Failed to upsert points: {str(e)}")
            raise
    
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        filter_conditions: Optional[Dict[str, Any]] = None,
        score_threshold: float = 0.7,
        with_payload: bool = True,
        with_vectors: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents using vector similarity.
        
        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            filter_conditions: Optional filters to apply to search
                               Example: {'source': 'pubmed', 'year': 2023}
            score_threshold: Minimum similarity score threshold (0-1)
            with_payload: Whether to return document payload
            with_vectors: Whether to return embedding vectors
        
        Returns:
            List of search results with similarity scores
        
        This is the primary search method for semantic similarity. It finds
        documents whose vectors are closest to the query vector using
        cosine similarity (since we normalized vectors and use COSINE distance).
        
        The search can be filtered by metadata fields using the filter_conditions,
        which is crucial for restricting search to specific document types.
        """
        if not self.collection_initialized:
            raise RuntimeError("Collection not initialized")
        
        # Build filter if conditions provided
        search_filter = None
        if filter_conditions:
            search_filter = self._build_filter(filter_conditions)
        
        try:
            # Execute search
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector.tolist(),
                query_filter=search_filter,
                limit=top_k,
                score_threshold=score_threshold,
                with_payload=with_payload,
                with_vectors=with_vectors
            )
            
            # Format results
            results = []
            for scored_point in search_result:
                result = {
                    'id': scored_point.id,
                    'score': scored_point.score
                }
                
                if with_payload:
                    result['payload'] = scored_point.payload
                
                if with_vectors and scored_point.vector:
                    result['vector'] = scored_point.vector
                
                results.append(result)
            
            logger.debug(f"Search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return []
    
    def _build_filter(self, conditions: Dict[str, Any]) -> Filter:
        """
        Build a Qdrant filter from dictionary conditions.
        
        Args:
            conditions: Dictionary of field-value pairs to filter on
        
        Returns:
            Qdrant Filter object for use in search
        
        This method converts simple key-value filters into Qdrant's
        filter structure. For example:
        {'source': 'pubmed', 'year': 2023}
        becomes a MUST filter with two conditions.
        
        Supports:
        - Equality: {'field': value}
        - Lists: {'field': [value1, value2]} (OR condition)
        - Range: {'year': {'gte': 2020, 'lte': 2023}}
        """
        must_conditions = []
        
        for field, value in conditions.items():
            if isinstance(value, dict):
                # Handle range conditions
                if 'gte' in value or 'lte' in value:
                    condition = FieldCondition(
                        key=field,
                        range=models.Range(
                            gte=value.get('gte'),
                            lte=value.get('lte')
                        )
                    )
                else:
                    # Unknown dict structure - skip
                    logger.warning(f"Unknown filter structure for {field}: {value}")
                    continue
            elif isinstance(value, list):
                # Handle OR conditions for lists
                # Create a filter with multiple conditions
                or_conditions = []
                for item in value:
                    or_conditions.append(
                        FieldCondition(
                            key=field,
                            match=MatchValue(value=item)
                        )
                    )
                condition = Filter(should=or_conditions)
            else:
                # Simple equality condition
                condition = FieldCondition(
                    key=field,
                    match=MatchValue(value=value)
                )
            
            must_conditions.append(condition)
        
        return Filter(must=must_conditions) if must_conditions else None
    
    def hybrid_search(
        self,
        query_vector: np.ndarray,
        text_query: str,
        top_k: int = 10,
        alpha: float = 0.5,
        filter_conditions: Optional[Dict[str, Any]] = None,
        score_threshold: float = 0.6
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining vector similarity and keyword matching.
        
        Args:
            query_vector: Query embedding vector
            text_query: Original text query for keyword search
            top_k: Number of results to return
            alpha: Weight for vector search (1-alpha for keyword)
            filter_conditions: Optional metadata filters
            score_threshold: Minimum combined score threshold
        
        Returns:
            List of search results with combined scores
        
        Hybrid search combines the semantic understanding of vector search
        with the precision of keyword matching. This is particularly useful
        for medical queries where specific terms (like drug names) are critical.
        
        The alpha parameter controls the balance:
        - alpha=1.0: Pure vector search
        - alpha=0.0: Pure keyword search
        - alpha=0.5: Balanced approach
        
        In production, Qdrant supports native hybrid search with BM25,
        but we implement a combined approach here for flexibility.
        """
        if not self.collection_initialized:
            raise RuntimeError("Collection not initialized")
        
        # Build filter if provided
        search_filter = None
        if filter_conditions:
            search_filter = self._build_filter(filter_conditions)
        
        try:
            # Perform vector search
            vector_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector.tolist(),
                query_filter=search_filter,
                limit=top_k * 2,  # Retrieve more for combination
                score_threshold=score_threshold
            )
            
            # In production, would perform keyword search with Qdrant's BM25
            # For now, we simulate keyword search based on text presence
            keyword_results = self._simulate_keyword_search(
                text_query,
                search_filter,
                top_k * 2
            )
            
            # Combine results with weighted scoring
            combined = self._combine_search_results(
                vector_results,
                keyword_results,
                alpha
            )
            
            # Return top K results
            return combined[:top_k]
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {str(e)}")
            return []
    
    def _simulate_keyword_search(
        self,
        query: str,
        search_filter: Optional[Filter],
        limit: int
    ) -> List[Dict[str, Any]]:
        """
        Simulate keyword search using scroll with text matching.
        
        Args:
            query: Text query for keyword matching
            search_filter: Optional metadata filter
            limit: Maximum number of results
        
        Returns:
            List of keyword search results
        
        This is a simplified keyword search for demonstration.
        In production, use Qdrant's full-text search with BM25 for
        better keyword matching.
        """
        try:
            # Use scroll to retrieve documents with filtering
            scroll_result = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=search_filter,
                limit=limit
            )
            
            points = scroll_result[0]
            keyword_results = []
            
            query_lower = query.lower()
            
            for point in points:
                text = point.payload.get('text', '').lower()
                
                # Simple keyword matching score
                # Count occurrences of query terms in text
                query_terms = query_lower.split()
                score = sum(text.count(term) for term in query_terms)
                
                if score > 0:
                    keyword_results.append({
                        'id': point.id,
                        'score': min(score / 10.0, 1.0),  # Normalize
                        'keyword_matches': score
                    })
            
            # Sort by score descending
            keyword_results.sort(key=lambda x: x['score'], reverse=True)
            
            return keyword_results
            
        except Exception as e:
            logger.error(f"Keyword search failed: {str(e)}")
            return []
    
    def _combine_search_results(
        self,
        vector_results: List,
        keyword_results: List,
        alpha: float
    ) -> List[Dict[str, Any]]:
        """
        Combine vector and keyword search results with weighted scoring.
        
        Args:
            vector_results: Results from vector similarity search
            keyword_results: Results from keyword search
            alpha: Weight for vector scores (1-alpha for keyword)
        
        Returns:
            Combined and sorted results
        """
        combined_dict = {}
        
        # Add vector results
        for result in vector_results:
            combined_dict[result.id] = {
                'id': result.id,
                'vector_score': result.score,
                'keyword_score': 0,
                'payload': result.payload
            }
        
        # Add or update keyword results
        for result in keyword_results:
            doc_id = result['id']
            if doc_id in combined_dict:
                combined_dict[doc_id]['keyword_score'] = result['score']
            else:
                combined_dict[doc_id] = {
                    'id': doc_id,
                    'vector_score': 0,
                    'keyword_score': result['score'],
                    'payload': result.get('payload', {})
                }
        
        # Calculate final scores
        results = []
        for doc_id, data in combined_dict.items():
            final_score = (
                alpha * data['vector_score'] +
                (1 - alpha) * data['keyword_score']
            )
            results.append({
                'id': doc_id,
                'score': final_score,
                'payload': data.get('payload', {})
            })
        
        # Sort by final score
        results.sort(key=lambda x: x['score'], reverse=True)
        
        return results
    
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a document by its ID.
        
        Args:
            doc_id: Document identifier
        
        Returns:
            Document dictionary or None if not found
        """
        try:
            result = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[doc_id]
            )
            
            if result:
                point = result[0]
                return {
                    'id': point.id,
                    'vector': point.vector,
                    'payload': point.payload
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to retrieve document {doc_id}: {str(e)}")
            return None
    
    def update_document(
        self,
        doc_id: str,
        updates: Dict[str, Any],
        embedding: Optional[np.ndarray] = None
    ) -> bool:
        """
        Update a document's payload and optionally its vector.
        
        Args:
            doc_id: Document identifier
            updates: Dictionary of fields to update
            embedding: New embedding vector if vector should change
        
        Returns:
            True if update successful, False otherwise
        
        This method allows partial updates to document payload without
        replacing the entire document. It's useful for updating metadata
        or marking documents as archived.
        """
        try:
            # Get existing document
            existing = self.get_document(doc_id)
            if not existing:
                logger.warning(f"Document {doc_id} not found")
                return False
            
            # Merge updates with existing payload
            updated_payload = existing['payload']
            updated_payload.update(updates)
            updated_payload['updated_at'] = datetime.now().isoformat()
            
            # Use existing vector if no new embedding provided
            vector = embedding if embedding is not None else existing['vector']
            
            # Update the point
            point = PointStruct(
                id=doc_id,
                vector=vector.tolist() if isinstance(vector, np.ndarray) else vector,
                payload=updated_payload
            )
            
            self._upsert_points([point])
            logger.info(f"Updated document {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update document {doc_id}: {str(e)}")
            return False
    
    def delete_documents(self, doc_ids: List[str], soft_delete: bool = True) -> int:
        """
        Delete documents by ID.
        
        Args:
            doc_ids: List of document identifiers
            soft_delete: If True, mark as deleted instead of removing
        
        Returns:
            Number of documents deleted
        
        Soft delete marks documents as deleted without removing them,
        which is useful for audit trails and recovery. Hard delete
        removes them permanently.
        """
        if not doc_ids:
            return 0
        
        deleted_count = 0
        
        if soft_delete:
            # Soft delete: update status to DELETED
            for doc_id in doc_ids:
                if self.update_document(doc_id, {'status': DocumentStatus.DELETED.value}):
                    deleted_count += 1
        else:
            # Hard delete: remove points
            try:
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=models.PointIdsList(points=doc_ids)
                )
                deleted_count = len(doc_ids)
                logger.info(f"Hard deleted {deleted_count} documents")
            except Exception as e:
                logger.error(f"Hard delete failed: {str(e)}")
        
        return deleted_count
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector collection.
        
        Returns:
            Dictionary with collection statistics
        
        Useful for monitoring and capacity planning.
        """
        try:
            collection_info = self.client.get_collection(self.collection_name)
            
            return {
                'name': self.collection_name,
                'vectors_count': collection_info.vectors_count,
                'segments_count': collection_info.segments_count,
                'status': collection_info.status,
                'vector_size': self.vector_size,
                'points_count': collection_info.points_count,
                'indexed_vectors_count': collection_info.indexed_vectors_count
            }
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {str(e)}")
            return {
                'name': self.collection_name,
                'error': str(e)
            }
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check the health of the Qdrant connection.
        
        Returns:
            Dictionary with health status information
        """
        try:
            # Test connection by getting collections
            collections = self.client.get_collections()
            
            return {
                'status': 'healthy',
                'host': self.host,
                'port': self.port,
                'collection': self.collection_name,
                'collection_exists': any(
                    col.name == self.collection_name 
                    for col in collections.collections
                ),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'host': self.host,
                'port': self.port,
                'timestamp': datetime.now().isoformat()
            }