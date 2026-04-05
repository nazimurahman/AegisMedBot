"""
Medical Document Retriever Module

This module implements specialized retrieval strategies for medical and clinical
documents. It provides dense, sparse, and hybrid retrieval methods optimized for
medical terminology and clinical concepts.

The retriever integrates with the vector store and embedding system to find
relevant medical documents based on semantic similarity and keyword matching.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import numpy as np

# Import from local modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vector_store.schema import (
    VectorDocument, SearchQuery, SearchResult, 
    RetrievalStrategy, DocumentCategory, DocumentSource
)
from vector_store.qdrant_manager import QdrantManager
from vector_store.embeddings import MedicalEmbeddingGenerator

# Configure logging
logger = logging.getLogger(__name__)


class MedicalRetriever:
    """
    Main medical document retriever class.
    
    This class coordinates all retrieval operations for medical documents,
    supporting multiple retrieval strategies and optimization for medical
    terminology.
    
    The retriever handles:
    - Dense vector similarity search
    - Sparse keyword-based search
    - Hybrid search combining both methods
    - Cross-encoder reranking for improved relevance
    - Medical term expansion and synonym handling
    
    Attributes:
        vector_store: QdrantManager instance for vector operations
        embedding_generator: MedicalEmbeddingGenerator for creating embeddings
        cross_encoder_model: Optional cross-encoder model for reranking
        medical_thesaurus: Dictionary of medical term synonyms
        max_retrieval_depth: Maximum number of documents to retrieve before filtering
        rerank_top_k: Number of documents to keep after reranking
    """
    
    def __init__(
        self,
        vector_store: QdrantManager,
        embedding_generator: MedicalEmbeddingGenerator,
        cross_encoder_model: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the medical retriever.
        
        Args:
            vector_store: Initialized QdrantManager instance
            embedding_generator: MedicalEmbeddingGenerator for embeddings
            cross_encoder_model: Optional cross-encoder for reranking
            config: Configuration dictionary with retrieval parameters
        """
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator
        self.cross_encoder_model = cross_encoder_model
        self.config = config or {}
        
        # Retrieve configuration parameters with defaults
        self.max_retrieval_depth = self.config.get('max_retrieval_depth', 50)
        self.rerank_top_k = self.config.get('rerank_top_k', 10)
        
        # Medical term thesaurus for query expansion
        # In production, this would load from a proper medical ontology
        self.medical_thesaurus = self._load_medical_thesaurus()
        
        logger.info("MedicalRetriever initialized successfully")
    
    def _load_medical_thesaurus(self) -> Dict[str, List[str]]:
        """
        Load medical term thesaurus for query expansion.
        
        This method creates a dictionary of common medical terms and their
        synonyms, abbreviations, and related terms to improve retrieval recall.
        
        Returns:
            Dictionary mapping base terms to lists of synonyms
        """
        # Core medical term mappings
        # In production, this would load from a file or database
        thesaurus = {
            # Cardiovascular terms
            'myocardial infarction': [
                'heart attack', 'mi', 'acute myocardial infarction',
                'ami', 'coronary thrombosis', 'cardiac infarction'
            ],
            'hypertension': [
                'high blood pressure', 'htn', 'elevated bp',
                'hypertensive disease', 'essential hypertension'
            ],
            'heart failure': [
                'hf', 'congestive heart failure', 'chf',
                'cardiac failure', 'decompensated heart failure'
            ],
            
            # Respiratory terms
            'pneumonia': [
                'lung infection', 'pneumonitis', 'bronchopneumonia',
                'community acquired pneumonia', 'cap'
            ],
            'asthma': [
                'bronchial asthma', 'reactive airway disease',
                'wheezing disorder', 'asthmatic bronchitis'
            ],
            
            # Metabolic terms
            'diabetes mellitus': [
                'diabetes', 'dm', 'type 2 diabetes', 'type 1 diabetes',
                'hyperglycemia', 'insulin resistance'
            ],
            
            # Infectious disease terms
            'sepsis': [
                'septicemia', 'blood infection', 'septic shock',
                'systemic inflammatory response syndrome', 'sirs'
            ],
            
            # Renal terms
            'acute kidney injury': [
                'aki', 'acute renal failure', 'arf',
                'acute kidney failure', 'renal insufficiency'
            ],
            
            # Neurological terms
            'stroke': [
                'cerebrovascular accident', 'cva', 'brain attack',
                'ischemic stroke', 'hemorrhagic stroke'
            ],
            
            # Oncology terms
            'cancer': [
                'malignancy', 'neoplasm', 'tumor', 'carcinoma',
                'malignant neoplasm', 'metastatic disease'
            ]
        }
        
        return thesaurus
    
    async def retrieve(
        self,
        query: SearchQuery,
        use_query_expansion: bool = True
    ) -> List[SearchResult]:
        """
        Main retrieval method that coordinates the retrieval process.
        
        This method handles the complete retrieval pipeline:
        1. Query expansion with medical synonyms
        2. Embedding generation (if needed)
        3. Vector search based on strategy
        4. Optional reranking with cross-encoder
        5. Result formatting and scoring
        
        Args:
            query: SearchQuery object with retrieval parameters
            use_query_expansion: Whether to expand query with medical synonyms
            
        Returns:
            List of SearchResult objects sorted by relevance
        """
        start_time = datetime.now()
        
        # Validate the query
        if not query.validate():
            logger.error(f"Invalid query: {query.query_text}")
            return []
        
        # Expand query with medical synonyms if enabled
        expanded_query_text = query.query_text
        if use_query_expansion:
            expanded_query_text = self._expand_query(query.query_text)
            logger.debug(f"Query expanded from '{query.query_text}' to '{expanded_query_text}'")
        
        # Generate embedding if not provided
        if query.query_vector is None:
            query.query_vector = await self.embedding_generator.generate_query_embedding(
                expanded_query_text,
                use_expansion=True
            )
        
        # Execute retrieval based on strategy
        if query.retrieval_strategy == RetrievalStrategy.DENSE:
            results = await self._dense_retrieval(query)
        elif query.retrieval_strategy == RetrievalStrategy.SPARSE:
            results = await self._sparse_retrieval(query)
        elif query.retrieval_strategy == RetrievalStrategy.HYBRID:
            results = await self._hybrid_retrieval(query)
        elif query.retrieval_strategy == RetrievalStrategy.CROSS_ENCODER:
            results = await self._cross_encoder_retrieval(query)
        else:
            logger.error(f"Unknown retrieval strategy: {query.retrieval_strategy}")
            return []
        
        # Apply score threshold
        results = [r for r in results if r.score >= query.min_score]
        
        # Rerank with cross-encoder if enabled and results available
        if query.rerank and self.cross_encoder_model and results:
            results = await self._rerank_with_cross_encoder(
                query.query_text,
                results,
                self.rerank_top_k
            )
        
        # Limit to top_k
        results = results[:query.top_k]
        
        # Log retrieval performance
        elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000
        logger.info(
            f"Retrieved {len(results)} documents for query '{query.query_text[:50]}...' "
            f"in {elapsed_ms:.2f}ms using {query.retrieval_strategy.value}"
        )
        
        return results
    
    def _expand_query(self, query: str) -> str:
        """
        Expand query with medical synonyms and related terms.
        
        This method identifies medical terms in the query and adds their
        synonyms to improve retrieval recall.
        
        Args:
            query: Original query text
            
        Returns:
            Expanded query text with synonyms added
        """
        expanded_terms = [query]
        query_lower = query.lower()
        
        # Check for medical terms and add synonyms
        for term, synonyms in self.medical_thesaurus.items():
            if term in query_lower:
                # Add the original term's synonyms
                expanded_terms.extend(synonyms)
                logger.debug(f"Expanded term '{term}' with {len(synonyms)} synonyms")
        
        # Remove duplicates and join
        unique_terms = list(dict.fromkeys(expanded_terms))
        expanded_query = ' '.join(unique_terms)
        
        return expanded_query
    
    async def _dense_retrieval(self, query: SearchQuery) -> List[SearchResult]:
        """
        Perform dense vector similarity search.
        
        This method uses the embedding vector to find semantically similar
        documents in the vector store.
        
        Args:
            query: SearchQuery with query vector and parameters
            
        Returns:
            List of search results from vector similarity
        """
        try:
            # Build filter conditions
            filter_conditions = query.build_filter() if query.filter_conditions else None
            
            # Perform search in vector store
            vector_results = await self.vector_store.search(
                query_vector=np.array(query.query_vector),
                top_k=self.max_retrieval_depth,
                filter_conditions=filter_conditions,
                score_threshold=query.min_score
            )
            
            # Convert to SearchResult objects
            results = []
            for rank, result in enumerate(vector_results):
                # Create VectorDocument from result
                document = VectorDocument(
                    id=result.get('id', ''),
                    text=result.get('text', ''),
                    embedding=result.get('embedding'),
                    metadata=result.get('metadata', {}),
                    text_hash=result.get('text_hash')
                )
                
                # Create SearchResult
                search_result = SearchResult(
                    document=document,
                    score=result.get('score', 0.0),
                    rank=rank,
                    retrieval_method='dense'
                )
                results.append(search_result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in dense retrieval: {str(e)}")
            return []
    
    async def _sparse_retrieval(self, query: SearchQuery) -> List[SearchResult]:
        """
        Perform sparse keyword-based search.
        
        This method uses traditional keyword matching (BM25-like) to find
        documents containing relevant terms.
        
        Args:
            query: SearchQuery with query text and parameters
            
        Returns:
            List of search results from keyword matching
        """
        try:
            # In production, this would use a full-text search index
            # For now, we'll use vector store with keyword filtering
            # This is a simplified implementation
            
            # Build filter with keyword matching
            filter_conditions = query.build_filter() if query.filter_conditions else {}
            
            # Add text content filtering
            if filter_conditions:
                filter_conditions['must'].append({
                    'key': 'text',
                    'match': {
                        'text': query.query_text
                    }
                })
            else:
                filter_conditions = {
                    'must': [{
                        'key': 'text',
                        'match': {'text': query.query_text}
                    }]
                }
            
            # For sparse retrieval, we use a dummy vector (zeros)
            dummy_vector = np.zeros(self.embedding_generator.get_embedding_dimension())
            
            vector_results = await self.vector_store.search(
                query_vector=dummy_vector,
                top_k=self.max_retrieval_depth,
                filter_conditions=filter_conditions,
                score_threshold=0.5
            )
            
            # Convert to SearchResult objects
            results = []
            for rank, result in enumerate(vector_results):
                document = VectorDocument.from_dict(result)
                search_result = SearchResult(
                    document=document,
                    score=result.get('score', 0.0),
                    rank=rank,
                    retrieval_method='sparse'
                )
                results.append(search_result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in sparse retrieval: {str(e)}")
            return []
    
    async def _hybrid_retrieval(self, query: SearchQuery) -> List[SearchResult]:
        """
        Perform hybrid search combining dense and sparse methods.
        
        This method combines vector similarity and keyword matching with
        a configurable weight (alpha) to balance semantic and exact matching.
        
        Args:
            query: SearchQuery with hybrid_alpha parameter
            
        Returns:
            Combined and weighted search results
        """
        try:
            # Get dense results
            dense_results = await self._dense_retrieval(query)
            
            # Get sparse results
            sparse_results = await self._sparse_retrieval(query)
            
            # Combine results with weighted scoring
            combined = {}
            
            # Process dense results with alpha weight
            for result in dense_results:
                combined[result.document.id] = {
                    'result': result,
                    'dense_score': result.score,
                    'sparse_score': 0.0,
                    'final_score': query.hybrid_alpha * result.score
                }
            
            # Process sparse results with (1-alpha) weight
            for result in sparse_results:
                if result.document.id in combined:
                    combined[result.document.id]['sparse_score'] = result.score
                    combined[result.document.id]['final_score'] += (
                        (1 - query.hybrid_alpha) * result.score
                    )
                else:
                    combined[result.document.id] = {
                        'result': result,
                        'dense_score': 0.0,
                        'sparse_score': result.score,
                        'final_score': (1 - query.hybrid_alpha) * result.score
                    }
            
            # Convert to list and sort by final score
            hybrid_results = []
            for item in combined.values():
                result = item['result']
                result.score = item['final_score']
                result.retrieval_method = 'hybrid'
                hybrid_results.append(result)
            
            hybrid_results.sort(key=lambda x: x.score, reverse=True)
            
            return hybrid_results
            
        except Exception as e:
            logger.error(f"Error in hybrid retrieval: {str(e)}")
            return []
    
    async def _cross_encoder_retrieval(self, query: SearchQuery) -> List[SearchResult]:
        """
        Perform retrieval with cross-encoder reranking.
        
        This method first retrieves candidate documents with dense search,
        then uses a cross-encoder model to compute precise relevance scores.
        
        Args:
            query: SearchQuery with query parameters
            
        Returns:
            Reranked search results
        """
        try:
            # First get candidate documents with dense retrieval
            candidate_results = await self._dense_retrieval(query)
            
            # Rerank with cross-encoder
            if self.cross_encoder_model and candidate_results:
                reranked = await self._rerank_with_cross_encoder(
                    query.query_text,
                    candidate_results,
                    len(candidate_results)
                )
                return reranked
            
            return candidate_results
            
        except Exception as e:
            logger.error(f"Error in cross-encoder retrieval: {str(e)}")
            return []
    
    async def _rerank_with_cross_encoder(
        self,
        query: str,
        results: List[SearchResult],
        top_k: int
    ) -> List[SearchResult]:
        """
        Rerank search results using a cross-encoder model.
        
        Cross-encoders provide more accurate relevance scoring by processing
        query and document together, but are computationally expensive.
        
        Args:
            query: Original query text
            results: Initial search results to rerank
            top_k: Number of results to keep after reranking
            
        Returns:
            Reranked results with updated scores
        """
        try:
            if not self.cross_encoder_model:
                return results
            
            # Prepare pairs for cross-encoder
            pairs = [(query, result.document.text) for result in results]
            
            # Get cross-encoder scores
            # This is a placeholder - actual implementation depends on the model
            # For example, with sentence-transformers cross-encoder:
            # scores = self.cross_encoder_model.predict(pairs)
            
            # Placeholder: simulate scores
            scores = [result.score * 1.1 for result in results]
            
            # Update results with new scores
            for result, score in zip(results, scores):
                result.score = score
                result.retrieval_method = 'cross_encoder_reranked'
            
            # Sort by new scores
            results.sort(key=lambda x: x.score, reverse=True)
            
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Error in cross-encoder reranking: {str(e)}")
            return results
    
    async def retrieve_by_medical_concept(
        self,
        concept: str,
        top_k: int = 10,
        specialty: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Specialized retrieval for medical concepts.
        
        This method is optimized for retrieving information about specific
        medical concepts, diseases, or conditions.
        
        Args:
            concept: Medical concept to retrieve information about
            top_k: Number of results to return
            specialty: Optional medical specialty to filter by
            
        Returns:
            List of documents about the medical concept
        """
        # Create specialized query
        query = SearchQuery(
            query_text=f"medical information about {concept}",
            top_k=top_k,
            retrieval_strategy=RetrievalStrategy.HYBRID,
            hybrid_alpha=0.6,  # Slight preference for semantic search
            specialty_filter=specialty,
            categories=[DocumentCategory.MEDICAL_TEXTBOOK, DocumentCategory.CLINICAL_GUIDELINE],
            min_score=0.6
        )
        
        # Retrieve results
        results = await self.retrieve(query)
        
        logger.info(f"Retrieved {len(results)} documents for concept '{concept}'")
        
        return results
    
    async def retrieve_clinical_guidelines(
        self,
        condition: str,
        top_k: int = 5
    ) -> List[SearchResult]:
        """
        Specialized retrieval for clinical practice guidelines.
        
        This method focuses on retrieving authoritative clinical guidelines
        for specific conditions.
        
        Args:
            condition: Medical condition to find guidelines for
            top_k: Number of guidelines to return
            
        Returns:
            List of clinical guideline documents
        """
        # Create specialized query for guidelines
        query = SearchQuery(
            query_text=f"clinical practice guidelines {condition}",
            top_k=top_k,
            retrieval_strategy=RetrievalStrategy.HYBRID,
            hybrid_alpha=0.7,
            categories=[DocumentCategory.CLINICAL_GUIDELINE],
            sources=[
                DocumentSource.WHO,
                DocumentSource.CDC,
                DocumentSource.SPECIALTY_SOCIETY
            ],
            min_score=0.65
        )
        
        results = await self.retrieve(query)
        
        logger.info(f"Retrieved {len(results)} guidelines for condition '{condition}'")
        
        return results
    
    async def retrieve_drug_information(
        self,
        drug_name: str,
        top_k: int = 10
    ) -> List[SearchResult]:
        """
        Specialized retrieval for pharmaceutical information.
        
        This method retrieves information about medications including
        indications, contraindications, and interactions.
        
        Args:
            drug_name: Name of the drug to look up
            top_k: Number of results to return
            
        Returns:
            List of documents about the drug
        """
        # Create specialized query for drug information
        query = SearchQuery(
            query_text=f"{drug_name} medication information indications contraindications side effects",
            top_k=top_k,
            retrieval_strategy=RetrievalStrategy.HYBRID,
            hybrid_alpha=0.5,
            categories=[DocumentCategory.DRUG_INFORMATION],
            min_score=0.7
        )
        
        results = await self.retrieve(query)
        
        logger.info(f"Retrieved {len(results)} documents for drug '{drug_name}'")
        
        return results