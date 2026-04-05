"""
Hybrid Retriever Module

This module implements sophisticated hybrid retrieval strategies that combine
multiple retrieval methods for optimal results. It supports weighted combinations
of dense and sparse retrieval, reciprocal rank fusion, and adaptive strategy
selection based on query characteristics.

The hybrid retriever is the primary retrieval interface for the RAG system,
providing the best balance of recall and precision for medical queries.
"""

import logging
import asyncio
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from collections import defaultdict

# Import from local modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vector_store.schema import (
    VectorDocument, SearchQuery, SearchResult,
    RetrievalStrategy, DocumentCategory
)
from retrievers.medical_retriever import MedicalRetriever
from retrievers.clinical_retriever import ClinicalRetriever

# Configure logging
logger = logging.getLogger(__name__)


class FusionMethod(Enum):
    """
    Methods for fusing multiple retrieval results.
    
    WEIGHTED_AVERAGE: Combine scores with configurable weights
    RECIPROCAL_RANK: Combine based on rank positions (RRF)
    COMBINED_SCORE: Simple sum or max of scores
    LEARNED_FUSION: ML-based fusion (requires training)
    """
    WEIGHTED_AVERAGE = "weighted_average"
    RECIPROCAL_RANK = "reciprocal_rank"
    COMBINED_SCORE = "combined_score"
    LEARNED_FUSION = "learned_fusion"


class QueryType(Enum):
    """
    Classification of query types for strategy selection.
    
    This helps choose the optimal retrieval strategy based on query characteristics.
    """
    CLINICAL_DECISION = "clinical_decision"
    DRUG_INFORMATION = "drug_information"
    GENERAL_MEDICAL = "general_medical"
    OPERATIONAL = "operational"
    RESEARCH = "research"
    PATIENT_SPECIFIC = "patient_specific"
    GUIDELINE = "guideline"


class HybridRetriever:
    """
    Advanced hybrid retriever combining multiple retrieval strategies.
    
    This class implements state-of-the-art hybrid retrieval techniques:
    - Multiple retriever ensembling
    - Query type classification for strategy selection
    - Adaptive fusion based on query characteristics
    - Reciprocal rank fusion for combining results
    
    Attributes:
        medical_retriever: MedicalRetriever for general medical content
        clinical_retriever: ClinicalRetriever for patient-specific content
        dense_weight: Weight for dense retrieval in weighted average
        sparse_weight: Weight for sparse retrieval in weighted average
        fusion_method: Method to use for fusing results
        adaptive_fusion: Whether to adapt fusion weights per query
        query_classifier: Optional classifier for query type detection
    """
    
    def __init__(
        self,
        medical_retriever: MedicalRetriever,
        clinical_retriever: Optional[ClinicalRetriever] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the hybrid retriever.
        
        Args:
            medical_retriever: MedicalRetriever instance
            clinical_retriever: Optional ClinicalRetriever instance
            config: Configuration dictionary
        """
        self.medical_retriever = medical_retriever
        self.clinical_retriever = clinical_retriever
        self.config = config or {}
        
        # Fusion parameters
        self.dense_weight = self.config.get('dense_weight', 0.6)
        self.sparse_weight = self.config.get('sparse_weight', 0.4)
        self.fusion_method = FusionMethod(
            self.config.get('fusion_method', 'reciprocal_rank')
        )
        self.adaptive_fusion = self.config.get('adaptive_fusion', True)
        
        # Reciprocal Rank Fusion parameters
        self.rrf_k = self.config.get('rrf_k', 60)  # Constant for RRF
        self.max_rank = self.config.get('max_rank', 100)
        
        # Query classification
        self.query_classifier = self._init_query_classifier()
        
        logger.info(f"HybridRetriever initialized with fusion method: {self.fusion_method.value}")
    
    def _init_query_classifier(self) -> Dict[str, List[str]]:
        """
        Initialize simple rule-based query classifier.
        
        Returns:
            Dictionary mapping query types to keyword lists
        """
        return {
            QueryType.CLINICAL_DECISION: [
                'diagnosis', 'treatment', 'therapy', 'management', 'care',
                'recommend', 'guideline', 'protocol', 'algorithm'
            ],
            QueryType.DRUG_INFORMATION: [
                'drug', 'medication', 'medicine', 'pharmaceutical', 'dosing',
                'dosage', 'interaction', 'contraindication', 'side effect'
            ],
            QueryType.GENERAL_MEDICAL: [
                'what is', 'explain', 'define', 'describe', 'overview',
                'information about', 'tell me about'
            ],
            QueryType.OPERATIONAL: [
                'bed', 'capacity', 'staff', 'schedule', 'resource',
                'occupancy', 'wait time', 'throughput'
            ],
            QueryType.RESEARCH: [
                'research', 'study', 'paper', 'literature', 'evidence',
                'meta-analysis', 'systematic review', 'trial'
            ],
            QueryType.PATIENT_SPECIFIC: [
                'patient', 'mr', 'vital', 'lab', 'result', 'history',
                'this patient', 'our patient'
            ],
            QueryType.GUIDELINE: [
                'guideline', 'protocol', 'standard', 'best practice',
                'recommendation', 'consensus'
            ]
        }
    
    def classify_query(self, query_text: str) -> QueryType:
        """
        Classify the type of query for strategy selection.
        
        Args:
            query_text: The query text to classify
            
        Returns:
            QueryType classification
        """
        query_lower = query_text.lower()
        scores = defaultdict(float)
        
        for qtype, keywords in self.query_classifier.items():
            for keyword in keywords:
                if keyword in query_lower:
                    scores[qtype] += 1
        
        if not scores:
            return QueryType.GENERAL_MEDICAL
        
        return max(scores.items(), key=lambda x: x[1])[0]
    
    async def retrieve(
        self,
        query: SearchQuery,
        use_clinical_context: bool = True
    ) -> List[SearchResult]:
        """
        Perform hybrid retrieval using multiple strategies.
        
        This is the main retrieval method that orchestrates the hybrid process.
        
        Args:
            query: SearchQuery with retrieval parameters
            use_clinical_context: Whether to use clinical retriever if available
            
        Returns:
            Fused search results
        """
        # Classify query for strategy adaptation
        query_type = self.classify_query(query.query_text)
        logger.debug(f"Query classified as: {query_type.value}")
        
        # Adapt fusion weights based on query type
        if self.adaptive_fusion:
            self._adapt_weights_for_query_type(query_type)
        
        # Determine which retrievers to use
        retrievers = [self.medical_retriever]
        if use_clinical_context and self.clinical_retriever:
            retrievers.append(self.clinical_retriever)
        
        # Execute all retrievers in parallel
        retrieval_tasks = []
        for retriever in retrievers:
            retrieval_tasks.append(
                self._execute_retriever(retriever, query)
            )
        
        # Gather results
        all_results = await asyncio.gather(*retrieval_tasks)
        
        # Flatten results
        all_results = [r for sublist in all_results for r in sublist]
        
        # Apply fusion based on selected method
        if self.fusion_method == FusionMethod.WEIGHTED_AVERAGE:
            fused_results = self._fuse_weighted_average(all_results)
        elif self.fusion_method == FusionMethod.RECIPROCAL_RANK:
            fused_results = self._fuse_reciprocal_rank(all_results)
        elif self.fusion_method == FusionMethod.COMBINED_SCORE:
            fused_results = self._fuse_combined_score(all_results)
        else:
            logger.warning(f"Unknown fusion method, using reciprocal rank")
            fused_results = self._fuse_reciprocal_rank(all_results)
        
        # Apply score threshold
        fused_results = [r for r in fused_results if r.score >= query.min_score]
        
        # Limit to top_k
        fused_results = fused_results[:query.top_k]
        
        logger.info(
            f"Hybrid retrieval: fused {len(all_results)} results into {len(fused_results)} "
            f"with {self.fusion_method.value}"
        )
        
        return fused_results
    
    async def _execute_retriever(
        self,
        retriever,
        query: SearchQuery
    ) -> List[SearchResult]:
        """
        Execute a single retriever with appropriate strategy.
        
        Args:
            retriever: Retriever instance to execute
            query: SearchQuery parameters
            
        Returns:
            List of search results
        """
        try:
            # Use dense strategy for clinical retrievers
            if isinstance(retriever, ClinicalRetriever):
                query_copy = SearchQuery(
                    query_text=query.query_text,
                    query_vector=query.query_vector,
                    top_k=query.top_k * 2,
                    retrieval_strategy=RetrievalStrategy.DENSE,
                    min_score=query.min_score
                )
                results = await retriever.base_retriever.retrieve(query_copy)
            else:
                results = await retriever.retrieve(query)
            
            return results
        except Exception as e:
            logger.error(f"Error executing retriever {type(retriever)}: {str(e)}")
            return []
    
    def _fuse_reciprocal_rank(
        self,
        results: List[SearchResult]
    ) -> List[SearchResult]:
        """
        Fuse results using Reciprocal Rank Fusion (RRF).
        
        RRF combines rankings by giving higher scores to documents that
        appear in high positions across multiple result sets.
        
        Args:
            results: Combined list of search results
            
        Returns:
            Fused results with RRF scores
        """
        # Group results by document ID
        doc_groups = defaultdict(list)
        for result in results:
            doc_groups[result.document.id].append(result)
        
        # Compute RRF score for each document
        fused_scores = {}
        for doc_id, doc_results in doc_groups.items():
            rrf_score = 0.0
            for result in doc_results:
                # RRF formula: sum(1 / (k + rank))
                rank = result.rank + 1  # Convert to 1-indexed
                rrf_score += 1.0 / (self.rrf_k + rank)
            
            fused_scores[doc_id] = {
                'score': rrf_score,
                'document': doc_results[0].document,
                'best_score': max(r.score for r in doc_results),
                'appearances': len(doc_results)
            }
        
        # Create fused results sorted by RRF score
        fused_results = []
        for idx, (doc_id, data) in enumerate(
            sorted(fused_scores.items(), key=lambda x: x[1]['score'], reverse=True)
        ):
            fused_results.append(SearchResult(
                document=data['document'],
                score=data['score'],
                rank=idx,
                retrieval_method=f'rrf_{self.fusion_method.value}'
            ))
        
        return fused_results
    
    def _fuse_weighted_average(
        self,
        results: List[SearchResult]
    ) -> List[SearchResult]:
        """
        Fuse results using weighted average of scores.
        
        Args:
            results: Combined list of search results
            
        Returns:
            Fused results with weighted average scores
        """
        # Group results by document ID
        doc_groups = defaultdict(list)
        for result in results:
            doc_groups[result.document.id].append(result)
        
        # Compute weighted average for each document
        fused_scores = {}
        for doc_id, doc_results in doc_groups.items():
            # Determine weights based on retrieval method
            total_weighted_score = 0.0
            total_weight = 0.0
            
            for result in doc_results:
                if result.retrieval_method == 'dense':
                    weight = self.dense_weight
                elif result.retrieval_method == 'sparse':
                    weight = self.sparse_weight
                else:
                    weight = 0.5  # Default weight
                
                total_weighted_score += result.score * weight
                total_weight += weight
            
            if total_weight > 0:
                final_score = total_weighted_score / total_weight
            else:
                final_score = 0.0
            
            fused_scores[doc_id] = {
                'score': final_score,
                'document': doc_results[0].document
            }
        
        # Create fused results sorted by final score
        fused_results = []
        for idx, (doc_id, data) in enumerate(
            sorted(fused_scores.items(), key=lambda x: x[1]['score'], reverse=True)
        ):
            fused_results.append(SearchResult(
                document=data['document'],
                score=data['score'],
                rank=idx,
                retrieval_method=f'weighted_{self.fusion_method.value}'
            ))
        
        return fused_results
    
    def _fuse_combined_score(
        self,
        results: List[SearchResult]
    ) -> List[SearchResult]:
        """
        Fuse results using simple score combination (max or sum).
        
        Args:
            results: Combined list of search results
            
        Returns:
            Fused results with combined scores
        """
        # Group results by document ID
        doc_groups = defaultdict(list)
        for result in results:
            doc_groups[result.document.id].append(result)
        
        # Compute combined score
        fusion_method = self.config.get('combined_method', 'max')
        
        fused_scores = {}
        for doc_id, doc_results in doc_groups.items():
            if fusion_method == 'max':
                final_score = max(r.score for r in doc_results)
            elif fusion_method == 'sum':
                final_score = sum(r.score for r in doc_results)
            else:
                final_score = max(r.score for r in doc_results)
            
            fused_scores[doc_id] = {
                'score': final_score,
                'document': doc_results[0].document
            }
        
        # Create fused results
        fused_results = []
        for idx, (doc_id, data) in enumerate(
            sorted(fused_scores.items(), key=lambda x: x[1]['score'], reverse=True)
        ):
            fused_results.append(SearchResult(
                document=data['document'],
                score=data['score'],
                rank=idx,
                retrieval_method=f'combined_{fusion_method}'
            ))
        
        return fused_results
    
    def _adapt_weights_for_query_type(self, query_type: QueryType):
        """
        Adapt fusion weights based on query type.
        
        Different query types benefit from different retrieval balances.
        
        Args:
            query_type: Classification of the query
        """
        if query_type == QueryType.CLINICAL_DECISION:
            self.dense_weight = 0.7
            self.sparse_weight = 0.3
        elif query_type == QueryType.DRUG_INFORMATION:
            self.dense_weight = 0.5
            self.sparse_weight = 0.5
        elif query_type == QueryType.PATIENT_SPECIFIC:
            self.dense_weight = 0.8
            self.sparse_weight = 0.2
        elif query_type == QueryType.RESEARCH:
            self.dense_weight = 0.6
            self.sparse_weight = 0.4
        elif query_type == QueryType.OPERATIONAL:
            self.dense_weight = 0.4
            self.sparse_weight = 0.6
        else:
            self.dense_weight = 0.6
            self.sparse_weight = 0.4
        
        logger.debug(f"Adapted weights: dense={self.dense_weight}, sparse={self.sparse_weight}")
    
    async def batch_retrieve(
        self,
        queries: List[SearchQuery],
        use_clinical_context: bool = True
    ) -> List[List[SearchResult]]:
        """
        Process multiple queries in batch for efficiency.
        
        Args:
            queries: List of SearchQuery objects
            use_clinical_context: Whether to use clinical retriever
            
        Returns:
            List of result lists for each query
        """
        tasks = [self.retrieve(q, use_clinical_context) for q in queries]
        results = await asyncio.gather(*tasks)
        
        logger.info(f"Batch retrieval completed for {len(queries)} queries")
        
        return results
    
    async def retrieve_with_feedback(
        self,
        query: SearchQuery,
        feedback_results: List[SearchResult],
        feedback_weights: List[float]
    ) -> List[SearchResult]:
        """
        Perform retrieval with relevance feedback.
        
        This method uses user feedback to adjust the retrieval strategy
        for subsequent queries.
        
        Args:
            query: Original search query
            feedback_results: Results marked as relevant by user
            feedback_weights: Importance weights for feedback
            
        Returns:
            Improved search results incorporating feedback
        """
        # Extract feedback documents
        feedback_docs = [r.document for r in feedback_results]
        
        # Create query expansion from feedback
        feedback_terms = []
        for doc in feedback_docs[:3]:
            # Extract key terms from feedback documents
            words = doc.text.split()[:50]  # First 50 words
            feedback_terms.extend(words)
        
        # Expand original query with feedback terms
        expanded_query_text = query.query_text + ' ' + ' '.join(feedback_terms[:10])
        
        # Create new query with expansion
        expanded_query = SearchQuery(
            query_text=expanded_query_text,
            query_vector=None,  # Will be regenerated
            top_k=query.top_k,
            retrieval_strategy=query.retrieval_strategy,
            hybrid_alpha=query.hybrid_alpha,
            min_score=query.min_score
        )
        
        # Execute expanded query
        results = await self.retrieve(expanded_query)
        
        logger.info(f"Retrieved {len(results)} results with relevance feedback")
        
        return results  