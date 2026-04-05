"""
Clinical Retriever Module

This module implements specialized retrieval for clinical decision support.
It focuses on retrieving patient-specific information, clinical protocols,
and evidence-based recommendations for direct clinical use.

The clinical retriever integrates with EHR systems and patient data to provide
context-aware retrieval that considers the specific patient's condition,
demographics, and clinical history.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np

# Import from local modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vector_store.schema import (
    VectorDocument, SearchQuery, SearchResult,
    RetrievalStrategy, DocumentCategory, DocumentSource
)
from retrievers.medical_retriever import MedicalRetriever

# Configure logging
logger = logging.getLogger(__name__)


class ClinicalRetriever:
    """
    Specialized retriever for clinical decision support.
    
    This class extends the base medical retriever with clinical-specific
    functionality including:
    - Patient context integration
    - Clinical protocol retrieval
    - Evidence-based medicine ranking
    - Medical specialty-specific filtering
    - Clinical trial matching
    
    Attributes:
        base_retriever: MedicalRetriever instance for base retrieval
        patient_context: Optional patient data to contextualize retrieval
        clinical_priority_weights: Weights for different clinical evidence types
        protocol_cache: Cache for frequently accessed clinical protocols
    """
    
    def __init__(
        self,
        base_retriever: MedicalRetriever,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the clinical retriever.
        
        Args:
            base_retriever: Initialized MedicalRetriever instance
            config: Configuration dictionary for clinical retrieval
        """
        self.base_retriever = base_retriever
        self.config = config or {}
        self.patient_context = None
        
        # Clinical evidence priority weights (higher = more important)
        # These reflect the hierarchy of clinical evidence
        self.evidence_weights = {
            'meta_analysis': 1.0,
            'systematic_review': 0.95,
            'randomized_controlled_trial': 0.9,
            'cohort_study': 0.7,
            'case_control': 0.6,
            'case_series': 0.4,
            'expert_opinion': 0.3,
            'clinical_guideline': 0.85,
            'protocol': 0.8
        }
        
        # Cache for frequently accessed protocols
        self.protocol_cache = {}
        
        logger.info("ClinicalRetriever initialized successfully")
    
    def set_patient_context(self, patient_data: Dict[str, Any]) -> None:
        """
        Set patient context for personalized clinical retrieval.
        
        This method stores patient information to be used in retrieval
        for generating patient-specific recommendations.
        
        Args:
            patient_data: Dictionary containing patient information
                - age: Patient age in years
                - gender: Patient gender
                - conditions: List of medical conditions
                - medications: List of current medications
                - allergies: List of allergies
                - lab_results: Recent laboratory results
                - vitals: Recent vital signs
        """
        self.patient_context = patient_data
        logger.info(f"Patient context set for patient ID: {patient_data.get('id', 'unknown')}")
    
    def clear_patient_context(self) -> None:
        """
        Clear the current patient context.
        
        This should be called after completing a clinical interaction
        to prevent context leakage between patients.
        """
        self.patient_context = None
        logger.info("Patient context cleared")
    
    async def retrieve_clinical_recommendation(
        self,
        condition: str,
        patient_specific: bool = True,
        top_k: int = 5
    ) -> List[SearchResult]:
        """
        Retrieve clinical recommendations for a condition.
        
        This method combines general medical knowledge with patient-specific
        context to generate personalized recommendations.
        
        Args:
            condition: Medical condition to get recommendations for
            patient_specific: Whether to incorporate patient context
            top_k: Number of recommendations to return
            
        Returns:
            List of clinical recommendation documents
        """
        # Build query with patient context if available
        query_text = f"clinical recommendations treatment guidelines for {condition}"
        
        if patient_specific and self.patient_context:
            # Add patient-specific modifiers to query
            age = self.patient_context.get('age')
            if age:
                if age < 18:
                    query_text += " pediatric"
                elif age > 65:
                    query_text += " geriatric"
            
            gender = self.patient_context.get('gender')
            if gender:
                query_text += f" {gender.lower()}"
            
            # Add specific conditions if present
            conditions = self.patient_context.get('conditions', [])
            if conditions:
                comorbidities = ', '.join(conditions[:2])
                query_text += f" with comorbidities {comorbidities}"
        
        # Create specialized query
        query = SearchQuery(
            query_text=query_text,
            top_k=top_k * 2,  # Retrieve extra for filtering
            retrieval_strategy=RetrievalStrategy.HYBRID,
            hybrid_alpha=0.7,
            categories=[
                DocumentCategory.CLINICAL_GUIDELINE,
                DocumentCategory.MEDICAL_TEXTBOOK
            ],
            min_score=0.7
        )
        
        # Retrieve base results
        results = await self.base_retriever.retrieve(query)
        
        # Apply clinical evidence weighting
        results = self._apply_evidence_weighting(results)
        
        # Filter and rank for patient-specificity
        if patient_specific and self.patient_context:
            results = await self._filter_for_patient_context(results)
        
        # Limit to requested number
        results = results[:top_k]
        
        logger.info(
            f"Retrieved {len(results)} clinical recommendations for condition '{condition}'"
        )
        
        return results
    
    async def retrieve_clinical_protocol(
        self,
        protocol_type: str,
        department: Optional[str] = None
    ) -> Optional[SearchResult]:
        """
        Retrieve a specific clinical protocol.
        
        This method is optimized for retrieving exact clinical protocols
        by name or type, with caching for frequently accessed protocols.
        
        Args:
            protocol_type: Type of protocol (e.g., 'sepsis', 'stroke', 'mi')
            department: Optional department to filter protocols
            
        Returns:
            Clinical protocol document if found, None otherwise
        """
        # Check cache first
        cache_key = f"{protocol_type}_{department or 'all'}"
        if cache_key in self.protocol_cache:
            cached = self.protocol_cache[cache_key]
            # Check if cache is still valid (less than 24 hours old)
            if (datetime.now() - cached['timestamp']).total_seconds() < 86400:
                logger.info(f"Retrieved protocol '{protocol_type}' from cache")
                return cached['result']
        
        # Build query for protocol retrieval
        query = SearchQuery(
            query_text=f"{protocol_type} clinical protocol procedure algorithm",
            top_k=3,
            retrieval_strategy=RetrievalStrategy.HYBRID,
            hybrid_alpha=0.6,
            categories=[DocumentCategory.CLINICAL_GUIDELINE],
            department_filter=department,
            min_score=0.75
        )
        
        # Retrieve results
        results = await self.base_retriever.retrieve(query)
        
        # Find the most relevant protocol
        best_result = None
        best_score = 0
        
        for result in results:
            # Check if result is actually a protocol
            text = result.document.text.lower()
            if any(keyword in text for keyword in ['protocol', 'algorithm', 'procedure']):
                if result.score > best_score:
                    best_score = result.score
                    best_result = result
        
        # Cache the result if found
        if best_result:
            self.protocol_cache[cache_key] = {
                'result': best_result,
                'timestamp': datetime.now()
            }
            logger.info(f"Cached protocol '{protocol_type}'")
        
        return best_result
    
    async def retrieve_medication_guidance(
        self,
        medication: str,
        condition: Optional[str] = None,
        patient_specific: bool = True
    ) -> List[SearchResult]:
        """
        Retrieve medication guidance including dosing, interactions, and monitoring.
        
        Args:
            medication: Name of the medication
            condition: Optional condition being treated
            patient_specific: Whether to incorporate patient context
            
        Returns:
            List of medication guidance documents
        """
        # Build query text
        query_parts = [medication, "medication guidance dosing monitoring"]
        if condition:
            query_parts.append(f"for {condition}")
        
        query_text = ' '.join(query_parts)
        
        # Add patient-specific factors
        if patient_specific and self.patient_context:
            age = self.patient_context.get('age')
            if age:
                if age < 18:
                    query_text += " pediatric dosing"
                elif age > 65:
                    query_text += " geriatric precautions"
            
            # Check for renal function considerations
            renal_function = self.patient_context.get('renal_function')
            if renal_function and renal_function.get('egfr'):
                egfr = renal_function['egfr']
                if egfr < 30:
                    query_text += " renal dose adjustment"
            
            # Check for hepatic impairment
            if self.patient_context.get('hepatic_impairment'):
                query_text += " hepatic impairment"
        
        # Create query
        query = SearchQuery(
            query_text=query_text,
            top_k=10,
            retrieval_strategy=RetrievalStrategy.HYBRID,
            hybrid_alpha=0.6,
            categories=[DocumentCategory.DRUG_INFORMATION],
            min_score=0.7
        )
        
        results = await self.base_retriever.retrieve(query)
        
        logger.info(f"Retrieved {len(results)} medication guidance documents for '{medication}'")
        
        return results
    
    async def retrieve_lab_interpretation(
        self,
        lab_test: str,
        result_value: float,
        unit: str,
        reference_range: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Retrieve interpretation guidance for laboratory results.
        
        Args:
            lab_test: Name of the laboratory test
            result_value: The result value
            unit: Unit of measurement
            reference_range: Optional reference range string
            
        Returns:
            Dictionary with interpretation and clinical guidance
        """
        # Build query for lab interpretation
        query_text = f"{lab_test} interpretation clinical significance abnormal findings"
        
        query = SearchQuery(
            query_text=query_text,
            top_k=5,
            retrieval_strategy=RetrievalStrategy.HYBRID,
            hybrid_alpha=0.8,
            min_score=0.7
        )
        
        results = await self.base_retriever.retrieve(query)
        
        # Synthesize interpretation
        interpretation = {
            'test': lab_test,
            'value': result_value,
            'unit': unit,
            'reference_range': reference_range,
            'is_abnormal': False,
            'severity': 'normal',
            'clinical_implications': [],
            'follow_up_recommendations': [],
            'references': []
        }
        
        # Check if result is abnormal
        if reference_range:
            try:
                # Parse reference range (e.g., "10-20 mg/dL")
                import re
                range_match = re.search(r'(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)', reference_range)
                if range_match:
                    low = float(range_match.group(1))
                    high = float(range_match.group(2))
                    
                    if result_value < low:
                        interpretation['is_abnormal'] = True
                        interpretation['severity'] = 'low'
                    elif result_value > high:
                        interpretation['is_abnormal'] = True
                        interpretation['severity'] = 'high'
            except Exception as e:
                logger.warning(f"Could not parse reference range: {reference_range}")
        
        # Extract clinical implications from retrieved documents
        for result in results[:3]:
            text = result.document.text
            if 'implication' in text.lower() or 'significance' in text.lower():
                # Extract relevant sentences (simplified)
                sentences = text.split('.')
                for sentence in sentences:
                    if 'implication' in sentence.lower() or 'significance' in sentence.lower():
                        interpretation['clinical_implications'].append(sentence.strip())
            
            interpretation['references'].append({
                'source': result.document.metadata.source,
                'score': result.score
            })
        
        return interpretation
    
    async def retrieve_evidence_summary(
        self,
        topic: str,
        evidence_level: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Retrieve and synthesize evidence summary for a clinical topic.
        
        This method retrieves the highest quality evidence available
        and synthesizes it into a structured summary.
        
        Args:
            topic: Clinical topic to summarize
            evidence_level: Minimum evidence level to include
            
        Returns:
            Structured evidence summary with recommendations
        """
        # Build query for evidence retrieval
        query = SearchQuery(
            query_text=f"{topic} evidence systematic review meta-analysis clinical trial",
            top_k=15,
            retrieval_strategy=RetrievalStrategy.HYBRID,
            hybrid_alpha=0.7,
            categories=[DocumentCategory.LITERATURE, DocumentCategory.CLINICAL_GUIDELINE],
            min_score=0.6
        )
        
        results = await self.base_retriever.retrieve(query)
        
        # Apply evidence weighting
        results = self._apply_evidence_weighting(results)
        
        # Group by evidence type
        evidence_by_type = {
            'meta_analysis': [],
            'systematic_review': [],
            'randomized_controlled_trial': [],
            'cohort_study': [],
            'guideline': [],
            'other': []
        }
        
        for result in results:
            # Determine evidence type from metadata or content
            evidence_type = self._determine_evidence_type(result)
            if evidence_type in evidence_by_type:
                evidence_by_type[evidence_type].append(result)
        
        # Synthesize summary
        summary = {
            'topic': topic,
            'total_evidence_sources': len(results),
            'highest_evidence_level': self._get_highest_evidence_level(evidence_by_type),
            'key_findings': [],
            'recommendations': [],
            'evidence_breakdown': {
                k: len(v) for k, v in evidence_by_type.items()
            },
            'references': []
        }
        
        # Extract key findings from top evidence
        top_results = results[:5]
        for result in top_results:
            # Extract key sentences (simplified)
            text = result.document.text
            sentences = text.split('.')
            for sentence in sentences[:3]:  # First few sentences
                if len(sentence) > 50 and len(sentence) < 300:
                    if 'found' in sentence.lower() or 'show' in sentence.lower() or 'demonstrate' in sentence.lower():
                        summary['key_findings'].append(sentence.strip())
            
            summary['references'].append({
                'title': result.document.metadata.get('title', 'Unknown'),
                'authors': result.document.metadata.get('author', 'Unknown'),
                'year': result.document.metadata.get('publication_date', 'Unknown')[:4],
                'evidence_type': self._determine_evidence_type(result)
            })
        
        # Deduplicate key findings
        summary['key_findings'] = list(dict.fromkeys(summary['key_findings']))[:5]
        
        return summary
    
    def _apply_evidence_weighting(
        self,
        results: List[SearchResult]
    ) -> List[SearchResult]:
        """
        Apply weights based on evidence level.
        
        Args:
            results: Search results to weight
            
        Returns:
            Weighted and re-sorted results
        """
        for result in results:
            evidence_type = self._determine_evidence_type(result)
            weight = self.evidence_weights.get(evidence_type, 0.5)
            result.score = result.score * weight
        
        # Re-sort by weighted score
        results.sort(key=lambda x: x.score, reverse=True)
        
        return results
    
    def _determine_evidence_type(self, result: SearchResult) -> str:
        """
        Determine the evidence level of a document.
        
        Args:
            result: SearchResult to classify
            
        Returns:
            String identifying the evidence type
        """
        text = result.document.text.lower()
        metadata = result.document.metadata
        
        # Check metadata first
        source = metadata.get('source', '').lower()
        category = metadata.get('category', '').lower()
        
        if 'meta-analysis' in text or 'meta analysis' in text:
            return 'meta_analysis'
        elif 'systematic review' in text:
            return 'systematic_review'
        elif 'randomized controlled trial' in text or 'rct' in text:
            return 'randomized_controlled_trial'
        elif 'cohort study' in text or 'cohort' in text:
            return 'cohort_study'
        elif 'guideline' in text or category == 'guideline':
            return 'guideline'
        elif 'protocol' in text:
            return 'protocol'
        elif 'review' in text:
            return 'systematic_review'
        
        return 'other'
    
    def _get_highest_evidence_level(
        self,
        evidence_by_type: Dict[str, List[SearchResult]]
    ) -> str:
        """
        Get the highest evidence level present.
        
        Args:
            evidence_by_type: Dictionary grouping results by type
            
        Returns:
            String with highest evidence level
        """
        # Order of evidence quality
        evidence_priority = [
            'meta_analysis',
            'systematic_review',
            'randomized_controlled_trial',
            'guideline',
            'cohort_study',
            'protocol',
            'other'
        ]
        
        for evidence_type in evidence_priority:
            if evidence_by_type.get(evidence_type):
                return evidence_type
        
        return 'none'
    
    async def _filter_for_patient_context(
        self,
        results: List[SearchResult]
    ) -> List[SearchResult]:
        """
        Filter and re-rank results based on patient context.
        
        This method increases scores for documents that are relevant to the
        specific patient's characteristics.
        
        Args:
            results: Initial search results
            
        Returns:
            Filtered and re-ranked results
        """
        if not self.patient_context:
            return results
        
        # Define patient characteristics that affect relevance
        age = self.patient_context.get('age')
        gender = self.patient_context.get('gender')
        conditions = self.patient_context.get('conditions', [])
        
        for result in results:
            text = result.document.text.lower()
            boost = 1.0
            
            # Age-specific boosts
            if age:
                if age < 18 and any(term in text for term in ['pediatric', 'child', 'infant']):
                    boost *= 1.3
                elif age > 65 and any(term in text for term in ['geriatric', 'elderly', 'older adult']):
                    boost *= 1.3
            
            # Gender-specific boosts
            if gender and gender.lower() in text:
                boost *= 1.2
            
            # Comorbidity boosts
            for condition in conditions:
                if condition.lower() in text:
                    boost *= 1.15
            
            # Apply boost to score
            result.score = result.score * boost
        
        # Re-sort by boosted score
        results.sort(key=lambda x: x.score, reverse=True)
        
        return results