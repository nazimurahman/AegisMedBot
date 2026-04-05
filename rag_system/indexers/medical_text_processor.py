"""
Medical Text Processor Module

This module provides specialized text processing utilities for medical and
clinical documents. It handles medical terminology normalization, abbreviation
expansion, and domain-specific text cleaning.

The processor is essential for improving retrieval quality by normalizing
medical text to a consistent representation.
"""

import re
import logging
from typing import List, Dict, Set, Tuple, Optional
from pathlib import Path
import json

# Configure logging
logger = logging.getLogger(__name__)


class MedicalTextProcessor:
    """
    Specialized text processor for medical and clinical content.
    
    This class provides medical-domain text processing capabilities:
    - Medical abbreviation expansion
    - Medical term normalization
    - De-identification of PHI
    - Clinical concept extraction
    - Medical spell checking and correction
    
    Attributes:
        abbreviation_dict: Dictionary mapping abbreviations to full forms
        medical_terms: Set of common medical terms
        phi_patterns: Patterns for detecting PHI
        stopwords: Medical-specific stopwords to remove
        config: Configuration dictionary
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the medical text processor.
        
        Args:
            config: Configuration dictionary with processing parameters
        """
        self.config = config or {}
        
        # Initialize abbreviation dictionary
        self.abbreviation_dict = self._load_abbreviation_dict()
        
        # Initialize medical terms
        self.medical_terms = self._load_medical_terms()
        
        # Initialize PHI patterns
        self.phi_patterns = self._init_phi_patterns()
        
        # Medical stopwords (clinical terms that don't add semantic value)
        self.stopwords = self._init_medical_stopwords()
        
        # Processing flags
        self.expand_abbreviations = self.config.get('expand_abbreviations', True)
        self.normalize_terms = self.config.get('normalize_terms', True)
        self.detect_phi = self.config.get('detect_phi', True)
        self.remove_stopwords = self.config.get('remove_stopwords', False)
        
        logger.info("MedicalTextProcessor initialized")
    
    def _load_abbreviation_dict(self) -> Dict[str, str]:
        """
        Load medical abbreviation dictionary.
        
        This creates a dictionary of common medical abbreviations and
        their expanded forms.
        
        Returns:
            Dictionary mapping abbreviations to full forms
        """
        # Common medical abbreviations
        abbreviations = {
            # Cardiovascular
            'MI': 'myocardial infarction',
            'CAD': 'coronary artery disease',
            'HTN': 'hypertension',
            'CHF': 'congestive heart failure',
            'ACS': 'acute coronary syndrome',
            'CABG': 'coronary artery bypass graft',
            'PCI': 'percutaneous coronary intervention',
            
            # Respiratory
            'COPD': 'chronic obstructive pulmonary disease',
            'ARDS': 'acute respiratory distress syndrome',
            'PE': 'pulmonary embolism',
            'DVT': 'deep vein thrombosis',
            
            # Neurological
            'CVA': 'cerebrovascular accident',
            'TIA': 'transient ischemic attack',
            'ICP': 'intracranial pressure',
            'LOC': 'level of consciousness',
            
            # Metabolic
            'DM': 'diabetes mellitus',
            'DKA': 'diabetic ketoacidosis',
            'ESRD': 'end stage renal disease',
            'CKD': 'chronic kidney disease',
            
            # Infectious
            'UTI': 'urinary tract infection',
            'PNA': 'pneumonia',
            'HIV': 'human immunodeficiency virus',
            'AIDS': 'acquired immunodeficiency syndrome',
            
            # Laboratory
            'CBC': 'complete blood count',
            'BMP': 'basic metabolic panel',
            'CMP': 'comprehensive metabolic panel',
            'ABG': 'arterial blood gas',
            'INR': 'international normalized ratio',
            
            # Medications
            'NSAID': 'non-steroidal anti-inflammatory drug',
            'ACEI': 'angiotensin converting enzyme inhibitor',
            'ARB': 'angiotensin receptor blocker',
            'CCB': 'calcium channel blocker',
            
            # Procedures
            'EKG': 'electrocardiogram',
            'ECG': 'electrocardiogram',
            'EEG': 'electroencephalogram',
            'CT': 'computed tomography',
            'MRI': 'magnetic resonance imaging',
            'PET': 'positron emission tomography',
            
            # Units and measurements
            'BP': 'blood pressure',
            'HR': 'heart rate',
            'RR': 'respiratory rate',
            'O2': 'oxygen',
            'SpO2': 'oxygen saturation',
            
            # Miscellaneous
            'STAT': 'immediately',
            'PRN': 'as needed',
            'QD': 'once daily',
            'BID': 'twice daily',
            'TID': 'three times daily',
            'QID': 'four times daily'
        }
        
        return abbreviations
    
    def _load_medical_terms(self) -> Set[str]:
        """
        Load common medical terms.
        
        Returns:
            Set of medical terms for term normalization
        """
        # Core medical terms (sample - would be larger in production)
        medical_terms = {
            # Diseases
            'diabetes', 'hypertension', 'asthma', 'pneumonia', 'sepsis',
            'myocardial infarction', 'stroke', 'cancer', 'tumor', 'infection',
            
            # Symptoms
            'fever', 'pain', 'nausea', 'vomiting', 'dizziness', 'fatigue',
            'shortness of breath', 'chest pain', 'headache', 'cough',
            
            # Treatments
            'surgery', 'medication', 'therapy', 'procedure', 'intervention',
            'ventilator', 'intubation', 'resuscitation', 'transfusion',
            
            # Anatomy
            'heart', 'lung', 'liver', 'kidney', 'brain', 'artery', 'vein',
            
            # Diagnostics
            'biopsy', 'imaging', 'laboratory', 'test', 'scan', 'ultrasound',
            'x-ray', 'mammogram', 'colonoscopy'
        }
        
        return medical_terms
    
    def _init_phi_patterns(self) -> List[Tuple[str, str]]:
        """
        Initialize PHI (Protected Health Information) detection patterns.
        
        Returns:
            List of (pattern, replacement) tuples for PHI detection
        """
        return [
            # Names
            (r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', '[NAME]'),
            (r'\bMr\.\s+[A-Z][a-z]+\b', '[NAME]'),
            (r'\bMrs\.\s+[A-Z][a-z]+\b', '[NAME]'),
            (r'\bDr\.\s+[A-Z][a-z]+\b', '[NAME]'),
            
            # Medical Record Numbers
            (r'\b\d{3}-\d{2}-\d{4}\b', '[MRN]'),
            (r'\b\d{8,10}\b', '[MRN]'),
            
            # Dates
            (r'\b\d{1,2}/\d{1,2}/\d{4}\b', '[DATE]'),
            (r'\b\d{4}-\d{2}-\d{2}\b', '[DATE]'),
            
            # Phone Numbers
            (r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]'),
            
            # Email Addresses
            (r'\b[\w\.-]+@[\w\.-]+\.\w+\b', '[EMAIL]'),
            
            # Social Security Numbers
            (r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]'),
            
            # Addresses
            (r'\b\d+\s+[A-Z][a-z]+\s+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd)\b', '[ADDRESS]')
        ]
    
    def _init_medical_stopwords(self) -> Set[str]:
        """
        Initialize medical-specific stopwords.
        
        Returns:
            Set of words to remove that don't add semantic value
        """
        return {
            'patient', 'presented', 'presenting', 'complained', 'complains',
            'reported', 'reports', 'states', 'stated', 'noted', 'notes',
            'observed', 'observes', 'found', 'findings', 'showed', 'shows',
            'demonstrated', 'demonstrates', 'revealed', 'reveals',
            'according', 'accordingly', 'therefore', 'however',
            'additionally', 'furthermore', 'consequently'
        }
    
    def process_text(
        self,
        text: str,
        return_cleaned: bool = True
    ) -> Dict[str, Any]:
        """
        Process medical text with all configured operations.
        
        This is the main processing pipeline for medical text.
        
        Args:
            text: Raw text to process
            return_cleaned: Whether to return cleaned text
            
        Returns:
            Dictionary with processing results including cleaned text,
            detected PHI, expanded abbreviations, and extracted concepts
        """
        original_text = text
        cleaned_text = text
        detected_phi = []
        expanded_abbreviations = {}
        extracted_concepts = set()
        
        # Expand abbreviations
        if self.expand_abbreviations:
            cleaned_text, expanded = self.expand_medical_abbreviations(cleaned_text)
            expanded_abbreviations = expanded
        
        # Normalize medical terms
        if self.normalize_terms:
            cleaned_text = self.normalize_medical_terms(cleaned_text)
        
        # Detect and redact PHI
        if self.detect_phi:
            cleaned_text, phi_instances = self.redact_phi(cleaned_text)
            detected_phi = phi_instances
        
        # Extract clinical concepts
        extracted_concepts = self.extract_clinical_concepts(cleaned_text)
        
        # Remove stopwords if configured
        if self.remove_stopwords:
            cleaned_text = self.remove_medical_stopwords(cleaned_text)
        
        # Clean up extra whitespace
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        
        result = {
            'original_text': original_text if not return_cleaned else None,
            'cleaned_text': cleaned_text if return_cleaned else None,
            'detected_phi': detected_phi,
            'expanded_abbreviations': expanded_abbreviations,
            'extracted_concepts': list(extracted_concepts),
            'text_length': len(cleaned_text),
            'word_count': len(cleaned_text.split())
        }
        
        return result
    
    def expand_medical_abbreviations(
        self,
        text: str
    ) -> Tuple[str, Dict[str, str]]:
        """
        Expand medical abbreviations in text.
        
        Args:
            text: Text containing abbreviations
            
        Returns:
            Tuple of (expanded_text, expansion_dict)
        """
        expanded_text = text
        expansions = {}
        
        # Sort abbreviations by length (longest first) to avoid partial matches
        sorted_abbr = sorted(
            self.abbreviation_dict.items(),
            key=lambda x: len(x[0]),
            reverse=True
        )
        
        for abbr, full_form in sorted_abbr:
            # Match word boundaries
            pattern = r'\b' + re.escape(abbr) + r'\b'
            if re.search(pattern, expanded_text, re.IGNORECASE):
                expanded_text = re.sub(
                    pattern,
                    full_form,
                    expanded_text,
                    flags=re.IGNORECASE
                )
                expansions[abbr] = full_form
        
        return expanded_text, expansions
    
    def normalize_medical_terms(self, text: str) -> str:
        """
        Normalize medical terms to standard forms.
        
        Args:
            text: Text with medical terms
            
        Returns:
            Text with normalized medical terms
        """
        # Common term normalizations
        term_normalizations = {
            # Cardiovascular
            r'heart attack': 'myocardial infarction',
            r'high blood pressure': 'hypertension',
            r'high bp': 'hypertension',
            r'congestive heart failure': 'heart failure',
            
            # Respiratory
            r'lung infection': 'pneumonia',
            r'breathing difficulty': 'dyspnea',
            r'shortness of breath': 'dyspnea',
            
            # Metabolic
            r'sugar': 'glucose',
            r'blood sugar': 'glucose',
            
            # General
            r'fever': 'pyrexia',
            r'vomiting': 'emesis',
            r'nausea': 'emesis',
            r'pain': 'algia'
        }
        
        normalized_text = text.lower()
        
        for term, normalized in term_normalizations.items():
            normalized_text = re.sub(
                r'\b' + re.escape(term) + r'\b',
                normalized,
                normalized_text
            )
        
        return normalized_text
    
    def redact_phi(self, text: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Detect and redact PHI from text.
        
        Args:
            text: Text that may contain PHI
            
        Returns:
            Tuple of (redacted_text, detected_phi_instances)
        """
        redacted_text = text
        detected = []
        
        for pattern, replacement in self.phi_patterns:
            matches = re.finditer(pattern, redacted_text)
            for match in matches:
                detected.append({
                    'type': replacement.strip('[]'),
                    'text': match.group(),
                    'position': match.start()
                })
            
            redacted_text = re.sub(pattern, replacement, redacted_text)
        
        return redacted_text, detected
    
    def extract_clinical_concepts(self, text: str) -> Set[str]:
        """
        Extract clinical concepts from text.
        
        This identifies medical terms and concepts in the text.
        
        Args:
            text: Processed text
            
        Returns:
            Set of extracted clinical concepts
        """
        text_lower = text.lower()
        concepts = set()
        
        # Check for known medical terms
        for term in self.medical_terms:
            if term in text_lower:
                concepts.add(term)
        
        # Extract drug names (simple pattern - would use NER in production)
        drug_pattern = r'\b(?:aspirin|ibuprofen|acetaminophen|metformin|lisinopril|atorvastatin)\b'
        drug_matches = re.findall(drug_pattern, text_lower)
        concepts.update(drug_matches)
        
        # Extract lab tests (simple pattern)
        lab_pattern = r'\b(?:glucose|creatinine|potassium|sodium|hemoglobin|hematocrit)\b'
        lab_matches = re.findall(lab_pattern, text_lower)
        concepts.update(lab_matches)
        
        return concepts
    
    def remove_medical_stopwords(self, text: str) -> str:
        """
        Remove medical-specific stopwords from text.
        
        Args:
            text: Text to process
            
        Returns:
            Text with stopwords removed
        """
        words = text.split()
        filtered_words = [w for w in words if w.lower() not in self.stopwords]
        return ' '.join(filtered_words)
    
    def process_document_batch(
        self,
        documents: List[Dict[str, Any]],
        text_field: str = 'text'
    ) -> List[Dict[str, Any]]:
        """
        Process a batch of documents.
        
        Args:
            documents: List of document dictionaries
            text_field: Field containing text to process
            
        Returns:
            List of documents with processed text
        """
        processed_docs = []
        
        for doc in documents:
            if text_field in doc:
                processed = self.process_text(doc[text_field])
                doc['processed_text'] = processed['cleaned_text']
                doc['extracted_concepts'] = processed['extracted_concepts']
                doc['phi_detected'] = bool(processed['detected_phi'])
            
            processed_docs.append(doc)
        
        logger.info(f"Processed batch of {len(documents)} documents")
        
        return processed_docs
    
    def get_text_statistics(self, text: str) -> Dict[str, Any]:
        """
        Get statistical information about the text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with text statistics
        """
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        
        # Count medical terms
        medical_term_count = 0
        text_lower = text.lower()
        for term in self.medical_terms:
            if term in text_lower:
                medical_term_count += text_lower.count(term)
        
        return {
            'total_characters': len(text),
            'total_words': len(words),
            'total_sentences': len([s for s in sentences if s.strip()]),
            'avg_word_length': sum(len(w) for w in words) / len(words) if words else 0,
            'medical_term_count': medical_term_count,
            'abbreviation_count': sum(
                1 for abbr in self.abbreviation_dict
                if re.search(r'\b' + re.escape(abbr) + r'\b', text, re.IGNORECASE)
            )
        }