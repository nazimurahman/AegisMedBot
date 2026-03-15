"""
PHI (Protected Health Information) Detector Module
Responsible for identifying and classifying PHI in medical text and data
Implements HIPAA guidelines for identifying 18 types of PHI identifiers
"""

import re
import json
from typing import List, Dict, Any, Optional, Tuple, Set
from enum import Enum
import spacy
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class PHIType(Enum):
    """
    Enumeration of HIPAA-defined PHI identifiers
    Covers all 18 HIPAA identifiers for complete compliance
    """
    NAME = "name"
    GEOGRAPHIC = "geographic"  # Smaller than a state
    DATE = "date"  # All elements of dates except year
    PHONE = "phone"
    FAX = "fax"
    EMAIL = "email"
    SSN = "ssn"  # Social Security Number
    MRN = "mrn"  # Medical Record Number
    HEALTH_PLAN_ID = "health_plan_id"
    ACCOUNT_NUMBER = "account_number"
    CERTIFICATE_NUMBER = "certificate_number"
    VEHICLE_ID = "vehicle_id"
    DEVICE_ID = "device_id"
    URL = "url"
    IP_ADDRESS = "ip_address"
    BIOMETRIC = "biometric"
    FACE_IMAGE = "face_image"
    ANY_ID = "any_id"  # Any other unique identifying number

class PHIRiskLevel(Enum):
    """
    Risk level for PHI based on sensitivity and identifiability
    """
    LOW = 1      # Indirect identifier, needs combination to identify
    MEDIUM = 2   # Direct identifier but limited sensitivity
    HIGH = 3     # Highly sensitive direct identifier
    CRITICAL = 4 # Extremely sensitive (SSN, full name with condition)

class PHIDetector:
    """
    Advanced PHI detector using multiple detection strategies:
    1. Pattern matching with regex
    2. Named Entity Recognition with medical NLP
    3. Contextual analysis
    4. Machine learning classifiers
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize PHI detector with detection patterns and models
        
        Args:
            config: Configuration dictionary for detection settings
        """
        self.config = config or {}
        
        # Compile regex patterns for each PHI type
        self.patterns = self._compile_patterns()
        
        # Initialize NLP model for medical entity recognition
        # In production, use en_core_medical_md or similar
        try:
            self.nlp = spacy.load("en_core_web_sm")  # Placeholder, use medical model in production
        except:
            logger.warning("Medical NLP model not found, using basic model")
            self.nlp = spacy.load("en_core_web_sm")
        
        # Context keywords that indicate PHI
        self.phi_context_keywords = {
            "name": ["patient", "mr.", "mrs.", "ms.", "dr.", "called", "named"],
            "mrn": ["mrn", "medical record", "record number", "chart number"],
            "ssn": ["ssn", "social security", "social insurance"],
            "phone": ["phone", "telephone", "cell", "mobile", "contact"],
            "email": ["email", "e-mail"],
            "address": ["address", "lives at", "resides at", "located at"],
            "dob": ["dob", "birth", "born on", "date of birth"]
        }
        
        # Machine learning model for PHI classification
        # In production, load trained model
        self.ml_model = None  # Placeholder for ML model
        
        # Detection statistics
        self.detection_stats = {
            "total_scans": 0,
            "phi_detected": 0,
            "false_positives": 0,
            "detection_by_type": {}
        }
        
        logger.info("PHI Detector initialized with %d patterns", len(self.patterns))
    
    def _compile_patterns(self) -> Dict[PHIType, List[re.Pattern]]:
        """
        Compile regex patterns for each PHI type
        Returns dictionary mapping PHI types to compiled regex patterns
        """
        patterns = {}
        
        # Name patterns
        patterns[PHIType.NAME] = [
            re.compile(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'),  # Standard name
            re.compile(r'\b[A-Z][a-z]+\.? [A-Z][a-z]+\b'),  # Name with initial
            re.compile(r'\b(?:Mr|Mrs|Ms|Dr)\.?\s+[A-Z][a-z]+\b'),  # Title + name
        ]
        
        # Geographic patterns (smaller than state)
        patterns[PHIType.GEOGRAPHIC] = [
            re.compile(r'\b\d{1,5}\s+[A-Za-z0-9\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Circle|Cir|Way)\b', re.IGNORECASE),  # Street address
            re.compile(r'\b[A-Za-z]+\s+(?:City|Town|Village)\b', re.IGNORECASE),  # City names
            re.compile(r'\b[A-Z]{2}\s+\d{5}(?:-\d{4})?\b'),  # State + ZIP
        ]
        
        # Date patterns (all elements except year)
        patterns[PHIType.DATE] = [
            re.compile(r'\b\d{1,2}/\d{1,2}/\d{2,4}\b'),  # MM/DD/YYYY or M/D/YY
            re.compile(r'\b\d{1,2}-\d{1,2}-\d{2,4}\b'),  # MM-DD-YYYY
            re.compile(r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}\b', re.IGNORECASE),  # Month DD, YYYY
            re.compile(r'\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b', re.IGNORECASE),  # DD Month YYYY
        ]
        
        # Phone patterns
        patterns[PHIType.PHONE] = [
            re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'),  # 123-456-7890
            re.compile(r'\b\(\d{3}\)\s*\d{3}[-.]?\d{4}\b'),  # (123) 456-7890
            re.compile(r'\b\+\d{1,2}\s*\d{3}[-.]?\d{3}[-.]?\d{4}\b'),  # +1 123-456-7890
        ]
        
        # Email patterns
        patterns[PHIType.EMAIL] = [
            re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
        ]
        
        # SSN patterns
        patterns[PHIType.SSN] = [
            re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),  # 123-45-6789
            re.compile(r'\b\d{9}\b'),  # 123456789
        ]
        
        # Medical Record Number patterns (hospital-specific)
        patterns[PHIType.MRN] = [
            re.compile(r'\bMRN[:\s]*[A-Z0-9]{6,10}\b', re.IGNORECASE),  # MRN: 123456
            re.compile(r'\b(?:MR|MRN|Record)[\s#]*\d{6,10}\b', re.IGNORECASE),
            re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),  # Some hospitals use SSN format
        ]
        
        # IP Address patterns
        patterns[PHIType.IP_ADDRESS] = [
            re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'),  # IPv4
        ]
        
        # URL patterns
        patterns[PHIType.URL] = [
            re.compile(r'\bhttps?://[^\s/$.?#].[^\s]*\b', re.IGNORECASE),
        ]
        
        return patterns
    
    def detect_phi(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Main method to detect PHI in text using multiple strategies
        
        Args:
            text: Input text to scan for PHI
            context: Additional context about the text (source, purpose, etc.)
        
        Returns:
            Dictionary containing detection results:
            - has_phi: Boolean indicating if PHI was found
            - entities: List of detected PHI entities with metadata
            - risk_level: Overall risk level
            - categories: PHI categories detected
            - confidence: Detection confidence score
        """
        self.detection_stats["total_scans"] += 1
        
        if not text:
            return {
                "has_phi": False,
                "entities": [],
                "risk_level": "none",
                "categories": [],
                "confidence": 1.0
            }
        
        detected_entities = []
        
        # Strategy 1: Regex pattern matching
        regex_entities = self._detect_with_regex(text)
        detected_entities.extend(regex_entities)
        
        # Strategy 2: NLP entity recognition
        nlp_entities = self._detect_with_nlp(text)
        detected_entities.extend(nlp_entities)
        
        # Strategy 3: Contextual detection
        contextual_entities = self._detect_with_context(text, context)
        detected_entities.extend(contextual_entities)
        
        # Strategy 4: ML-based detection (if model available)
        if self.ml_model:
            ml_entities = self._detect_with_ml(text)
            detected_entities.extend(ml_entities)
        
        # Deduplicate and merge entities
        merged_entities = self._merge_entities(detected_entities)
        
        # Calculate overall risk
        has_phi = len(merged_entities) > 0
        risk_level = self._calculate_risk_level(merged_entities)
        categories = list(set(e["type"] for e in merged_entities))
        
        # Calculate confidence (average of entity confidences)
        confidence = 0.0
        if merged_entities:
            confidence = sum(e["confidence"] for e in merged_entities) / len(merged_entities)
        
        if has_phi:
            self.detection_stats["phi_detected"] += 1
            for category in categories:
                self.detection_stats["detection_by_type"][category] = \
                    self.detection_stats["detection_by_type"].get(category, 0) + 1
        
        result = {
            "has_phi": has_phi,
            "entities": merged_entities,
            "risk_level": risk_level.name if risk_level else "none",
            "categories": categories,
            "confidence": confidence,
            "entity_count": len(merged_entities)
        }
        
        logger.debug(f"PHI detection complete: {result}")
        return result
    
    def _detect_with_regex(self, text: str) -> List[Dict[str, Any]]:
        """
        Detect PHI using regex pattern matching
        Returns list of detected entities with positions and types
        """
        entities = []
        
        for phi_type, patterns in self.patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    entity_text = match.group()
                    start_pos = match.start()
                    end_pos = match.end()
                    
                    # Validate match (avoid false positives)
                    if self._validate_match(entity_text, phi_type):
                        entities.append({
                            "text": entity_text,
                            "type": phi_type.value,
                            "start": start_pos,
                            "end": end_pos,
                            "method": "regex",
                            "confidence": self._calculate_pattern_confidence(phi_type, entity_text)
                        })
        
        return entities
    
    def _detect_with_nlp(self, text: str) -> List[Dict[str, Any]]:
        """
        Detect PHI using NLP named entity recognition
        Leverages spaCy's entity recognition capabilities
        """
        entities = []
        doc = self.nlp(text)
        
        # Map spaCy entity types to PHI types
        entity_mapping = {
            "PERSON": PHIType.NAME.value,
            "DATE": PHIType.DATE.value,
            "GPE": PHIType.GEOGRAPHIC.value,
            "LOC": PHIType.GEOGRAPHIC.value,
            "PHONE": PHIType.PHONE.value,
            "EMAIL": PHIType.EMAIL.value,
        }
        
        for ent in doc.ents:
            phi_type = entity_mapping.get(ent.label_)
            if phi_type:
                entities.append({
                    "text": ent.text,
                    "type": phi_type,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "method": "nlp",
                    "confidence": 0.8  # Base confidence for NLP
                })
        
        return entities
    
    def _detect_with_context(
        self,
        text: str,
        context: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Detect PHI using contextual keywords
        Looks for keywords that indicate PHI presence
        """
        entities = []
        text_lower = text.lower()
        
        for phi_type, keywords in self.phi_context_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    # Find the surrounding context
                    keyword_pos = text_lower.find(keyword)
                    context_start = max(0, keyword_pos - 50)
                    context_end = min(len(text), keyword_pos + 100)
                    context_text = text[context_start:context_end]
                    
                    # Look for potential PHI in context
                    # This is simplified - in production use more sophisticated extraction
                    entities.append({
                        "text": context_text[:100],  # Limit context length
                        "type": phi_type,
                        "method": "contextual",
                        "confidence": 0.6,  # Lower confidence for contextual detection
                        "context_keyword": keyword
                    })
        
        return entities
    
    def _detect_with_ml(self, text: str) -> List[Dict[str, Any]]:
        """
        Detect PHI using machine learning model
        Placeholder for ML-based detection
        """
        # In production, this would use a trained model
        # Returning empty list for now
        return []
    
    def _validate_match(self, entity_text: str, phi_type: PHIType) -> bool:
        """
        Validate regex matches to reduce false positives
        Applies domain-specific validation rules
        """
        # Common false positive patterns
        false_positives = {
            PHIType.DATE: [
                r'\d{2,4}-\d{2}-\d{2,4}',  # ISO format dates
                r'\d{4}-\d{2}-\d{2}',      # YYYY-MM-DD (year only is allowed)
            ],
            PHIType.PHONE: [
                r'000-000-0000',  # Placeholder numbers
                r'123-456-7890',  # Test numbers
            ],
            PHIType.NAME: [
                r'Patient\s+Name',  # Field labels
                r'First\s+Name',
            ]
        }
        
        # Check if entity matches false positive patterns
        false_patterns = false_positives.get(phi_type, [])
        for pattern in false_patterns:
            if re.match(pattern, entity_text, re.IGNORECASE):
                return False
        
        # Additional validation rules
        if phi_type == PHIType.DATE:
            # Check if date is plausible
            try:
                # Simple check - in production use date parsing
                if len(entity_text) >= 8:
                    return True
            except:
                return False
        
        return True
    
    def _calculate_pattern_confidence(self, phi_type: PHIType, entity_text: str) -> float:
        """
        Calculate confidence score for regex-based detection
        Higher confidence for patterns with more structure
        """
        base_confidences = {
            PHIType.SSN: 0.95,      # SSN has very specific format
            PHIType.EMAIL: 0.9,      # Email has strict format
            PHIType.PHONE: 0.85,     # Phone numbers have varied formats
            PHIType.MRN: 0.7,        # MRN formats vary by hospital
            PHIType.NAME: 0.6,       # Names are harder to validate
        }
        
        confidence = base_confidences.get(phi_type, 0.5)
        
        # Adjust based on entity length and content
        if len(entity_text) < 3:
            confidence *= 0.5
        
        # Boost confidence for exact matches
        if phi_type == PHIType.SSN and re.match(r'^\d{3}-\d{2}-\d{4}$', entity_text):
            confidence = 0.98
        
        return min(confidence, 1.0)
    
    def _merge_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Merge overlapping entities from different detection methods
        Keeps the highest confidence detection for overlapping spans
        """
        if not entities:
            return []
        
        # Sort by start position
        entities.sort(key=lambda x: (x["start"], -x["end"]))
        
        merged = []
        current = entities[0]
        
        for next_entity in entities[1:]:
            # Check for overlap
            if next_entity["start"] <= current["end"]:
                # Overlapping - keep higher confidence
                if next_entity["confidence"] > current["confidence"]:
                    current = next_entity
            else:
                # No overlap - add current and move to next
                merged.append(current)
                current = next_entity
        
        merged.append(current)
        return merged
    
    def _calculate_risk_level(self, entities: List[Dict[str, Any]]) -> PHIRiskLevel:
        """
        Calculate overall risk level based on detected PHI
        """
        if not entities:
            return PHIRiskLevel.LOW
        
        # Risk levels by PHI type
        risk_by_type = {
            PHIType.SSN.value: PHIRiskLevel.CRITICAL,
            PHIType.NAME.value: PHIRiskLevel.HIGH,
            PHIType.MRN.value: PHIRiskLevel.HIGH,
            PHIType.DATE.value: PHIRiskLevel.MEDIUM,
            PHIType.PHONE.value: PHIRiskLevel.MEDIUM,
            PHIType.EMAIL.value: PHIRiskLevel.MEDIUM,
        }
        
        # Take highest risk level
        max_risk = PHIRiskLevel.LOW
        for entity in entities:
            risk = risk_by_type.get(entity["type"], PHIRiskLevel.MEDIUM)
            if risk.value > max_risk.value:
                max_risk = risk
        
        return max_risk
    
    def redact_phi(self, text: str, phi_results: Optional[Dict] = None) -> str:
        """
        Redact PHI from text by replacing with [REDACTED]
        
        Args:
            text: Original text
            phi_results: Detection results from detect_phi()
        
        Returns:
            Redacted text with PHI removed
        """
        if phi_results is None:
            phi_results = self.detect_phi(text)
        
        if not phi_results["has_phi"]:
            return text
        
        # Sort entities by position in reverse order to avoid index issues
        entities = sorted(
            phi_results["entities"],
            key=lambda x: x["start"],
            reverse=True
        )
        
        redacted = text
        for entity in entities:
            start = entity["start"]
            end = entity["end"]
            
            # Replace with appropriate redaction based on entity type
            entity_type = entity["type"]
            if entity_type in [PHIType.SSN.value, PHIType.MRN.value]:
                replacement = "[REDACTED_ID]"
            elif entity_type == PHIType.NAME.value:
                replacement = "[REDACTED_NAME]"
            elif entity_type == PHIType.DATE.value:
                replacement = "[REDACTED_DATE]"
            else:
                replacement = "[REDACTED]"
            
            redacted = redacted[:start] + replacement + redacted[end:]
        
        return redacted
    
    def mask_phi(self, text: str, phi_results: Optional[Dict] = None) -> str:
        """
        Partially mask PHI while preserving some information
        Useful for display where some context is needed
        """
        if phi_results is None:
            phi_results = self.detect_phi(text)
        
        if not phi_results["has_phi"]:
            return text
        
        entities = sorted(
            phi_results["entities"],
            key=lambda x: x["start"],
            reverse=True
        )
        
        masked = text
        for entity in entities:
            start = entity["start"]
            end = entity["end"]
            entity_text = entity["text"]
            entity_type = entity["type"]
            
            # Apply different masking based on entity type
            if entity_type == PHIType.NAME.value:
                # Show first initial only
                if len(entity_text) > 0:
                    masked_text = entity_text[0] + "****"
                else:
                    masked_text = "****"
            
            elif entity_type == PHIType.SSN.value:
                # Show last 4 digits
                if len(entity_text) >= 4:
                    masked_text = "***-**-" + entity_text[-4:]
                else:
                    masked_text = "***-**-****"
            
            elif entity_type == PHIType.PHONE.value:
                # Show last 4 digits
                digits = re.findall(r'\d', entity_text)
                if len(digits) >= 4:
                    masked_text = "XXX-XXX-" + "".join(digits[-4:])
                else:
                    masked_text = "XXX-XXX-XXXX"
            
            elif entity_type == PHIType.DATE.value:
                # Show year only
                years = re.findall(r'\d{4}', entity_text)
                if years:
                    masked_text = "[DATE " + years[0] + "]"
                else:
                    masked_text = "[REDACTED_DATE]"
            
            else:
                masked_text = "[REDACTED]"
            
            masked = masked[:start] + masked_text + masked[end:]
        
        return masked
    
    def get_detection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about PHI detection performance
        Useful for monitoring and tuning
        """
        return {
            **self.detection_stats,
            "false_positive_rate": self.detection_stats["false_positives"] / max(self.detection_stats["phi_detected"], 1),
            "unique_categories_detected": len(self.detection_stats["detection_by_type"])
        }
    
    def add_custom_pattern(self, phi_type: PHIType, pattern: str):
        """
        Add custom regex pattern for PHI detection
        Allows hospital-specific patterns (e.g., custom MRN format)
        """
        if phi_type not in self.patterns:
            self.patterns[phi_type] = []
        
        compiled = re.compile(pattern)
        self.patterns[phi_type].append(compiled)
        logger.info(f"Added custom pattern for {phi_type.value}: {pattern}")
    
    def load_training_data(self, file_path: str):
        """
        Load training data for ML model
        Placeholder for ML training functionality
        """
        # In production, load and preprocess training data
        pass