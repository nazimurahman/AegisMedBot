"""
Drug Interaction Checker Tool for Clinical Agent
This module provides comprehensive drug-drug interaction checking capabilities
using medical knowledge bases and clinical guidelines.
"""

import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import torch
import numpy as np
from pathlib import Path
import hashlib
from datetime import datetime, timedelta

# Configure logging for the drug interaction module
logger = logging.getLogger(__name__)

class InteractionSeverity(Enum):
    """
    Enumeration defining the severity levels of drug interactions.
    Used to classify interactions based on clinical significance.
    """
    MINOR = "minor"           # Mild interaction, usually no medical intervention needed
    MODERATE = "moderate"     # Moderate interaction, may require monitoring or dose adjustment
    MAJOR = "major"           # Severe interaction, potentially life-threatening
    CONTRAINDICATED = "contraindicated"  # Should never be used together
    UNKNOWN = "unknown"       # Interaction unknown or not studied

class EvidenceLevel(Enum):
    """
    Enumeration defining the strength of evidence for drug interactions.
    Based on clinical studies and medical literature.
    """
    WELL_ESTABLISHED = "well_established"  # Confirmed by multiple studies
    PROBABLE = "probable"                    # Likely based on evidence
    POSSIBLE = "possible"                    # Suspected but not confirmed
    THEORETICAL = "theoretical"              # Based on pharmacological principles
    UNKNOWN = "unknown"                       # No evidence available

@dataclass
class DrugInteraction:
    """
    Data class representing a single drug-drug interaction.
    Contains all relevant information about how two drugs interact.
    """
    drug1: str                    # Name of first drug
    drug2: str                    # Name of second drug
    severity: InteractionSeverity  # Clinical severity of interaction
    mechanism: str                 # Pharmacological mechanism of interaction
    description: str               # Detailed description of the interaction
    clinical_effects: List[str]    # List of potential clinical effects
    management: List[str]          # Recommended management strategies
    evidence_level: EvidenceLevel  # Strength of evidence
    references: List[str]          # Medical literature references
    onset_time: Optional[str] = None  # Time until interaction manifests
    risk_factors: List[str] = field(default_factory=list)  # Patient-specific risk factors
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the interaction object to a dictionary for JSON serialization.
        Returns:
            Dictionary representation of the interaction
        """
        return {
            "drug1": self.drug1,
            "drug2": self.drug2,
            "severity": self.severity.value,
            "mechanism": self.mechanism,
            "description": self.description,
            "clinical_effects": self.clinical_effects,
            "management": self.management,
            "evidence_level": self.evidence_level.value,
            "references": self.references,
            "onset_time": self.onset_time,
            "risk_factors": self.risk_factors
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DrugInteraction':
        """
        Create a DrugInteraction object from a dictionary.
        
        Args:
            data: Dictionary containing interaction data
            
        Returns:
            DrugInteraction object
        """
        return cls(
            drug1=data["drug1"],
            drug2=data["drug2"],
            severity=InteractionSeverity(data["severity"]),
            mechanism=data["mechanism"],
            description=data["description"],
            clinical_effects=data["clinical_effects"],
            management=data["management"],
            evidence_level=EvidenceLevel(data["evidence_level"]),
            references=data["references"],
            onset_time=data.get("onset_time"),
            risk_factors=data.get("risk_factors", [])
        )

class DrugInteractionDatabase:
    """
    Manages the drug interaction knowledge base.
    Loads interactions from various sources and provides querying capabilities.
    """
    
    def __init__(self, database_path: Optional[str] = None):
        """
        Initialize the drug interaction database.
        
        Args:
            database_path: Optional path to custom interaction database file
        """
        self.interactions: Dict[str, Dict[str, DrugInteraction]] = {}
        self.drug_metadata: Dict[str, Dict[str, Any]] = {}
        self.cache: Dict[str, Tuple[List[DrugInteraction], datetime]] = {}
        self.cache_duration = timedelta(hours=24)  # Cache interactions for 24 hours
        
        # Load default interaction database
        self._load_default_database()
        
        # Load custom database if provided
        if database_path and Path(database_path).exists():
            self._load_custom_database(database_path)
        
        logger.info(f"Drug interaction database initialized with {len(self.interactions)} drug pairs")
    
    def _load_default_database(self):
        """
        Load the default drug interaction database.
        Contains common drug interactions based on medical literature.
        """
        # This is a simplified database for demonstration
        # In production, this would load from comprehensive medical databases
        
        # Define common drug interactions
        default_interactions = [
            {
                "drug1": "warfarin",
                "drug2": "aspirin",
                "severity": "major",
                "mechanism": "Additive anticoagulant effect",
                "description": "Concurrent use increases risk of bleeding complications",
                "clinical_effects": ["Increased bleeding time", "Gastrointestinal bleeding", "Easy bruising"],
                "management": ["Monitor INR closely", "Consider alternative antiplatelet therapy", "Educate patient about bleeding signs"],
                "evidence_level": "well_established",
                "references": ["PubMed ID: 12345678", "Drug Interaction Facts 2024"],
                "onset_time": "Days to weeks",
                "risk_factors": ["Elderly", "History of GI bleeding", "Renal impairment"]
            },
            {
                "drug1": "simvastatin",
                "drug2": "amiodarone",
                "severity": "major",
                "mechanism": "CYP3A4 inhibition by amiodarone increasing statin levels",
                "description": "Increased risk of statin-induced myopathy and rhabdomyolysis",
                "clinical_effects": ["Muscle pain", "Muscle weakness", "Dark urine", "Elevated CK levels"],
                "management": ["Limit simvastatin dose to 20mg/day", "Consider alternative statin", "Monitor CK levels"],
                "evidence_level": "well_established",
                "references": ["FDA Drug Safety Communication", "PubMed ID: 87654321"],
                "onset_time": "Weeks to months",
                "risk_factors": ["High statin dose", "Renal impairment", "Hypothyroidism"]
            },
            {
                "drug1": "metformin",
                "drug2": "iodinated_contrast",
                "severity": "moderate",
                "mechanism": "Risk of contrast-induced nephropathy leading to metformin accumulation",
                "description": "Increased risk of lactic acidosis in patients with renal impairment",
                "clinical_effects": ["Lactic acidosis", "Renal failure", "Nausea and vomiting"],
                "management": ["Hold metformin before contrast procedure", "Check renal function", "Restart after 48 hours if renal function stable"],
                "evidence_level": "well_established",
                "references": ["ACR Manual on Contrast Media", "PubMed ID: 23456789"],
                "onset_time": "Hours to days",
                "risk_factors": ["Pre-existing renal impairment", "Dehydration", "High contrast dose"]
            },
            {
                "drug1": "digoxin",
                "drug2": "furosemide",
                "severity": "moderate",
                "mechanism": "Hypokalemia induced by furosemide increases digoxin toxicity",
                "description": "Potassium depletion enhances digoxin binding to Na/K-ATPase",
                "clinical_effects": ["Arrhythmias", "Nausea", "Visual disturbances", "Confusion"],
                "management": ["Monitor potassium levels", "Consider potassium supplementation", "Monitor digoxin levels"],
                "evidence_level": "well_established",
                "references": ["PubMed ID: 34567890", "Goodman and Gilman's Pharmacology"],
                "onset_time": "Days",
                "risk_factors": ["Elderly", "Renal impairment", "Low potassium intake"]
            },
            {
                "drug1": "sildenafil",
                "drug2": "nitroglycerin",
                "severity": "contraindicated",
                "mechanism": "Potentiation of vasodilatory effects",
                "description": "Severe hypotension and syncope due to additive vasodilation",
                "clinical_effects": ["Severe hypotension", "Syncope", "Myocardial infarction"],
                "management": ["Absolute contraindication", "Use alternative therapy", "If taken together, emergency medical attention required"],
                "evidence_level": "well_established",
                "references": ["FDA Label", "PubMed ID: 45678901"],
                "onset_time": "Immediate to hours",
                "risk_factors": ["Cardiovascular disease", "Volume depletion", "Elderly"]
            },
            {
                "drug1": "lisinopril",
                "drug2": "spironolactone",
                "severity": "moderate",
                "mechanism": "Additive hyperkalemic effect",
                "description": "Increased risk of hyperkalemia, especially in patients with renal impairment",
                "clinical_effects": ["Elevated potassium", "Arrhythmias", "Muscle weakness"],
                "management": ["Monitor potassium levels", "Avoid in renal impairment", "Consider dose adjustment"],
                "evidence_level": "well_established",
                "references": ["PubMed ID: 56789012", "JNC 8 Guidelines"],
                "onset_time": "Days to weeks",
                "risk_factors": ["Renal impairment", "Diabetes", "Elderly"]
            },
            {
                "drug1": "clopidogrel",
                "drug2": "omeprazole",
                "severity": "moderate",
                "mechanism": "CYP2C19 inhibition reduces clopidogrel activation",
                "description": "Reduced antiplatelet effect potentially leading to thrombotic events",
                "clinical_effects": ["Reduced platelet inhibition", "Increased risk of stent thrombosis"],
                "management": ["Consider pantoprazole instead", "Monitor for cardiovascular events", "Consider alternative antiplatelet"],
                "evidence_level": "probable",
                "references": ["PubMed ID: 67890123", "FDA Drug Interaction Guidance"],
                "onset_time": "Days",
                "risk_factors": ["Poor CYP2C19 metabolizers", "High-risk cardiovascular patients"]
            },
            {
                "drug1": "lithium",
                "drug2": "ibuprofen",
                "severity": "major",
                "mechanism": "NSAID-induced reduction in lithium excretion",
                "description": "Increased lithium levels leading to toxicity",
                "clinical_effects": ["Tremor", "Confusion", "Seizures", "Renal impairment"],
                "management": ["Avoid NSAIDs if possible", "Monitor lithium levels", "Hydration", "Consider alternative analgesics"],
                "evidence_level": "well_established",
                "references": ["PubMed ID: 78901234", "Lithium Treatment Guidelines"],
                "onset_time": "Days",
                "risk_factors": ["Elderly", "Renal impairment", "Dehydration"]
            }
        ]
        
        # Load interactions into database
        for interaction_data in default_interactions:
            interaction = DrugInteraction.from_dict(interaction_data)
            self.add_interaction(interaction)
        
        logger.info(f"Loaded {len(default_interactions)} default drug interactions")
    
    def _load_custom_database(self, database_path: str):
        """
        Load custom drug interactions from a JSON file.
        
        Args:
            database_path: Path to custom database JSON file
        """
        try:
            with open(database_path, 'r') as f:
                custom_interactions = json.load(f)
            
            for interaction_data in custom_interactions:
                interaction = DrugInteraction.from_dict(interaction_data)
                self.add_interaction(interaction)
            
            logger.info(f"Loaded {len(custom_interactions)} custom drug interactions")
        except Exception as e:
            logger.error(f"Error loading custom database: {str(e)}")
    
    def add_interaction(self, interaction: DrugInteraction):
        """
        Add a drug interaction to the database.
        
        Args:
            interaction: DrugInteraction object to add
        """
        # Normalize drug names for consistent lookup
        drug1 = interaction.drug1.lower().strip()
        drug2 = interaction.drug2.lower().strip()
        
        # Create nested dictionaries if they don't exist
        if drug1 not in self.interactions:
            self.interactions[drug1] = {}
        if drug2 not in self.interactions:
            self.interactions[drug2] = {}
        
        # Store interaction in both directions
        self.interactions[drug1][drug2] = interaction
        self.interactions[drug2][drug1] = interaction
        
        # Update drug metadata
        self._update_drug_metadata(interaction.drug1)
        self._update_drug_metadata(interaction.drug2)
    
    def _update_drug_metadata(self, drug_name: str):
        """
        Update metadata for a drug (interaction count, etc.).
        
        Args:
            drug_name: Name of the drug
        """
        drug_key = drug_name.lower().strip()
        if drug_key not in self.drug_metadata:
            self.drug_metadata[drug_key] = {
                "name": drug_name,
                "interaction_count": 0,
                "first_seen": datetime.now().isoformat()
            }
        
        # Count interactions for this drug
        if drug_key in self.interactions:
            self.drug_metadata[drug_key]["interaction_count"] = len(self.interactions[drug_key])
    
    def get_interaction(self, drug1: str, drug2: str) -> Optional[DrugInteraction]:
        """
        Get the interaction between two specific drugs.
        
        Args:
            drug1: Name of first drug
            drug2: Name of second drug
            
        Returns:
            DrugInteraction object if found, None otherwise
        """
        drug1_key = drug1.lower().strip()
        drug2_key = drug2.lower().strip()
        
        # Check if interaction exists
        if drug1_key in self.interactions and drug2_key in self.interactions[drug1_key]:
            return self.interactions[drug1_key][drug2_key]
        
        return None
    
    def get_all_interactions_for_drug(self, drug_name: str) -> List[DrugInteraction]:
        """
        Get all interactions for a specific drug.
        
        Args:
            drug_name: Name of the drug
            
        Returns:
            List of DrugInteraction objects
        """
        drug_key = drug_name.lower().strip()
        interactions = []
        
        if drug_key in self.interactions:
            for other_drug, interaction in self.interactions[drug_key].items():
                interactions.append(interaction)
        
        return interactions
    
    def search_drugs(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for drugs in the database.
        
        Args:
            query: Search query string
            limit: Maximum number of results
            
        Returns:
            List of drug metadata dictionaries
        """
        query = query.lower().strip()
        results = []
        
        for drug_key, metadata in self.drug_metadata.items():
            if query in drug_key or query in metadata["name"].lower():
                results.append({
                    "name": metadata["name"],
                    "interaction_count": metadata["interaction_count"],
                    "matched_term": drug_key
                })
        
        # Sort by relevance (exact matches first)
        results.sort(key=lambda x: (
            - (1 if x["name"].lower() == query else 0),
            - (0.5 if query in x["name"].lower() else 0),
            x["name"]
        ))
        
        return results[:limit]
    
    def get_cache_key(self, drug_list: List[str]) -> str:
        """
        Generate a cache key for a list of drugs.
        
        Args:
            drug_list: List of drug names
            
        Returns:
            Cache key string
        """
        # Sort drugs to ensure consistent key regardless of order
        sorted_drugs = sorted([d.lower().strip() for d in drug_list])
        combined = ",".join(sorted_drugs)
        return hashlib.md5(combined.encode()).hexdigest()
    
    def cache_interactions(self, drug_list: List[str], interactions: List[DrugInteraction]):
        """
        Cache interaction results for a drug combination.
        
        Args:
            drug_list: List of drug names
            interactions: List of found interactions
        """
        cache_key = self.get_cache_key(drug_list)
        self.cache[cache_key] = (interactions, datetime.now())
    
    def get_cached_interactions(self, drug_list: List[str]) -> Optional[List[DrugInteraction]]:
        """
        Get cached interactions for a drug combination if available and not expired.
        
        Args:
            drug_list: List of drug names
            
        Returns:
            Cached interactions if available and valid, None otherwise
        """
        cache_key = self.get_cache_key(drug_list)
        
        if cache_key in self.cache:
            interactions, timestamp = self.cache[cache_key]
            if datetime.now() - timestamp < self.cache_duration:
                return interactions
            else:
                # Remove expired cache entry
                del self.cache[cache_key]
        
        return None

class DrugInteractionChecker:
    """
    Main class for checking drug interactions.
    Combines database lookup with ML-based interaction prediction.
    """
    
    def __init__(self, database_path: Optional[str] = None):
        """
        Initialize the drug interaction checker.
        
        Args:
            database_path: Optional path to custom interaction database
        """
        self.database = DrugInteractionDatabase(database_path)
        self.ml_model = None
        self.model_loaded = False
        
        # Try to load ML model if available
        self._load_ml_model()
        
        logger.info("DrugInteractionChecker initialized")
    
    def _load_ml_model(self):
        """
        Load the machine learning model for interaction prediction.
        Uses a simple neural network for demonstration.
        """
        try:
            # Define a simple neural network for interaction prediction
            class InteractionPredictor(torch.nn.Module):
                def __init__(self, input_size: int = 100, hidden_size: int = 64):
                    super().__init__()
                    self.layer1 = torch.nn.Linear(input_size, hidden_size)
                    self.layer2 = torch.nn.Linear(hidden_size, hidden_size // 2)
                    self.layer3 = torch.nn.Linear(hidden_size // 2, 5)  # 5 severity classes
                    self.dropout = torch.nn.Dropout(0.3)
                    self.relu = torch.nn.ReLU()
                
                def forward(self, x):
                    x = self.relu(self.layer1(x))
                    x = self.dropout(x)
                    x = self.relu(self.layer2(x))
                    x = self.dropout(x)
                    x = self.layer3(x)
                    return torch.softmax(x, dim=-1)
            
            # Initialize model
            self.ml_model = InteractionPredictor()
            
            # In production, you would load trained weights here
            # self.ml_model.load_state_dict(torch.load('models/interaction_predictor.pt'))
            
            self.ml_model.eval()
            self.model_loaded = True
            logger.info("ML model loaded successfully")
            
        except Exception as e:
            logger.warning(f"Could not load ML model: {str(e)}")
            self.model_loaded = False
    
    async def check_interactions(
        self,
        medications: List[str],
        patient_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Check for interactions among a list of medications.
        
        Args:
            medications: List of medication names
            patient_context: Optional patient-specific information
            
        Returns:
            Dictionary containing interaction check results
        """
        if not medications or len(medications) < 2:
            return {
                "interactions": [],
                "total_checked": len(medications),
                "has_interactions": False,
                "severity_summary": {},
                "recommendations": [],
                "confidence": 1.0
            }
        
        # Check cache first
        cached_result = self.database.get_cached_interactions(medications)
        if cached_result is not None:
            logger.info(f"Using cached interactions for {len(medications)} drugs")
            return self._format_results(cached_result, medications)
        
        # Find interactions
        found_interactions = []
        interaction_pairs = []
        
        # Check each unique pair of medications
        for i in range(len(medications)):
            for j in range(i + 1, len(medications)):
                drug1 = medications[i]
                drug2 = medications[j]
                
                # Look up in database
                interaction = self.database.get_interaction(drug1, drug2)
                
                if interaction:
                    found_interactions.append(interaction)
                    interaction_pairs.append((drug1, drug2))
                elif self.model_loaded:
                    # If not in database and ML model available, predict
                    predicted = self._predict_interaction(drug1, drug2, patient_context)
                    if predicted:
                        found_interactions.append(predicted)
                        interaction_pairs.append((drug1, drug2))
        
        # Cache results
        self.database.cache_interactions(medications, found_interactions)
        
        # Format and return results
        return self._format_results(found_interactions, medications, patient_context)
    
    def _predict_interaction(
        self,
        drug1: str,
        drug2: str,
        patient_context: Optional[Dict[str, Any]] = None
    ) -> Optional[DrugInteraction]:
        """
        Predict potential interaction using ML model.
        
        Args:
            drug1: Name of first drug
            drug2: Name of second drug
            patient_context: Optional patient context
            
        Returns:
            Predicted DrugInteraction or None
        """
        if not self.model_loaded:
            return None
        
        try:
            # Create drug embeddings (simplified for demonstration)
            # In production, you would use pre-trained drug embeddings
            drug1_embedding = self._create_drug_embedding(drug1)
            drug2_embedding = self._create_drug_embedding(drug2)
            
            # Combine embeddings
            combined = torch.cat([drug1_embedding, drug2_embedding], dim=0)
            
            # Add patient context if available
            if patient_context:
                context_embedding = self._create_context_embedding(patient_context)
                combined = torch.cat([combined, context_embedding], dim=0)
            
            # Pad or truncate to expected input size
            if combined.shape[0] < 100:
                padding = torch.zeros(100 - combined.shape[0])
                combined = torch.cat([combined, padding])
            else:
                combined = combined[:100]
            
            # Add batch dimension
            combined = combined.unsqueeze(0)
            
            # Make prediction
            with torch.no_grad():
                prediction = self.ml_model(combined)
            
            # Get predicted severity class
            severity_idx = torch.argmax(prediction, dim=1).item()
            confidence = prediction[0][severity_idx].item()
            
            # Only return if confidence is high enough
            if confidence > 0.7:
                # Map index to severity
                severity_map = [
                    InteractionSeverity.MINOR,
                    InteractionSeverity.MODERATE,
                    InteractionSeverity.MAJOR,
                    InteractionSeverity.CONTRAINDICATED,
                    InteractionSeverity.UNKNOWN
                ]
                
                # Create predicted interaction
                return DrugInteraction(
                    drug1=drug1,
                    drug2=drug2,
                    severity=severity_map[severity_idx],
                    mechanism="Predicted based on molecular similarity",
                    description=f"Potential interaction predicted with {confidence:.1%} confidence",
                    clinical_effects=["Monitor patient closely"],
                    management=["Consult pharmacist", "Monitor for adverse effects"],
                    evidence_level=EvidenceLevel.THEORETICAL,
                    references=["ML Model Prediction"],
                    risk_factors=["Based on computational prediction"]
                )
            
        except Exception as e:
            logger.error(f"Error in interaction prediction: {str(e)}")
        
        return None
    
    def _create_drug_embedding(self, drug_name: str) -> torch.Tensor:
        """
        Create a simple embedding for a drug name.
        In production, this would use pre-trained molecular embeddings.
        
        Args:
            drug_name: Name of the drug
            
        Returns:
            Tensor embedding
        """
        # Simple character-level embedding for demonstration
        drug_name = drug_name.lower().strip()
        embedding = []
        
        for char in drug_name:
            # Convert character to a simple numeric representation
            char_val = ord(char) / 255.0  # Normalize to [0, 1]
            embedding.append(char_val)
        
        # Pad or truncate to fixed length
        target_length = 20
        if len(embedding) < target_length:
            embedding.extend([0.0] * (target_length - len(embedding)))
        else:
            embedding = embedding[:target_length]
        
        return torch.tensor(embedding, dtype=torch.float32)
    
    def _create_context_embedding(self, context: Dict[str, Any]) -> torch.Tensor:
        """
        Create embedding from patient context.
        
        Args:
            context: Patient context dictionary
            
        Returns:
            Tensor embedding
        """
        features = []
        
        # Age feature
        age = context.get('age', 50)
        features.append(age / 100.0)  # Normalize to [0, 1]
        
        # Renal function
        renal_function = context.get('renal_function', 'normal')
        renal_map = {'normal': 0.0, 'mild_impairment': 0.3, 'moderate_impairment': 0.6, 'severe_impairment': 1.0}
        features.append(renal_map.get(renal_function, 0.0))
        
        # Hepatic function
        hepatic_function = context.get('hepatic_function', 'normal')
        hepatic_map = {'normal': 0.0, 'mild_impairment': 0.3, 'moderate_impairment': 0.6, 'severe_impairment': 1.0}
        features.append(hepatic_map.get(hepatic_function, 0.0))
        
        # Number of medications
        num_meds = context.get('num_medications', 0)
        features.append(min(num_meds / 20.0, 1.0))  # Cap at 20 medications
        
        return torch.tensor(features, dtype=torch.float32)
    
    def _format_results(
        self,
        interactions: List[DrugInteraction],
        medications: List[str],
        patient_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Format interaction results into a structured response.
        
        Args:
            interactions: List of found interactions
            medications: Original medication list
            patient_context: Patient context
            
        Returns:
            Formatted results dictionary
        """
        # Count interactions by severity
        severity_counts = {}
        recommendations = set()
        clinical_effects = set()
        
        for interaction in interactions:
            severity = interaction.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            # Collect recommendations and clinical effects
            for rec in interaction.management:
                recommendations.add(rec)
            for effect in interaction.clinical_effects:
                clinical_effects.add(effect)
        
        # Determine overall severity
        if InteractionSeverity.CONTRAINDICATED.value in severity_counts:
            overall_severity = "contraindicated"
        elif InteractionSeverity.MAJOR.value in severity_counts:
            overall_severity = "major"
        elif InteractionSeverity.MODERATE.value in severity_counts:
            overall_severity = "moderate"
        elif InteractionSeverity.MINOR.value in severity_counts:
            overall_severity = "minor"
        else:
            overall_severity = "none"
        
        # Calculate confidence score
        if interactions:
            confidence = sum(i.evidence_level.value in ['well_established', 'probable'] for i in interactions) / len(interactions)
        else:
            confidence = 1.0
        
        # Create interaction summaries
        interaction_summaries = []
        for interaction in interactions:
            summary = {
                "drug_pair": f"{interaction.drug1} - {interaction.drug2}",
                "severity": interaction.severity.value,
                "description": interaction.description[:100] + "..." if len(interaction.description) > 100 else interaction.description,
                "mechanism": interaction.mechanism,
                "management": interaction.management[:3],  # Top 3 management strategies
                "evidence": interaction.evidence_level.value
            }
            interaction_summaries.append(summary)
        
        return {
            "interactions": [i.to_dict() for i in interactions],
            "interaction_summaries": interaction_summaries,
            "total_checked": len(medications),
            "total_interactions": len(interactions),
            "has_interactions": len(interactions) > 0,
            "severity_summary": severity_counts,
            "overall_severity": overall_severity,
            "recommendations": list(recommendations)[:5],  # Top 5 recommendations
            "clinical_effects": list(clinical_effects)[:5],  # Top 5 effects
            "confidence": confidence,
            "requires_pharmacist_review": overall_severity in ['major', 'contraindicated'],
            "requires_physician_review": overall_severity in ['moderate', 'major', 'contraindicated']
        }
    
    def get_drug_information(self, drug_name: str) -> Dict[str, Any]:
        """
        Get comprehensive information about a drug.
        
        Args:
            drug_name: Name of the drug
            
        Returns:
            Dictionary containing drug information
        """
        # Get all interactions for this drug
        interactions = self.database.get_all_interactions_for_drug(drug_name)
        
        # Get drug metadata
        drug_key = drug_name.lower().strip()
        metadata = self.database.drug_metadata.get(drug_key, {"name": drug_name, "interaction_count": len(interactions)})
        
        # Categorize interactions by severity
        by_severity = {}
        for interaction in interactions:
            severity = interaction.severity.value
            if severity not in by_severity:
                by_severity[severity] = []
            by_severity[severity].append({
                "drug": interaction.drug2 if interaction.drug1.lower() == drug_key else interaction.drug1,
                "severity": severity,
                "description": interaction.description[:100]
            })
        
        return {
            "name": drug_name,
            "interaction_count": len(interactions),
            "interactions_by_severity": {k: len(v) for k, v in by_severity.items()},
            "contraindications": [i for i in interactions if i.severity == InteractionSeverity.CONTRAINDICATED],
            "major_interactions": [i for i in interactions if i.severity == InteractionSeverity.MAJOR],
            "metadata": metadata
        }

# Export the main classes
__all__ = [
    'DrugInteractionChecker',
    'DrugInteractionDatabase',
    'DrugInteraction',
    'InteractionSeverity',
    'EvidenceLevel'
]