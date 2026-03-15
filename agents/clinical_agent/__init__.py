"""
Clinical Agent Module Initialization
This module provides clinical knowledge and decision support capabilities for the AegisMedBot platform.
It exposes the main ClinicalAgent class and its associated tools for medical information retrieval,
drug interaction checking, and clinical guideline access.
"""

# Import the main ClinicalAgent class from the clinical_agent module
# This is the primary class that will be used by other parts of the system
from .clinical_agent import ClinicalAgent

# Import the MedicalRetriever tool which handles medical literature and guideline retrieval
# This tool interfaces with vector databases and external medical knowledge sources
from .tools.medical_retriever import MedicalRetriever

# Import the DrugInteractionChecker tool which specializes in analyzing drug-drug interactions
# This tool uses medical databases and clinical rules to identify potential medication conflicts
from .tools.drug_interaction import DrugInteractionChecker

# Define what gets imported when someone uses "from clinical_agent import *"
# This controls the module's public interface and prevents accidental exposure of internal classes
__all__ = [
    'ClinicalAgent',           # Main agent class for clinical decision support
    'MedicalRetriever',        # Tool for retrieving medical information
    'DrugInteractionChecker'   # Tool for checking drug interactions
]

# Module metadata for documentation and debugging purposes
__version__ = '1.0.0'  # Semantic versioning for the clinical agent module
__author__ = 'AegisMedBot Team'  # Author information for maintainability
__description__ = 'Clinical knowledge and decision support agent for hospital intelligence platform'