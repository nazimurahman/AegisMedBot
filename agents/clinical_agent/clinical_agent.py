"""
Clinical Agent Implementation
This module contains the ClinicalAgent class which serves as the primary interface for clinical
knowledge retrieval, drug interaction checking, and medical decision support within the hospital platform.
The agent uses specialized tools and language models to provide evidence-based medical information.
"""

import asyncio  # For asynchronous operations and concurrent task execution
from datetime import datetime  # For timestamping and tracking processing times
import logging  # For structured logging and debugging
from typing import Dict, Any, Optional, List, Tuple  # Type hints for better code documentation

# Import base agent classes and protocols from the parent agents module
# The double dot notation (..) means go up two directories in the package hierarchy
from ..base_agent import BaseAgent, AgentMessage, AgentResponse, AgentStatus

# Import specialized tools for medical information retrieval and drug interaction checking
from .tools.medical_retriever import MedicalRetriever
from .tools.drug_interaction import DrugInteractionChecker

# Import prompt templates that guide the agent's behavior and responses
from .prompts import (
    CLINICAL_SYSTEM_PROMPT,      # System-level instructions for the agent's role
    CLINICAL_FEW_SHOT,           # Example interactions for few-shot learning
    DRUG_INTERACTION_TEMPLATE,   # Template for formatting drug interaction responses
    GUIDELINE_TEMPLATE,          # Template for formatting clinical guideline responses
    DISEASE_INFO_TEMPLATE         # Template for formatting disease information responses
)

# Configure module-level logger for consistent error tracking and debugging
# The __name__ variable gives us the fully qualified module name for log messages
logger = logging.getLogger(__name__)


class ClinicalAgent(BaseAgent):
    """
    Clinical Knowledge Agent responsible for medical information retrieval and analysis.
    
    This agent specializes in:
    1. Medical literature retrieval using RAG (Retrieval-Augmented Generation)
    2. Clinical guidelines access and interpretation
    3. Drug-drug interaction checking and contraindication analysis
    4. Disease information and treatment pathway explanations
    5. Evidence-based medical recommendations
    
    The agent inherits from BaseAgent which provides common functionality like
    status tracking, message validation, and human escalation capabilities.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Clinical Agent with its tools and configuration.
        
        Args:
            config: Optional dictionary containing configuration parameters like
                   model settings, API keys, and tool-specific configurations
        """
        # Call the parent class (BaseAgent) constructor with agent metadata
        # This sets up the agent's identity and basic capabilities
        super().__init__(
            name="clinical_agent",  # Unique identifier for this agent in the system
            role="Clinical Knowledge Specialist",  # Human-readable role description
            description="Provides evidence-based medical information and clinical guidelines",  # Detailed description
            config=config  # Pass configuration to the base class
        )
        
        # Initialize specialized tools with the provided configuration
        # The MedicalRetriever handles searching medical literature and guidelines
        self.medical_retriever = MedicalRetriever(config)
        
        # The DrugInteractionChecker specializes in analyzing medication interactions
        self.drug_checker = DrugInteractionChecker(config)
        
        # Store prompt templates for consistent response formatting
        # These templates guide the agent's language and structure in different scenarios
        self.system_prompt = CLINICAL_SYSTEM_PROMPT  # Base instructions for the agent's behavior
        self.few_shot_examples = CLINICAL_FEW_SHOT  # Example conversations for context
        
        # Initialize a cache for frequently accessed medical information
        # This improves performance by avoiding repeated retrievals
        self.response_cache = {}  # Simple dictionary cache (in production, use Redis)
        
        # Track agent statistics for monitoring and improvement
        self.stats = {
            'total_queries': 0,  # Total number of queries processed
            'drug_interaction_queries': 0,  # Count of drug interaction queries
            'guideline_queries': 0,  # Count of clinical guideline queries
            'disease_queries': 0,  # Count of disease information queries
            'average_confidence': 0.0,  # Running average of confidence scores
            'average_processing_time_ms': 0.0  # Running average of processing times
        }
        
        logger.info(f"Clinical Agent initialized with config: {config}")
    
    async def process(self, message: AgentMessage) -> AgentResponse:
        """
        Process incoming clinical queries and return evidence-based responses.
        This is the main entry point for all interactions with this agent.
        
        The method follows a structured pipeline:
        1. Validate the incoming message format
        2. Classify the type of clinical query
        3. Route to specialized handler based on query type
        4. Measure processing time for monitoring
        5. Return formatted response with metadata
        
        Args:
            message: AgentMessage object containing the query and conversation context
            
        Returns:
            AgentResponse object with the processed result and metadata
        """
        # Record start time for performance monitoring
        start_time = datetime.now()
        
        # Update agent status to show we're actively processing
        # This is used by the orchestrator for monitoring and load balancing
        self.update_status(status=AgentStatus.PROCESSING, task_id=message.message_id)
        
        # Validate the incoming message has all required fields
        if not self.validate_input(message):
            logger.warning(f"Invalid message format received: {message}")
            self.update_status(status=AgentStatus.ERROR)
            return AgentResponse(
                message_id=message.message_id,
                content={"error": "Invalid message format - missing required fields"},
                confidence=0.0,
                processing_time_ms=0.0
            )
        
        try:
            # Extract the query text from the message content
            # The query could be in different fields, so we handle multiple possibilities
            query = message.content.get("query", "") or message.content.get("text", "")
            
            # Get any patient context if provided (for personalized responses)
            patient_context = message.content.get("patient_context", {})
            
            # Update statistics
            self.stats['total_queries'] += 1
            
            # Classify the type of clinical query for routing
            query_type = self._classify_query(query)
            logger.debug(f"Classified query as: {query_type}")
            
            # Check cache for identical queries to avoid redundant processing
            cache_key = f"{query}_{query_type}"
            if cache_key in self.response_cache:
                logger.info(f"Cache hit for query: {query[:50]}...")
                cached_response = self.response_cache[cache_key]
                return AgentResponse(
                    message_id=message.message_id,
                    content=cached_response,
                    confidence=0.95,  # High confidence for cached responses
                    processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000
                )
            
            # Route to the appropriate handler based on query classification
            if query_type == "drug_interaction":
                response = await self._handle_drug_interaction(query, message)
                self.stats['drug_interaction_queries'] += 1
            elif query_type == "guideline":
                response = await self._handle_guideline_query(query, message, patient_context)
                self.stats['guideline_queries'] += 1
            elif query_type == "disease_info":
                response = await self._handle_disease_info(query, message)
                self.stats['disease_queries'] += 1
            else:  # general clinical query
                response = await self._handle_general_clinical(query, message)
            
            # Calculate processing time in milliseconds
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Add processing time to response
            response.processing_time_ms = processing_time
            
            # Update running averages for statistics
            self._update_stats(response.confidence, processing_time)
            
            # Cache the response for future similar queries
            # Only cache if confidence is high enough
            if response.confidence > 0.8:
                self.response_cache[cache_key] = response.content
                # Limit cache size to prevent memory issues
                if len(self.response_cache) > 1000:
                    # Remove oldest item (simple FIFO)
                    self.response_cache.pop(next(iter(self.response_cache)))
            
            # Update status to completed
            self.update_status(status=AgentStatus.COMPLETED)
            
            logger.info(f"Successfully processed query in {processing_time:.2f}ms with confidence {response.confidence:.2f}")
            return response
            
        except Exception as e:
            # Log the full error with traceback for debugging
            logger.error(f"Error in clinical agent processing: {str(e)}", exc_info=True)
            
            # Update status to error
            self.update_status(status=AgentStatus.ERROR)
            
            # Return a graceful error response
            return AgentResponse(
                message_id=message.message_id,
                content={
                    "error": f"Clinical processing error: {str(e)}",
                    "suggestion": "Please rephrase your question or consult a medical professional directly."
                },
                confidence=0.0,
                processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000
            )
    
    def _classify_query(self, query: str) -> str:
        """
        Classify the type of clinical query based on keyword analysis.
        
        This method examines the query text for keywords that indicate the type
        of medical information being requested. It helps route the query to the
        appropriate specialized handler.
        
        Args:
            query: The user's query string
            
        Returns:
            String indicating query type: "drug_interaction", "guideline", 
            "disease_info", or "general"
        """
        # Convert query to lowercase for case-insensitive matching
        query_lower = query.lower()
        
        # Define keyword sets for each query type
        # Drug interaction keywords - words that suggest medication-related queries
        drug_keywords = [
            "interact", "drug", "medication", "contraindication", "prescription",
            "dose", "dosage", "side effect", "adverse", "reaction", "allergic",
            "medicine", "pill", "tablet", "capsule", "injection", "infusion"
        ]
        
        # Clinical guideline keywords - words that suggest protocol/guideline queries
        guideline_keywords = [
            "guideline", "protocol", "recommendation", "standard", "best practice",
            "pathway", "algorithm", "procedure", "step", "flowchart", "criteria",
            "indication", "when to", "how to treat", "management of"
        ]
        
        # Disease information keywords - words that suggest disease-related queries
        disease_keywords = [
            "disease", "condition", "syndrome", "disorder", "illness", "sickness",
            "diagnosis", "symptom", "sign", "presentation", "prognosis", "outcome",
            "what is", "define", "explain", "description", "overview"
        ]
        
        # Count matches for each category
        drug_matches = sum(1 for keyword in drug_keywords if keyword in query_lower)
        guideline_matches = sum(1 for keyword in guideline_keywords if keyword in query_lower)
        disease_matches = sum(1 for keyword in disease_keywords if keyword in query_lower)
        
        # Determine the most likely query type based on highest match count
        # If there's a tie, we prioritize in this order: drug > guideline > disease > general
        if drug_matches > 0 and drug_matches >= guideline_matches and drug_matches >= disease_matches:
            return "drug_interaction"
        elif guideline_matches > 0 and guideline_matches >= disease_matches:
            return "guideline"
        elif disease_matches > 0:
            return "disease_info"
        else:
            return "general"
    
    def _extract_medications(self, query: str) -> List[str]:
        """
        Extract medication names from the query text.
        
        This method uses a combination of pattern matching and a medical dictionary
        to identify medication names in the user's query.
        
        Args:
            query: The user's query string
            
        Returns:
            List of identified medication names
        """
        # Comprehensive list of common medications
        # In production, this would be loaded from a medical database or API
        common_drugs = {
            # Cardiovascular drugs
            "lisinopril", "amlodipine", "metoprolol", "atenolol", "carvedilol",
            "losartan", "valsartan", "hydrochlorothiazide", "furosemide", "spironolactone",
            
            # Diabetes drugs
            "metformin", "glipizide", "glyburide", "pioglitazone", "sitagliptin",
            "empagliflozin", "dapagliflozin", "liraglutide", "insulin",
            
            # Pain relievers
            "ibuprofen", "naproxen", "acetaminophen", "aspirin", "celecoxib",
            "tramadol", "oxycodone", "hydrocodone", "morphine",
            
            # Antibiotics
            "amoxicillin", "azithromycin", "ciprofloxacin", "doxycycline", "penicillin",
            "cephalexin", "sulfamethoxazole", "trimethoprim",
            
            # Antidepressants
            "sertraline", "fluoxetine", "citalopram", "escitalopram", "paroxetine",
            "duloxetine", "venlafaxine", "bupropion",
            
            # Cholesterol drugs
            "atorvastatin", "simvastatin", "rosuvastatin", "pravastatin", "ezetimibe",
            
            # Asthma/COPD
            "albuterol", "salmeterol", "fluticasone", "budesonide", "montelukast",
            
            # Blood thinners
            "warfarin", "apixaban", "rivaroxaban", "dabigatran", "clopidogrel",
            
            # Thyroid
            "levothyroxine", "methimazole", "propylthiouracil",
            
            # Psychiatric
            "lorazepam", "alprazolam", "clonazepam", "diazepam", "olanzapine",
            "quetiapine", "risperidone", "haloperidol",
            
            # Gastrointestinal
            "omeprazole", "pantoprazole", "esomeprazole", "ranitidine", "famotidine",
            "ondansetron", "metoclopramide"
        }
        
        # Convert query to lowercase for matching
        query_lower = query.lower()
        
        # Find all medications that appear in the query
        found_medications = []
        for drug in common_drugs:
            # Check if the drug name appears as a whole word
            # This prevents partial matches like "aspirin" matching in "aspiration"
            if f" {drug} " in f" {query_lower} " or query_lower.startswith(drug + " ") or query_lower.endswith(" " + drug):
                found_medications.append(drug)
        
        # Also check for drug names without word boundaries for simple cases
        # This is a fallback for short queries
        if not found_medications:
            for drug in common_drugs:
                if drug in query_lower:
                    found_medications.append(drug)
                    break  # Only take the first match for simplicity
        
        return found_medications
    
    def _extract_condition(self, query: str) -> Optional[str]:
        """
        Extract medical condition from query text.
        
        Args:
            query: The user's query string
            
        Returns:
            The identified condition or None if not found
        """
        # Comprehensive list of medical conditions with their variations
        conditions = {
            # Cardiovascular
            "hypertension": ["hypertension", "high blood pressure", "htn"],
            "heart failure": ["heart failure", "chf", "congestive heart failure"],
            "coronary artery disease": ["coronary artery disease", "cad", "heart disease"],
            "atrial fibrillation": ["atrial fibrillation", "afib", "arrhythmia"],
            
            # Endocrine
            "diabetes": ["diabetes", "diabetes mellitus", "dm", "type 2 diabetes", "type 1 diabetes"],
            "hypothyroidism": ["hypothyroidism", "underactive thyroid"],
            "hyperthyroidism": ["hyperthyroidism", "overactive thyroid"],
            
            # Respiratory
            "asthma": ["asthma", "reactive airway disease"],
            "copd": ["copd", "chronic obstructive pulmonary disease", "emphysema"],
            "pneumonia": ["pneumonia", "lung infection"],
            
            # Infectious
            "uti": ["uti", "urinary tract infection", "bladder infection"],
            "cellulitis": ["cellulitis", "skin infection"],
            "sepsis": ["sepsis", "blood infection"],
            
            # Neurological
            "stroke": ["stroke", "cva", "cerebrovascular accident"],
            "migraine": ["migraine", "severe headache"],
            "alzheimer": ["alzheimer", "dementia"],
            
            # Gastrointestinal
            "gerd": ["gerd", "reflux", "acid reflux", "heartburn"],
            "gastritis": ["gastritis", "stomach inflammation"],
            "ibd": ["ibd", "inflammatory bowel disease", "crohn", "ulcerative colitis"],
            
            # Musculoskeletal
            "arthritis": ["arthritis", "osteoarthritis", "rheumatoid arthritis"],
            "back pain": ["back pain", "lower back pain"],
            "osteoporosis": ["osteoporosis", "brittle bones"],
            
            # Mental health
            "depression": ["depression", "depressive disorder"],
            "anxiety": ["anxiety", "generalized anxiety disorder", "gad"],
            "bipolar": ["bipolar", "bipolar disorder", "manic depression"]
        }
        
        query_lower = query.lower()
        
        # Check each condition and its variations
        for condition, variations in conditions.items():
            for variation in variations:
                if variation in query_lower:
                    return condition
        
        return None
    
    def _extract_disease(self, query: str) -> Optional[str]:
        """
        Extract disease name from query. This is similar to condition extraction
        but with a focus on disease entities.
        
        Args:
            query: The user's query string
            
        Returns:
            The identified disease or None if not found
        """
        # Reuse condition extraction as it serves the same purpose
        return self._extract_condition(query)
    
    async def _handle_drug_interaction(
        self,
        query: str,
        message: AgentMessage
    ) -> AgentResponse:
        """
        Handle drug interaction queries by checking medication combinations.
        
        This method:
        1. Extracts medication names from the query
        2. Checks for known interactions using the drug interaction tool
        3. Retrieves supporting literature
        4. Formats a comprehensive response
        
        Args:
            query: The user's query string
            message: The original agent message
            
        Returns:
            Formatted AgentResponse with drug interaction information
        """
        # Extract medications from the query
        medications = self._extract_medications(query)
        
        # If no medications found, ask for clarification
        if not medications:
            return AgentResponse(
                message_id=message.message_id,
                content={
                    "response": "I couldn't identify specific medications in your query. Please specify the medication names you're asking about.",
                    "suggestions": [
                        "Include both generic and brand names",
                        "Specify if prescription or over-the-counter",
                        "Mention dosages if known"
                    ],
                    "type": "clarification_needed"
                },
                confidence=0.4,
                requires_human_confirmation=False
            )
        
        # Check interactions between all identified medications
        # The drug checker handles pairwise and multi-drug interactions
        interactions = await self.drug_checker.check_interactions(medications)
        
        # Retrieve supporting literature for the identified medications
        # This provides evidence-based context for the interactions
        literature = await self.medical_retriever.retrieve(
            f"drug interactions {' '.join(medications)}",
            top_k=3,
            filter_conditions={"source": "pubmed"}  # Prioritize peer-reviewed sources
        )
        
        # Determine confidence based on interaction data quality
        confidence = interactions.get("confidence", 0.7)
        
        # Check if any severe interactions were found that require human review
        severe_interaction = any(
            interaction.get("severity") == "severe" or interaction.get("severity") == "contraindicated"
            for interaction in interactions.get("interactions", [])
        )
        
        # Format the response using the template
        formatted_response = DRUG_INTERACTION_TEMPLATE.format(
            medications=", ".join(medications),
            summary=interactions.get("summary", "No significant interactions found."),
            interactions=self._format_interactions_list(interactions.get("interactions", [])),
            references="\n".join([f"- {ref.get('title', 'Unknown')}" for ref in literature[:2]])
        )
        
        # Prepare the response content
        response_content = {
            "response": formatted_response,
            "medications_checked": medications,
            "interactions_found": interactions.get("interactions", []),
            "severity_level": interactions.get("max_severity", "unknown"),
            "supporting_literature": literature,
            "disclaimer": "This information is for reference only. Always consult with a pharmacist or physician before starting or changing medications.",
            "requires_professional_consultation": severe_interaction
        }
        
        # Determine if human confirmation is needed
        requires_human = confidence < 0.8 or severe_interaction
        
        return AgentResponse(
            message_id=message.message_id,
            content=response_content,
            tool_results=[
                {"interaction_check": interactions},
                {"literature_retrieved": literature}
            ],
            confidence=confidence,
            requires_human_confirmation=requires_human
        )
    
    def _format_interactions_list(self, interactions: List[Dict]) -> str:
        """
        Format a list of drug interactions into a readable string.
        
        Args:
            interactions: List of interaction dictionaries
            
        Returns:
            Formatted string with interaction details
        """
        if not interactions:
            return "No interactions found between the specified medications."
        
        formatted = []
        for interaction in interactions:
            severity = interaction.get("severity", "unknown")
            description = interaction.get("description", "No description available")
            
            # Add emoji indicators based on severity (using text-only alternatives)
            severity_indicator = {
                "severe": "[SEVERE]",
                "moderate": "[MODERATE]",
                "mild": "[MILD]",
                "contraindicated": "[CONTRAINDICATED - DO NOT USE]"
            }.get(severity, "[UNKNOWN]")
            
            formatted.append(f"{severity_indicator} {description}")
        
        return "\n".join(formatted)
    
    async def _handle_guideline_query(
        self,
        query: str,
        message: AgentMessage,
        patient_context: Dict[str, Any]
    ) -> AgentResponse:
        """
        Handle clinical guideline queries by retrieving relevant protocols.
        
        Args:
            query: The user's query string
            message: The original agent message
            patient_context: Optional patient-specific context for personalized guidelines
            
        Returns:
            Formatted AgentResponse with guideline information
        """
        # Extract condition from query
        condition = self._extract_condition(query)
        
        if not condition:
            return AgentResponse(
                message_id=message.message_id,
                content={
                    "response": "Please specify the medical condition or procedure you're asking about.",
                    "suggestions": [
                        "Example: What are the guidelines for hypertension?",
                        "Example: Show me the sepsis protocol",
                        "Example: How to manage diabetes?"
                    ],
                    "type": "clarification_needed"
                },
                confidence=0.4,
                requires_human_confirmation=False
            )
        
        # Retrieve guidelines for the specific condition
        # The retriever can filter by source, year, and specialty
        guidelines = await self.medical_retriever.retrieve_guidelines(
            condition,
            patient_context=patient_context,  # For personalized recommendations
            include_recent_only=True  # Prefer recent guidelines
        )
        
        # If patient context is available, tailor the response
        personalized_note = ""
        if patient_context:
            age = patient_context.get("age")
            gender = patient_context.get("gender")
            if age and age > 65:
                personalized_note = "\nNote: Special considerations for elderly patients are included where applicable."
        
        # Format the response using the guideline template
        formatted_response = GUIDELINE_TEMPLATE.format(
            condition=condition.title(),
            summary=guidelines.get("summary", ""),
            key_recommendations=self._format_recommendations(guidelines.get("recommendations", [])),
            sources=", ".join(guidelines.get("sources", ["Clinical guidelines"])),
            year=guidelines.get("year", "Current"),
            personalized_note=personalized_note
        )
        
        response_content = {
            "response": formatted_response,
            "condition": condition,
            "guidelines": guidelines,
            "recommendations": guidelines.get("recommendations", []),
            "sources": guidelines.get("sources", []),
            "evidence_level": guidelines.get("evidence_level", "Not specified"),
            "disclaimer": "Guidelines may vary by region and institution. Always verify with local protocols."
        }
        
        return AgentResponse(
            message_id=message.message_id,
            content=response_content,
            tool_results=[{"guidelines_retrieved": guidelines}],
            confidence=guidelines.get("confidence", 0.8),
            requires_human_confirmation=False
        )
    
    def _format_recommendations(self, recommendations: List[Dict]) -> str:
        """
        Format clinical recommendations into a readable list.
        
        Args:
            recommendations: List of recommendation dictionaries
            
        Returns:
            Formatted string with recommendations
        """
        if not recommendations:
            return "No specific recommendations available."
        
        formatted = []
        for i, rec in enumerate(recommendations, 1):
            strength = rec.get("strength", "recommended")
            text = rec.get("text", "")
            formatted.append(f"{i}. [{strength.upper()}] {text}")
        
        return "\n".join(formatted)
    
    async def _handle_disease_info(
        self,
        query: str,
        message: AgentMessage
    ) -> AgentResponse:
        """
        Handle disease information queries by retrieving comprehensive disease data.
        
        Args:
            query: The user's query string
            message: The original agent message
            
        Returns:
            Formatted AgentResponse with disease information
        """
        disease = self._extract_disease(query)
        
        if not disease:
            return AgentResponse(
                message_id=message.message_id,
                content={
                    "response": "Please specify the disease or condition you'd like information about.",
                    "suggestions": [
                        "Example: What is diabetes?",
                        "Example: Tell me about pneumonia",
                        "Example: Explain heart failure"
                    ],
                    "type": "clarification_needed"
                },
                confidence=0.4,
                requires_human_confirmation=False
            )
        
        # Retrieve comprehensive disease information
        disease_info = await self.medical_retriever.retrieve_disease_info(disease)
        
        # Format the response using the disease info template
        formatted_response = DISEASE_INFO_TEMPLATE.format(
            disease=disease.title(),
            overview=disease_info.get("overview", "Information not available."),
            symptoms=self._format_list(disease_info.get("symptoms", [])),
            causes=disease_info.get("causes", "Information not available."),
            diagnosis=self._format_list(disease_info.get("diagnosis", [])),
            treatment=self._format_list(disease_info.get("treatment", [])),
            prognosis=disease_info.get("prognosis", "Information not available."),
            references="\n".join([f"- {ref}" for ref in disease_info.get("references", [])[:3]])
        )
        
        response_content = {
            "response": formatted_response,
            "disease": disease,
            "overview": disease_info.get("overview", ""),
            "symptoms": disease_info.get("symptoms", []),
            "causes": disease_info.get("causes", ""),
            "diagnosis": disease_info.get("diagnosis", []),
            "treatment": disease_info.get("treatment", []),
            "prognosis": disease_info.get("prognosis", ""),
            "references": disease_info.get("references", [])
        }
        
        return AgentResponse(
            message_id=message.message_id,
            content=response_content,
            tool_results=[{"disease_info": disease_info}],
            confidence=0.85,
            requires_human_confirmation=False
        )
    
    def _format_list(self, items: List[str]) -> str:
        """
        Format a list of items into a bulleted string.
        
        Args:
            items: List of strings to format
            
        Returns:
            Bulleted list as string
        """
        if not items:
            return "None listed."
        
        return "\n".join([f"• {item}" for item in items])
    
    async def _handle_general_clinical(
        self,
        query: str,
        message: AgentMessage
    ) -> AgentResponse:
        """
        Handle general clinical queries that don't fit specific categories.
        Uses the medical retriever to find relevant information.
        
        Args:
            query: The user's query string
            message: The original agent message
            
        Returns:
            Formatted AgentResponse with general clinical information
        """
        # Retrieve general medical information
        results = await self.medical_retriever.retrieve(query, top_k=5)
        
        if not results:
            return AgentResponse(
                message_id=message.message_id,
                content={
                    "response": "I couldn't find specific information about your query. Please rephrase or consult a medical professional directly.",
                    "suggestions": [
                        "Try using more specific medical terminology",
                        "Include relevant symptoms or conditions",
                        "Ask about a particular aspect of the topic"
                    ]
                },
                confidence=0.3,
                requires_human_confirmation=False
            )
        
        # Extract the most relevant result
        top_result = results[0]
        
        # Build response from retrieved information
        response_text = f"Based on medical literature:\n\n{top_result.get('content', '')}"
        
        if len(results) > 1:
            response_text += f"\n\nAdditional sources: {', '.join([r.get('source', 'Unknown') for r in results[1:3]])}"
        
        response_content = {
            "response": response_text,
            "sources": [r.get("source") for r in results],
            "related_topics": [r.get("title") for r in results if r.get("title")],
            "disclaimer": "This information is for educational purposes. Consult healthcare providers for medical advice."
        }
        
        return AgentResponse(
            message_id=message.message_id,
            content=response_content,
            tool_results=[{"retrieved_documents": results}],
            confidence=top_result.get("score", 0.7),
            requires_human_confirmation=False
        )
    
    def _update_stats(self, confidence: float, processing_time_ms: float):
        """
        Update running statistics for monitoring and improvement.
        
        Args:
            confidence: Confidence score from the response
            processing_time_ms: Processing time in milliseconds
        """
        # Update average confidence using running average formula
        total = self.stats['total_queries']
        if total > 0:
            self.stats['average_confidence'] = (
                (self.stats['average_confidence'] * (total - 1) + confidence) / total
            )
            self.stats['average_processing_time_ms'] = (
                (self.stats['average_processing_time_ms'] * (total - 1) + processing_time_ms) / total
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get current agent statistics for monitoring.
        
        Returns:
            Dictionary with agent statistics
        """
        return {
            **self.stats,
            'cache_size': len(self.response_cache),
            'current_status': self.status.value
        }
    
    async def can_handle(self, task_type: str, context: Dict[str, Any]) -> float:
        """
        Determine if this agent can handle a given task.
        Overrides the base class method with clinical-specific logic.
        
        Args:
            task_type: Type of task to evaluate
            context: Additional context for decision making
            
        Returns:
            Confidence score between 0 and 1
        """
        # High confidence for clinical tasks
        if task_type in ["clinical", "medical", "drug", "disease", "guideline"]:
            return 0.9
        
        # Medium confidence for related tasks
        if task_type in ["patient", "treatment", "diagnosis"]:
            return 0.7
        
        # Low confidence for unrelated tasks
        return 0.2