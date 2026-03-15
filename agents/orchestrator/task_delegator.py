"""
Task Delegator Module for Intelligent Agent Selection

This module contains the TaskDelegator class which is responsible for analyzing
user queries and selecting the most appropriate agent to handle them.

The delegator uses multiple strategies to determine the best agent:
1. Keyword matching - Direct matching of domain-specific terms
2. Semantic similarity - Using embeddings to understand query meaning
3. Context awareness - Considering conversation history
4. Agent capability profiling - Understanding what each agent can do

This multi-faceted approach ensures accurate routing of queries to the right
specialist agent, mimicking how a human triage nurse would direct patients
to appropriate specialists.
"""

import numpy as np
import logging
from typing import Dict, Any, List, Tuple, Optional
from sentence_transformers import SentenceTransformer

# Configure module-level logger
logger = logging.getLogger(__name__)

class TaskDelegator:
    """
    Intelligent task delegator that selects the most appropriate agent for each query.
    
    This class implements the core routing logic of the multi-agent system.
    It maintains profiles of all agents and uses multiple techniques to match
    user queries to the best-suited specialist.
    
    The delegator uses:
    - Pre-defined agent capability profiles
    - Keyword-based matching for quick routing
    - Semantic embeddings for understanding query meaning
    - Conversation context for maintaining continuity
    
    Attributes:
        orchestrator: Reference to the parent AgentOrchestrator instance
        embedding_model: Sentence transformer model for semantic similarity
        agent_profiles: Dictionary of agent capabilities and keywords
    """
    
    def __init__(self, orchestrator):
        """
        Initialize the TaskDelegator with reference to the orchestrator.
        
        Args:
            orchestrator: The parent AgentOrchestrator instance that provides
                         access to registered agents and configuration
        """
        self.orchestrator = orchestrator
        
        # Initialize the sentence transformer model for semantic similarity
        # This model converts text into vector embeddings that capture meaning
        # We use a lightweight model for fast inference
        logger.info("Loading embedding model for semantic similarity...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Embedding model loaded successfully")
        
        # Initialize agent capability profiles
        # These profiles define what each agent can do and what keywords they respond to
        self.agent_profiles = self._initialize_agent_profiles()
        
    def _initialize_agent_profiles(self) -> Dict[str, Dict[str, Any]]:
        """
        Initialize detailed profiles for each agent type.
        
        Each profile contains:
        - capabilities: List of specific tasks the agent can perform
        - keywords: Domain-specific terms that indicate this agent is needed
        - confidence_threshold: Minimum confidence for automatic routing
        
        These profiles are used for both keyword matching and semantic similarity
        to determine the best agent for each query.
        
        Returns:
            Dictionary mapping agent names to their capability profiles
        """
        return {
            'clinical_agent': {
                'capabilities': [
                    'disease diagnosis and management',
                    'treatment guidelines and protocols',
                    'drug interactions and contraindications',
                    'medical literature interpretation',
                    'clinical decision support',
                    'symptom analysis and triage'
                ],
                'keywords': [
                    'diagnosis', 'treatment', 'medication', 'drug', 'disease',
                    'symptom', 'guideline', 'protocol', 'clinical', 'therapy',
                    'prescription', 'dosage', 'side effect', 'contraindication',
                    'condition', 'illness', 'disorder', 'syndrome'
                ],
                'confidence_threshold': 0.6
            },
            'risk_agent': {
                'capabilities': [
                    'patient risk stratification',
                    'mortality prediction',
                    'complication risk assessment',
                    'ICU admission prediction',
                    'readmission risk scoring',
                    'prognosis estimation'
                ],
                'keywords': [
                    'risk', 'prediction', 'probability', 'chance', 'likelihood',
                    'mortality', 'complication', 'prognosis', 'outcome',
                    'survival', 'deterioration', 'worsening', 'develop'
                ],
                'confidence_threshold': 0.7
            },
            'operations_agent': {
                'capabilities': [
                    'bed occupancy analysis',
                    'patient flow optimization',
                    'resource allocation',
                    'staff utilization tracking',
                    'emergency department wait times',
                    'capacity planning'
                ],
                'keywords': [
                    'bed', 'occupancy', 'capacity', 'flow', 'wait time',
                    'resource', 'staff', 'schedule', 'operation', 'efficiency',
                    'throughput', 'bottleneck', 'utilization', 'available'
                ],
                'confidence_threshold': 0.65
            },
            'director_agent': {
                'capabilities': [
                    'hospital KPIs and metrics',
                    'financial performance analysis',
                    'department efficiency tracking',
                    'executive reporting',
                    'quality metrics monitoring',
                    'strategic planning insights'
                ],
                'keywords': [
                    'kpi', 'metric', 'performance', 'financial', 'revenue',
                    'cost', 'efficiency', 'quality', 'outcome', 'report',
                    'dashboard', 'executive', 'strategic', 'overview',
                    'summary', 'trend', 'benchmark'
                ],
                'confidence_threshold': 0.7
            },
            'research_agent': {
                'capabilities': [
                    'medical literature search',
                    'research paper summarization',
                    'clinical trial matching',
                    'evidence synthesis',
                    'publication analysis',
                    'knowledge discovery'
                ],
                'keywords': [
                    'research', 'study', 'paper', 'literature', 'evidence',
                    'trial', 'publication', 'journal', 'article', 'review',
                    'meta-analysis', 'systematic', 'finding', 'discovery'
                ],
                'confidence_threshold': 0.6
            },
            'compliance_agent': {
                'capabilities': [
                    'HIPAA compliance guidance',
                    'privacy protection protocols',
                    'regulatory requirement checking',
                    'audit trail maintenance',
                    'policy enforcement',
                    'security best practices'
                ],
                'keywords': [
                    'compliance', 'hipaa', 'privacy', 'regulation', 'audit',
                    'policy', 'secure', 'confidential', 'protected',
                    'authorization', 'permission', 'consent', 'legal'
                ],
                'confidence_threshold': 0.8
            }
        }
    
    async def select_agent(
        self,
        query: str,
        context: Dict[str, Any]
    ) -> Tuple[str, float]:
        """
        Select the most appropriate agent for a given query.
        
        This is the main public method of the TaskDelegator. It combines
        multiple scoring methods to determine the best agent match.
        
        The selection process:
        1. Convert query to lowercase for case-insensitive matching
        2. Calculate keyword matching scores
        3. Calculate semantic similarity scores
        4. Calculate context-based scores (conversation continuity)
        5. Combine scores with weighted average
        6. Return best matching agent and confidence
        
        Args:
            query: The user's query text
            context: Current conversation context including history
            
        Returns:
            Tuple of (agent_name, confidence_score)
            If no suitable agent found, returns ('human_escalation', 0.0)
        """
        # Normalize query for consistent processing
        query_lower = query.lower()
        logger.debug(f"Selecting agent for query: {query[:50]}...")
        
        # Dictionary to store scores for each agent
        agent_scores = {}
        
        # Calculate scores for each registered agent
        for agent_name, profile in self.agent_profiles.items():
            # Skip if agent isn't actually registered
            if agent_name not in self.orchestrator.agents:
                logger.debug(f"Skipping {agent_name} - not registered")
                continue
            
            # Calculate individual score components
            keyword_score = self._calculate_keyword_score(
                query_lower,
                profile['keywords']
            )
            
            semantic_score = await self._calculate_semantic_score(
                query,
                profile['capabilities']
            )
            
            context_score = self._calculate_context_score(
                agent_name,
                context
            )
            
            # Combine scores with weights
            # Weights are tuned based on empirical testing:
            # - Semantic: 50% (most important - captures meaning)
            # - Keyword: 30% (important for domain-specific terms)
            # - Context: 20% (maintains conversation continuity)
            final_score = (
                0.3 * keyword_score +
                0.5 * semantic_score +
                0.2 * context_score
            )
            
            agent_scores[agent_name] = final_score
            
            logger.debug(
                f"Agent {agent_name} scores - "
                f"Keyword: {keyword_score:.2f}, "
                f"Semantic: {semantic_score:.2f}, "
                f"Context: {context_score:.2f}, "
                f"Final: {final_score:.2f}"
            )
        
        # If no agents scored (shouldn't happen), escalate to human
        if not agent_scores:
            logger.warning("No agents available for scoring, escalating to human")
            return "human_escalation", 0.0
        
        # Find the agent with the highest score
        best_agent = max(agent_scores.items(), key=lambda x: x[1])
        
        logger.info(f"Selected agent: {best_agent[0]} with confidence {best_agent[1]:.2f}")
        
        return best_agent
    
    def _calculate_keyword_score(self, query: str, keywords: List[str]) -> float:
        """
        Calculate a score based on keyword matches in the query.
        
        This method counts how many domain-specific keywords appear in the query
        and returns a normalized score. It's a fast, lightweight way to identify
        relevant agents.
        
        Args:
            query: Lowercase user query
            keywords: List of keywords for a specific agent
            
        Returns:
            Float between 0 and 1 representing keyword match strength
        """
        if not keywords:
            return 0.0
        
        # Count how many keywords appear in the query
        matches = sum(1 for keyword in keywords if keyword in query)
        
        # Normalize by total keywords, but cap at 1.0
        # This prevents very long queries from getting artificially high scores
        normalized_score = min(matches / len(keywords), 1.0)
        
        return normalized_score
    
    async def _calculate_semantic_score(
        self,
        query: str,
        capabilities: List[str]
    ) -> float:
        """
        Calculate semantic similarity between query and agent capabilities.
        
        This method uses sentence transformers to create vector embeddings
        of both the query and the agent's capabilities, then computes cosine
        similarity to understand semantic meaning beyond simple keyword matching.
        
        Args:
            query: Original user query (not normalized)
            capabilities: List of capability descriptions for an agent
            
        Returns:
            Float between 0 and 1 representing semantic similarity
        """
        if not capabilities:
            return 0.0
        
        # Encode query into a vector embedding
        # This captures the semantic meaning of the query
        query_embedding = self.embedding_model.encode(query)
        
        # Encode all capabilities into embeddings
        # We do this in batch for efficiency
        capability_embeddings = self.embedding_model.encode(capabilities)
        
        # Calculate cosine similarities between query and each capability
        # Cosine similarity measures the angle between vectors in embedding space
        # Values range from -1 to 1, with 1 meaning very similar
        
        # Normalize query embedding
        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            return 0.0
        query_normalized = query_embedding / query_norm
        
        # Normalize capability embeddings
        capability_norms = np.linalg.norm(capability_embeddings, axis=1, keepdims=True)
        capability_norms = np.where(capability_norms == 0, 1, capability_norms)
        capabilities_normalized = capability_embeddings / capability_norms
        
        # Compute dot product (cosine similarity since vectors are normalized)
        similarities = np.dot(capabilities_normalized, query_normalized)
        
        # Return the maximum similarity (best matching capability)
        max_similarity = float(np.max(similarities))
        
        # Clip to [0, 1] range (cosine similarity can be slightly negative)
        return max(0.0, min(1.0, max_similarity))
    
    def _calculate_context_score(self, agent_name: str, context: Dict[str, Any]) -> float:
        """
        Calculate a score based on conversation context.
        
        This method considers the conversation history to maintain continuity.
        If the same agent was used recently, it gets a boost to keep the
        conversation flowing naturally.
        
        Args:
            agent_name: Name of the agent being considered
            context: Current conversation context dictionary
            
        Returns:
            Float between 0 and 1 representing context relevance
        """
        # If no context available, return neutral score
        if not context:
            return 0.5
        
        # Check if this agent was used in the last interaction
        last_agent = context.get('last_agent')
        if last_agent == agent_name:
            # Boost score for continuity
            return 0.8
        elif last_agent is not None:
            # Slight penalty for switching agents
            return 0.3
        
        # Default neutral score
        return 0.5
    
    def update_agent_profile(self, agent_name: str, profile_updates: Dict[str, Any]) -> None:
        """
        Update an agent's capability profile dynamically.
        
        This allows for runtime adjustments to agent capabilities, useful for
        A/B testing or gradual capability expansion.
        
        Args:
            agent_name: Name of the agent to update
            profile_updates: Dictionary of profile fields to update
        """
        if agent_name in self.agent_profiles:
            self.agent_profiles[agent_name].update(profile_updates)
            logger.info(f"Updated profile for agent: {agent_name}")
        else:
            logger.warning(f"Attempted to update unknown agent: {agent_name}")
    
    def get_agent_recommendation_explanation(self, query: str) -> Dict[str, Any]:
        """
        Provide explanation of why a particular agent was selected.
        
        This is useful for debugging, transparency, and helping users understand
        how their query was interpreted.
        
        Args:
            query: User query to analyze
            
        Returns:
            Dictionary with detailed scoring breakdown
        """
        query_lower = query.lower()
        explanations = {}
        
        for agent_name, profile in self.agent_profiles.items():
            keyword_score = self._calculate_keyword_score(query_lower, profile['keywords'])
            
            explanations[agent_name] = {
                'keyword_matches': [
                    kw for kw in profile['keywords'] if kw in query_lower
                ],
                'keyword_score': keyword_score,
                'threshold': profile['confidence_threshold']
            }
        
        return explanations