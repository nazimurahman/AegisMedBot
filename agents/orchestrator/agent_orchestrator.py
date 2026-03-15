"""
Central orchestrator that manages all specialized agents in the AegisMedBot system.
Implements hierarchical multi-agent coordination with intelligent task delegation,
context management, and human oversight integration.
"""

from typing import Dict, Any, Optional, List, Tuple
import asyncio
from datetime import datetime
from collections import defaultdict
import logging
import json
import traceback

from ..base_agent import BaseAgent, AgentMessage, AgentResponse, AgentStatus
from .task_delegator import TaskDelegator
from .context_manager import ContextManager

# Configure logging for the orchestrator
logger = logging.getLogger(__name__)

class AgentOrchestrator:
    """
    Central orchestrator that manages the entire multi-agent system.
    
    Responsibilities:
    1. Register and maintain all specialized agents
    2. Receive and route incoming messages
    3. Maintain conversation context across agents
    4. Coordinate multi-agent workflows
    5. Handle human escalation when needed
    6. Collect and report system metrics
    
    The orchestrator implements a hierarchical architecture where it acts as
    the single entry point for all user interactions, delegating to specialized
    agents based on intent analysis and context.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the agent orchestrator with configuration.
        
        Args:
            config: Optional configuration dictionary containing:
                - human_threshold: Confidence threshold for human escalation
                - max_iterations: Maximum agent chaining iterations
                - timeout_seconds: Default timeout for agent processing
                - enable_metrics: Whether to collect performance metrics
        """
        self.config = config or {}
        
        # Initialize core components
        self.agents: Dict[str, BaseAgent] = {}  # Registry of all agents
        self.task_delegator = TaskDelegator(self)  # Handles agent selection
        self.context_manager = ContextManager()  # Manages conversation context
        
        # Track active conversations for monitoring
        self.active_conversations: Dict[str, Dict[str, Any]] = {}
        
        # Performance metrics collection
        self.metrics: Dict[str, List[float]] = defaultdict(list)
        self.metrics.update({
            "processing_times": [],  # Time per request
            "agent_usage": [],  # Which agents were used
            "escalation_rates": [],  # How often human escalation needed
            "error_rates": [],  # Error frequency
            "confidence_scores": []  # Confidence distribution
        })
        
        # Configuration defaults
        self.human_threshold = self.config.get("human_threshold", 0.7)
        self.max_iterations = self.config.get("max_iterations", 5)
        self.timeout_seconds = self.config.get("timeout_seconds", 30)
        self.enable_metrics = self.config.get("enable_metrics", True)
        
        logger.info("Agent Orchestrator initialized successfully")
    
    def register_agent(self, agent: BaseAgent):
        """
        Register a new agent with the orchestrator.
        
        Args:
            agent: Instance of BaseAgent to register
            
        The orchestrator maintains a registry of all available agents
        and can discover them by name or capability.
        """
        if agent.name in self.agents:
            logger.warning(f"Agent {agent.name} already registered. Overwriting.")
        
        self.agents[agent.name] = agent
        logger.info(f"Registered agent: {agent.name} - {agent.role}")
        
        # Log all registered agents for debugging
        logger.debug(f"Current agents: {list(self.agents.keys())}")
    
    def get_agent(self, agent_name: str) -> Optional[BaseAgent]:
        """
        Get an agent by name.
        
        Args:
            agent_name: Name of the agent to retrieve
            
        Returns:
            Agent instance if found, None otherwise
        """
        agent = self.agents.get(agent_name)
        if not agent:
            logger.warning(f"Agent {agent_name} not found")
        return agent
    
    async def process_message(
        self,
        message: Dict[str, Any],
        conversation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process an incoming message through the agent pipeline.
        This is the main entry point for all user interactions.
        
        Args:
            message: Dictionary containing at least a "query" field
            conversation_id: Optional existing conversation ID for context
            
        Returns:
            Dictionary containing:
                - conversation_id: Current conversation ID
                - response: The generated response
                - agent: Name of the primary agent used
                - confidence: Confidence score (0-1)
                - requires_human: Whether human review is needed
                - processing_time_ms: Time taken in milliseconds
        """
        start_time = datetime.now()
        
        # Extract query from message
        query = message.get("query", "")
        if not query:
            return {
                "error": "No query provided",
                "requires_human": True
            }
        
        logger.info(f"Processing message: {query[:100]}...")
        
        # Initialize or retrieve conversation context
        if not conversation_id:
            conversation_id = self.context_manager.create_conversation()
            logger.info(f"Created new conversation: {conversation_id}")
        
        # Get current context for this conversation
        context = self.context_manager.get_context(conversation_id)
        
        # Create standardized agent message
        agent_message = AgentMessage(
            conversation_id=conversation_id,
            sender="user",
            recipient="orchestrator",
            message_type="request",
            content=message,
            metadata={
                "user_id": message.get("user_id", "anonymous"),
                "patient_id": message.get("patient_id"),
                "timestamp": datetime.now().isoformat()
            }
        )
        
        try:
            # Step 1: Detect intent and select appropriate agent
            logger.info(f"Selecting agent for query: {query[:50]}...")
            selected_agent, confidence = await self.task_delegator.select_agent(
                query,
                context
            )
            
            logger.info(f"Selected agent: {selected_agent} with confidence: {confidence:.2f}")
            
            # Track confidence for metrics
            if self.enable_metrics:
                self.metrics["confidence_scores"].append(confidence)
            
            # Step 2: Check if confidence is too low for autonomous processing
            if confidence < self.human_threshold:
                logger.info(f"Confidence {confidence:.2f} below threshold {self.human_threshold}. Escalating.")
                return await self._handle_low_confidence(
                    agent_message,
                    selected_agent,
                    confidence
                )
            
            # Step 3: Process with selected agent
            logger.info(f"Routing to agent: {selected_agent}")
            response = await self._route_to_agent(
                agent_message,
                selected_agent,
                context
            )
            
            # Step 4: Update conversation context with results
            self.context_manager.update_context(conversation_id, {
                "last_agent": selected_agent,
                "last_response": response.dict(),
                "timestamp": datetime.now().isoformat()
            })
            
            # Step 5: Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Track metrics
            if self.enable_metrics:
                self.metrics["processing_times"].append(processing_time)
                self.metrics["agent_usage"].append(selected_agent)
            
            logger.info(f"Message processed successfully in {processing_time:.2f}ms")
            
            # Step 6: Format and return response
            return {
                "conversation_id": conversation_id,
                "response": response.content,
                "agent": selected_agent,
                "confidence": response.confidence,
                "requires_human": response.requires_human_confirmation,
                "processing_time_ms": processing_time,
                "suggestions": response.content.get("suggestions", []),
                "sources": response.content.get("sources", [])
            }
            
        except asyncio.TimeoutError:
            logger.error(f"Timeout processing message for conversation {conversation_id}")
            return await self._handle_error(
                agent_message,
                "Processing timeout - please try again"
            )
            
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}\n{traceback.format_exc()}")
            return await self._handle_error(agent_message, str(e))
    
    async def _route_to_agent(
        self,
        message: AgentMessage,
        agent_name: str,
        context: Dict[str, Any],
        iteration: int = 0
    ) -> AgentResponse:
        """
        Route message to specific agent and handle multi-step processes.
        
        Args:
            message: The message to route
            agent_name: Target agent name
            context: Current conversation context
            iteration: Current iteration count for loop prevention
            
        Returns:
            AgentResponse from the processing chain
        """
        # Prevent infinite loops
        if iteration >= self.max_iterations:
            logger.warning(f"Max iterations ({self.max_iterations}) reached")
            return AgentResponse(
                message_id=message.message_id,
                content={
                    "error": "Processing chain too long",
                    "response": "I need to break this down into simpler questions."
                },
                confidence=0.3,
                requires_human_confirmation=True,
                processing_time_ms=0
            )
        
        # Get the target agent
        agent = self.get_agent(agent_name)
        if not agent:
            raise ValueError(f"Agent {agent_name} not found")
        
        # Add context to message
        message.metadata["context"] = context
        message.metadata["iteration"] = iteration
        
        # Update agent status
        agent.update_status(AgentStatus.PROCESSING, message.message_id)
        
        # Process with agent (with timeout)
        try:
            response = await asyncio.wait_for(
                agent.process(message),
                timeout=self.timeout_seconds
            )
        except asyncio.TimeoutError:
            logger.error(f"Agent {agent_name} timed out after {self.timeout_seconds}s")
            agent.update_status(AgentStatus.ERROR)
            raise
        finally:
            # Reset agent status if not waiting for human
            if agent.status != AgentStatus.WAITING_FOR_HUMAN:
                agent.update_status(AgentStatus.IDLE)
        
        # Log the interaction
        await agent.log_interaction(message, response)
        
        # Check if we need to involve another agent
        if response.next_agent and response.next_agent != agent_name:
            logger.info(f"Chaining from {agent_name} to {response.next_agent}")
            
            # Create next message
            next_message = AgentMessage(
                conversation_id=message.conversation_id,
                sender=agent_name,
                recipient=response.next_agent,
                message_type="request",
                content=response.content,
                parent_message_id=message.message_id,
                metadata={
                    "previous_response": response.dict(),
                    "iteration": iteration + 1
                }
            )
            
            # Route to next agent
            return await self._route_to_agent(
                next_message,
                response.next_agent,
                context,
                iteration + 1
            )
        
        return response
    
    async def _handle_low_confidence(
        self,
        message: AgentMessage,
        agent_name: str,
        confidence: float
    ) -> Dict[str, Any]:
        """
        Handle cases where agent confidence is below threshold.
        
        Args:
            message: Original message
            agent_name: Suggested agent
            confidence: Confidence score
            
        Returns:
            Response indicating escalation
        """
        # Track escalation in metrics
        if self.enable_metrics:
            self.metrics["escalation_rates"].append(1)
        
        # Create escalation record
        escalation = {
            "type": "escalation",
            "query": message.content.get("query", ""),
            "suggested_agent": agent_name,
            "confidence": confidence,
            "threshold": self.human_threshold,
            "timestamp": datetime.now().isoformat(),
            "conversation_id": message.conversation_id
        }
        
        # Store for human review (in production, this would trigger a notification)
        self.active_conversations[message.conversation_id] = {
            "status": "awaiting_human",
            "escalation": escalation,
            "context": self.context_manager.get_context(message.conversation_id)
        }
        
        logger.info(f"Conversation {message.conversation_id} escalated to human")
        
        # Return user-friendly response
        return {
            "conversation_id": message.conversation_id,
            "response": {
                "status": "escalated",
                "message": (
                    "I need to consult with a human specialist for this query. "
                    "A clinician will review your question shortly. "
                    "For urgent matters, please contact the hospital directly."
                ),
                "escalation_id": message.message_id
            },
            "agent": "human_escalation",
            "confidence": confidence,
            "requires_human": True,
            "processing_time_ms": 0
        }
    
    async def _handle_error(self, message: AgentMessage, error: str) -> Dict[str, Any]:
        """
        Handle errors gracefully with user-friendly messages.
        
        Args:
            message: Original message that caused error
            error: Error message
            
        Returns:
            User-friendly error response
        """
        # Track error in metrics
        if self.enable_metrics:
            self.metrics["error_rates"].append(1)
        
        logger.error(f"Error for conversation {message.conversation_id}: {error}")
        
        # Log to audit system
        await self._log_error(message, error)
        
        # Return user-friendly error
        return {
            "conversation_id": message.conversation_id,
            "response": {
                "status": "error",
                "message": (
                    "I encountered an error processing your request. "
                    "Our technical team has been notified. "
                    "Please try again or rephrase your question."
                ),
                "error_id": message.message_id
            },
            "agent": "error_handler",
            "confidence": 0,
            "requires_human": True,
            "processing_time_ms": 0
        }
    
    async def _log_error(self, message: AgentMessage, error: str):
        """
        Log error to monitoring and audit systems.
        
        Args:
            message: Original message
            error: Error details
        """
        error_log = {
            "timestamp": datetime.now().isoformat(),
            "conversation_id": message.conversation_id,
            "message_id": message.message_id,
            "error": error,
            "user_id": message.metadata.get("user_id", "unknown"),
            "query": message.content.get("query", "")
        }
        
        # In production, this would send to ELK stack or similar
        logger.error(f"Error log: {json.dumps(error_log)}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive orchestrator performance metrics.
        
        Returns:
            Dictionary with various performance metrics
        """
        if not self.enable_metrics:
            return {"metrics_disabled": True}
        
        # Calculate averages
        avg_processing_time = (
            sum(self.metrics["processing_times"]) / len(self.metrics["processing_times"])
            if self.metrics["processing_times"] else 0
        )
        
        avg_confidence = (
            sum(self.metrics["confidence_scores"]) / len(self.metrics["confidence_scores"])
            if self.metrics["confidence_scores"] else 0
        )
        
        # Agent usage statistics
        agent_usage = {}
        for agent in self.metrics["agent_usage"]:
            agent_usage[agent] = agent_usage.get(agent, 0) + 1
        
        return {
            "total_conversations": len(self.active_conversations),
            "total_requests": len(self.metrics["processing_times"]),
            "avg_processing_time_ms": avg_processing_time,
            "avg_confidence": avg_confidence,
            "escalation_rate": len(self.metrics["escalation_rates"]) / max(len(self.metrics["processing_times"]), 1),
            "error_rate": len(self.metrics["error_rates"]) / max(len(self.metrics["processing_times"]), 1),
            "agent_usage": agent_usage,
            "agent_status": {
                name: agent.get_metrics()
                for name, agent in self.agents.items()
            }
        }
    
    def reset_metrics(self):
        """Reset all performance metrics."""
        self.metrics = defaultdict(list)
        for agent in self.agents.values():
            agent.reset_metrics()
        logger.info("All metrics reset")