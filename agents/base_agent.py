"""
Base agent class that defines the foundation for all specialized agents in the AegisMedBot system.
This abstract base class ensures consistency across all agent implementations and provides
common functionality for message handling, status management, and communication protocols.
"""

from abc import ABC, abstractmethod  # Abstract base classes for defining interfaces
from typing import Dict, Any, Optional, List  # Type hints for better code documentation
from pydantic import BaseModel, Field  # Data validation and serialization
from datetime import datetime  # Timestamp handling
import asyncio  # Asynchronous operations
import uuid  # Unique identifier generation
from enum import Enum  # Enumerated types for status values
import logging  # Logging functionality

# Configure logger for this module
logger = logging.getLogger(__name__)

class AgentStatus(Enum):
    """
    Enumeration of possible agent states.
    Used for monitoring and orchestration purposes.
    """
    IDLE = "idle"  # Agent is ready to accept tasks
    PROCESSING = "processing"  # Agent is currently working on a task
    WAITING_FOR_HUMAN = "waiting_for_human"  # Agent requires human intervention
    ERROR = "error"  # Agent encountered an error
    COMPLETED = "completed"  # Agent finished processing

class AgentMessage(BaseModel):
    """
    Standardized message format for all agent-to-agent communication.
    Ensures consistent message structure across the entire system.
    
    Attributes:
        message_id: Unique identifier for tracking and auditing
        conversation_id: Groups related messages in a conversation
        sender: Name of the sending agent
        recipient: Name of the receiving agent
        message_type: Type of message (request, response, escalation, error)
        content: Actual message payload
        tools_requested: Optional list of tools needed
        requires_human: Whether this message needs human review
        human_approved: Human approval status if applicable
        confidence: Agent's confidence in the response (0-1)
        metadata: Additional context information
        created_at: Timestamp for audit purposes
        parent_message_id: For threading related messages
    """
    
    # Unique identifier for each message
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    
    # Conversation tracking
    conversation_id: str = Field(..., description="Unique conversation identifier")
    
    # Routing information
    sender: str = Field(..., description="Name of sending agent")
    recipient: str = Field(..., description="Name of receiving agent")
    
    # Message classification
    message_type: str = Field(..., description="Type: request, response, escalation, error")
    
    # Content and metadata
    content: Dict[str, Any] = Field(..., description="Actual message payload")
    tools_requested: Optional[List[str]] = Field(None, description="Tools needed for processing")
    
    # Human oversight fields
    requires_human: bool = Field(False, description="Whether human review is needed")
    human_approved: Optional[bool] = Field(None, description="Human approval status")
    
    # Quality metrics
    confidence: float = Field(1.0, description="Confidence score", ge=0.0, le=1.0)
    
    # Context and tracking
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional context")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    parent_message_id: Optional[str] = Field(None, description="Parent message for threading")

class AgentResponse(BaseModel):
    """
    Standardized response format from agents after processing.
    Includes processing metrics and optional tool results.
    
    Attributes:
        message_id: Reference to original message
        content: Response content
        tool_results: Results from any tools used
        confidence: Confidence in response
        next_agent: Optional next agent in processing chain
        requires_human_confirmation: Whether response needs human review
        error: Error message if any
        processing_time_ms: Time taken to process
    """
    
    message_id: str = Field(..., description="ID of original message")
    content: Dict[str, Any] = Field(..., description="Response content")
    tool_results: Optional[List[Dict[str, Any]]] = Field(None, description="Results from tools")
    confidence: float = Field(..., description="Confidence score", ge=0.0, le=1.0)
    next_agent: Optional[str] = Field(None, description="Next agent in chain")
    requires_human_confirmation: bool = Field(False, description="Needs human review")
    error: Optional[str] = Field(None, description="Error details if any")
    processing_time_ms: float = Field(0.0, description="Processing time in milliseconds")

class BaseAgent(ABC):
    """
    Abstract base class for all agents in the AegisMedBot system.
    Provides common functionality and enforces required methods.
    
    This class implements:
    - Standardized message processing interface
    - Status management
    - Human escalation workflows
    - Input validation
    - Audit logging
    """
    
    def __init__(
        self,
        name: str,  # Unique agent name
        role: str,  # Agent's role description
        description: str,  # Detailed agent description
        config: Optional[Dict[str, Any]] = None  # Configuration parameters
    ):
        """
        Initialize a new agent with basic properties.
        
        Args:
            name: Unique identifier for the agent
            role: Functional role of the agent
            description: Detailed description of agent capabilities
            config: Optional configuration dictionary
        """
        self.name = name
        self.role = role
        self.description = description
        self.config = config or {}
        
        # Initialize status tracking
        self.status = AgentStatus.IDLE
        self.current_task = None
        
        # Initialize conversation context storage
        self.conversation_context = {}
        
        # Track performance metrics
        self.metrics = {
            "tasks_processed": 0,
            "total_processing_time": 0,
            "errors": 0,
            "escalations": 0
        }
        
        logger.info(f"Initialized agent: {name} - {role}")
    
    @abstractmethod
    async def process(self, message: AgentMessage) -> AgentResponse:
        """
        Abstract method that must be implemented by all concrete agents.
        Processes an incoming message and returns a response.
        
        Args:
            message: Incoming AgentMessage to process
            
        Returns:
            AgentResponse containing processing results
        """
        pass
    
    async def can_handle(self, task_type: str, context: Dict[str, Any]) -> float:
        """
        Determine if this agent can handle a specific task type.
        Returns a confidence score between 0 and 1.
        
        Args:
            task_type: Type of task to evaluate
            context: Current conversation context
            
        Returns:
            Confidence score (0-1) indicating ability to handle task
        """
        # Default implementation returns 0.5
        # Concrete agents should override with their own logic
        return 0.5
    
    def update_status(self, status: AgentStatus, task_id: Optional[str] = None):
        """
        Update the agent's current status and optionally track the current task.
        
        Args:
            status: New status value
            task_id: Optional task identifier
        """
        old_status = self.status
        self.status = status
        if task_id:
            self.current_task = task_id
        
        logger.debug(f"Agent {self.name} status changed: {old_status.value} -> {status.value}")
    
    async def escalate_to_human(self, message: AgentMessage, reason: str) -> AgentResponse:
        """
        Escalate a message to human oversight when the agent cannot handle it confidently.
        
        Args:
            message: Original message that needs escalation
            reason: Reason for escalation
            
        Returns:
            AgentResponse indicating escalation
        """
        self.update_status(AgentStatus.WAITING_FOR_HUMAN, message.message_id)
        self.metrics["escalations"] += 1
        
        logger.info(f"Agent {self.name} escalating to human: {reason}")
        
        # Create escalation response
        return AgentResponse(
            message_id=message.message_id,
            content={
                "status": "escalated",
                "reason": reason,
                "human_intervention_required": True,
                "original_query": message.content.get("query", ""),
                "agent_name": self.name,
                "agent_role": self.role
            },
            confidence=0.0,
            requires_human_confirmation=True,
            processing_time_ms=0
        )
    
    def validate_input(self, message: AgentMessage) -> bool:
        """
        Validate incoming message format and required fields.
        
        Args:
            message: Message to validate
            
        Returns:
            True if message is valid, False otherwise
        """
        # Check required fields
        required_fields = ["conversation_id", "content"]
        for field in required_fields:
            if not hasattr(message, field) or getattr(message, field) is None:
                logger.warning(f"Message missing required field: {field}")
                return False
        
        # Check content has query
        if "query" not in message.content:
            logger.warning("Message content missing 'query' field")
            return False
        
        return True
    
    async def log_interaction(self, message: AgentMessage, response: AgentResponse):
        """
        Log agent interaction for audit and monitoring purposes.
        
        Args:
            message: Original incoming message
            response: Generated response
        """
        # In production, this would send to a centralized logging service
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "agent_name": self.name,
            "conversation_id": message.conversation_id,
            "message_id": message.message_id,
            "message_type": message.message_type,
            "requires_human": message.requires_human,
            "response_confidence": response.confidence,
            "processing_time_ms": response.processing_time_ms,
            "has_error": response.error is not None
        }
        
        # Update metrics
        self.metrics["tasks_processed"] += 1
        self.metrics["total_processing_time"] += response.processing_time_ms
        
        if response.error:
            self.metrics["errors"] += 1
        
        logger.info(f"Interaction logged: {log_entry}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current agent metrics for monitoring.
        
        Returns:
            Dictionary of performance metrics
        """
        avg_processing_time = 0
        if self.metrics["tasks_processed"] > 0:
            avg_processing_time = (
                self.metrics["total_processing_time"] / 
                self.metrics["tasks_processed"]
            )
        
        return {
            "agent_name": self.name,
            "agent_role": self.role,
            "status": self.status.value,
            "current_task": self.current_task,
            "tasks_processed": self.metrics["tasks_processed"],
            "avg_processing_time_ms": avg_processing_time,
            "errors": self.metrics["errors"],
            "escalations": self.metrics["escalations"],
            "error_rate": (
                self.metrics["errors"] / self.metrics["tasks_processed"]
                if self.metrics["tasks_processed"] > 0 else 0
            )
        }
    
    def reset_metrics(self):
        """Reset all performance metrics for this agent."""
        self.metrics = {
            "tasks_processed": 0,
            "total_processing_time": 0,
            "errors": 0,
            "escalations": 0
        }
        logger.info(f"Metrics reset for agent {self.name}")