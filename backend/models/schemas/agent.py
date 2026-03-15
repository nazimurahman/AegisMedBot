"""
Agent schemas module for AegisMedBot.

This module defines Pydantic models for agent-related data structures,
ensuring type safety and validation for all agent interactions within
the hospital intelligence platform.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
import uuid
import re

from ..enums import AgentStatus, AgentType, MessageType, ConfidenceLevel


class AgentBase(BaseModel):
    """
    Base Pydantic model for all agents.
    
    This is the foundational schema that defines the core attributes
    every agent must have. All other agent schemas inherit from this.
    
    Attributes:
        name: Unique identifier name for the agent
        agent_type: Type of agent from the AgentType enum
        version: Semantic version of the agent
        description: Human-readable description of agent capabilities
        is_active: Whether the agent is currently operational
    """
    
    name: str = Field(
        ...,
        min_length=3,
        max_length=100,
        regex="^[a-zA-Z0-9_-]+$",
        description="Unique agent name (alphanumeric, underscore, hyphen only)"
    )
    
    agent_type: AgentType = Field(
        ...,
        description="Type of agent from predefined categories"
    )
    
    version: str = Field(
        "1.0.0",
        regex="^\d+\.\d+\.\d+$",
        description="Semantic version number"
    )
    
    description: Optional[str] = Field(
        None,
        max_length=500,
        description="Detailed description of agent capabilities"
    )
    
    is_active: bool = Field(
        True,
        description="Whether the agent is currently operational"
    )
    
    @validator("name")
    def validate_agent_name(cls, value: str) -> str:
        """
        Validate agent name format.
        
        Ensures agent names follow naming conventions and don't contain
        invalid characters that could cause issues in routing.
        
        Args:
            value: The agent name to validate
            
        Returns:
            Validated agent name
            
        Raises:
            ValueError: If name format is invalid
        """
        # Check for minimum length after stripping
        stripped = value.strip()
        if len(stripped) < 3:
            raise ValueError("Agent name must be at least 3 characters")
        
        # Convert to lowercase for consistency
        return stripped.lower()


class AgentCreate(AgentBase):
    """
    Schema for creating a new agent.
    
    Extends AgentBase with additional fields required only during
    agent creation, such as initial configuration and capabilities.
    
    Attributes:
        config: Initial configuration dictionary for the agent
        capabilities: List of specific capabilities the agent possesses
        max_concurrent_tasks: Maximum number of tasks agent can handle
        timeout_seconds: Maximum processing time before timeout
    """
    
    config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Initial agent configuration parameters"
    )
    
    capabilities: List[str] = Field(
        default_factory=list,
        description="List of specific agent capabilities"
    )
    
    max_concurrent_tasks: int = Field(
        5,
        ge=1,
        le=100,
        description="Maximum concurrent tasks this agent can handle"
    )
    
    timeout_seconds: int = Field(
        30,
        ge=1,
        le=300,
        description="Maximum processing time in seconds"
    )
    
    @validator("capabilities")
    def validate_capabilities(cls, value: List[str]) -> List[str]:
        """
        Validate and normalize capability list.
        
        Ensures capabilities are properly formatted and unique.
        
        Args:
            value: List of capability strings
            
        Returns:
            Deduplicated and normalized capability list
        """
        if not value:
            return value
        
        # Remove duplicates while preserving order
        seen = set()
        unique_capabilities = []
        
        for cap in value:
            normalized = cap.lower().strip()
            if normalized not in seen and normalized:
                seen.add(normalized)
                unique_capabilities.append(normalized)
        
        return unique_capabilities


class AgentUpdate(BaseModel):
    """
    Schema for updating an existing agent.
    
    All fields are optional to allow partial updates. Only provided
    fields will be modified.
    
    Attributes:
        description: Updated description
        is_active: Updated operational status
        config: Updated configuration
        capabilities: Updated capabilities list
        version: Updated version number
        max_concurrent_tasks: Updated task limit
        timeout_seconds: Updated timeout value
    """
    
    description: Optional[str] = Field(
        None,
        max_length=500,
        description="Updated agent description"
    )
    
    is_active: Optional[bool] = Field(
        None,
        description="Updated operational status"
    )
    
    config: Optional[Dict[str, Any]] = Field(
        None,
        description="Updated configuration"
    )
    
    capabilities: Optional[List[str]] = Field(
        None,
        description="Updated capabilities list"
    )
    
    version: Optional[str] = Field(
        None,
        regex="^\d+\.\d+\.\d+$",
        description="Updated version number"
    )
    
    max_concurrent_tasks: Optional[int] = Field(
        None,
        ge=1,
        le=100,
        description="Updated concurrent task limit"
    )
    
    timeout_seconds: Optional[int] = Field(
        None,
        ge=1,
        le=300,
        description="Updated timeout value"
    )


class AgentInDB(AgentBase):
    """
    Schema for agent as stored in database.
    
    Extends AgentBase with database-specific fields like ID and timestamps.
    This represents the complete agent record from the database.
    
    Attributes:
        id: Unique database identifier
        config: Current agent configuration
        capabilities: Current agent capabilities
        max_concurrent_tasks: Current task limit
        timeout_seconds: Current timeout value
        current_tasks: Number of currently active tasks
        total_tasks_processed: Lifetime task count
        average_response_time_ms: Average response time in milliseconds
        last_heartbeat: Last time agent reported healthy
        created_at: Timestamp of agent creation
        updated_at: Timestamp of last update
        metadata: Additional flexible metadata
    """
    
    id: str = Field(
        ...,
        description="Unique database identifier"
    )
    
    config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Current agent configuration"
    )
    
    capabilities: List[str] = Field(
        default_factory=list,
        description="Current agent capabilities"
    )
    
    max_concurrent_tasks: int = Field(
        5,
        description="Maximum concurrent tasks"
    )
    
    timeout_seconds: int = Field(
        30,
        description="Maximum processing time in seconds"
    )
    
    current_tasks: int = Field(
        0,
        ge=0,
        description="Number of currently active tasks"
    )
    
    total_tasks_processed: int = Field(
        0,
        ge=0,
        description="Total tasks processed over lifetime"
    )
    
    average_response_time_ms: float = Field(
        0.0,
        ge=0.0,
        description="Average response time in milliseconds"
    )
    
    last_heartbeat: Optional[datetime] = Field(
        None,
        description="Last time agent reported healthy"
    )
    
    created_at: datetime = Field(
        ...,
        description="Timestamp of agent creation"
    )
    
    updated_at: datetime = Field(
        ...,
        description="Timestamp of last update"
    )
    
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional flexible metadata"
    )
    
    class Config:
        """Pydantic configuration for the schema."""
        
        # Allow ORM mode to work with SQLAlchemy models
        orm_mode = True
        
        # Example configuration for API documentation
        schema_extra = {
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "name": "clinical_knowledge_agent",
                "agent_type": "clinical",
                "version": "2.1.0",
                "description": "Provides evidence-based medical information",
                "is_active": True,
                "capabilities": ["drug_interaction", "guideline_retrieval"],
                "current_tasks": 3,
                "total_tasks_processed": 15420,
                "average_response_time_ms": 234.5,
                "created_at": "2025-01-15T10:30:00Z",
                "updated_at": "2025-03-14T15:45:00Z"
            }
        }


class AgentMessage(BaseModel):
    """
    Schema for messages exchanged between agents.
    
    This defines the standard communication protocol for all
    agent-to-agent and agent-to-orchestrator messaging.
    
    Attributes:
        message_id: Unique identifier for this message
        conversation_id: ID of the conversation this message belongs to
        sender: Name of the sending agent
        recipient: Name of the receiving agent or "orchestrator"
        message_type: Type of message from MessageType enum
        content: Main message payload
        parent_message_id: ID of message this is responding to
        requires_human: Whether this requires human intervention
        human_approved: Whether human has approved (if required)
        confidence: Confidence score of the response
        processing_time_ms: Time taken to process
        error: Error message if any
        metadata: Additional message metadata
        created_at: Timestamp of message creation
    """
    
    message_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique message identifier"
    )
    
    conversation_id: str = Field(
        ...,
        description="Conversation this message belongs to"
    )
    
    sender: str = Field(
        ...,
        description="Name of sending agent"
    )
    
    recipient: str = Field(
        ...,
        description="Name of receiving agent or orchestrator"
    )
    
    message_type: MessageType = Field(
        ...,
        description="Type of message"
    )
    
    content: Dict[str, Any] = Field(
        ...,
        description="Message payload"
    )
    
    parent_message_id: Optional[str] = Field(
        None,
        description="Parent message ID for threaded conversations"
    )
    
    requires_human: bool = Field(
        False,
        description="Whether this requires human intervention"
    )
    
    human_approved: Optional[bool] = Field(
        None,
        description="Human approval status if required"
    )
    
    confidence: float = Field(
        1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score of the response"
    )
    
    processing_time_ms: Optional[float] = Field(
        None,
        ge=0.0,
        description="Processing time in milliseconds"
    )
    
    error: Optional[str] = Field(
        None,
        description="Error message if any"
    )
    
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional message metadata"
    )
    
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="Message creation timestamp"
    )
    
    @validator("sender")
    def validate_sender(cls, value: str) -> str:
        """
        Validate sender name format.
        
        Args:
            value: Sender name to validate
            
        Returns:
            Validated sender name
        """
        if not value or len(value.strip()) < 2:
            raise ValueError("Sender name must be at least 2 characters")
        return value.strip().lower()
    
    @validator("recipient")
    def validate_recipient(cls, value: str) -> str:
        """
        Validate recipient name format.
        
        Args:
            value: Recipient name to validate
            
        Returns:
            Validated recipient name
        """
        if not value or len(value.strip()) < 2:
            raise ValueError("Recipient name must be at least 2 characters")
        return value.strip().lower()


class AgentResponse(BaseModel):
    """
    Schema for agent processing responses.
    
    Standardized response format that all agents must return after
    processing a message. Used for tracking and auditing.
    
    Attributes:
        message_id: ID of the original message
        content: Response content
        tool_results: Results from any tools used
        confidence: Confidence level of the response
        next_agent: Optional next agent in processing chain
        requires_human_confirmation: Whether human confirmation needed
        error: Error details if processing failed
        processing_time_ms: Time taken to process
        agent_name: Name of agent that generated response
        agent_version: Version of agent that generated response
        warnings: Any warnings generated during processing
        suggestions: Suggested follow-up actions
    """
    
    message_id: str = Field(
        ...,
        description="ID of the original message"
    )
    
    content: Dict[str, Any] = Field(
        ...,
        description="Response content"
    )
    
    tool_results: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Results from tools used"
    )
    
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence level of the response"
    )
    
    confidence_level: ConfidenceLevel = Field(
        ...,
        description="Human-readable confidence level"
    )
    
    next_agent: Optional[str] = Field(
        None,
        description="Next agent in processing chain if any"
    )
    
    requires_human_confirmation: bool = Field(
        False,
        description="Whether human confirmation is needed"
    )
    
    error: Optional[str] = Field(
        None,
        description="Error details if processing failed"
    )
    
    processing_time_ms: float = Field(
        ...,
        ge=0.0,
        description="Time taken to process in milliseconds"
    )
    
    agent_name: str = Field(
        ...,
        description="Name of agent that generated response"
    )
    
    agent_version: str = Field(
        ...,
        description="Version of agent that generated response"
    )
    
    warnings: List[str] = Field(
        default_factory=list,
        description="Any warnings generated during processing"
    )
    
    suggestions: List[str] = Field(
        default_factory=list,
        description="Suggested follow-up actions"
    )
    
    @validator("confidence_level", always=True)
    def set_confidence_level(cls, value: Optional[ConfidenceLevel], values: Dict) -> ConfidenceLevel:
        """
        Automatically set confidence level based on confidence score.
        
        Args:
            value: Provided confidence level (if any)
            values: All field values including confidence
            
        Returns:
            Appropriate ConfidenceLevel enum value
        """
        if value is not None:
            return value
        
        confidence = values.get("confidence", 0.5)
        
        if confidence >= 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif confidence >= 0.7:
            return ConfidenceLevel.HIGH
        elif confidence >= 0.5:
            return ConfidenceLevel.MEDIUM
        elif confidence >= 0.3:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    class Config:
        """Pydantic configuration."""
        
        schema_extra = {
            "example": {
                "message_id": "msg_123456",
                "content": {
                    "response": "Based on current guidelines...",
                    "sources": ["PubMed:12345", "Guideline:2025"]
                },
                "confidence": 0.92,
                "confidence_level": "very_high",
                "processing_time_ms": 245.6,
                "agent_name": "clinical_knowledge_agent",
                "agent_version": "2.1.0"
            }
        }


class AgentTask(BaseModel):
    """
    Schema for tasks assigned to agents.
    
    Represents a unit of work for an agent to process, including
    all necessary context and requirements.
    
    Attributes:
        task_id: Unique task identifier
        agent_name: Name of assigned agent
        task_type: Type of task
        input_data: Input data for processing
        priority: Task priority (higher = more important)
        status: Current task status
        created_at: Task creation time
        started_at: When processing started
        completed_at: When processing completed
        result: Processing result
        error: Error if any
        timeout_seconds: Maximum allowed processing time
        retry_count: Number of retry attempts
        max_retries: Maximum allowed retries
        metadata: Additional task metadata
    """
    
    task_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique task identifier"
    )
    
    agent_name: str = Field(
        ...,
        description="Name of assigned agent"
    )
    
    task_type: str = Field(
        ...,
        description="Type of task"
    )
    
    input_data: Dict[str, Any] = Field(
        ...,
        description="Input data for processing"
    )
    
    priority: int = Field(
        1,
        ge=0,
        le=10,
        description="Task priority (0-10, higher is more important)"
    )
    
    status: AgentStatus = Field(
        AgentStatus.PENDING,
        description="Current task status"
    )
    
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="Task creation timestamp"
    )
    
    started_at: Optional[datetime] = Field(
        None,
        description="When processing started"
    )
    
    completed_at: Optional[datetime] = Field(
        None,
        description="When processing completed"
    )
    
    result: Optional[Dict[str, Any]] = Field(
        None,
        description="Processing result"
    )
    
    error: Optional[str] = Field(
        None,
        description="Error if processing failed"
    )
    
    timeout_seconds: int = Field(
        30,
        ge=1,
        le=300,
        description="Maximum allowed processing time"
    )
    
    retry_count: int = Field(
        0,
        ge=0,
        le=10,
        description="Number of retry attempts"
    )
    
    max_retries: int = Field(
        3,
        ge=0,
        le=10,
        description="Maximum allowed retries"
    )
    
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional task metadata"
    )
    
    def is_expired(self) -> bool:
        """
        Check if the task has expired based on timeout.
        
        Returns:
            True if task has expired, False otherwise
        """
        if self.started_at and not self.completed_at:
            elapsed = (datetime.now() - self.started_at).total_seconds()
            return elapsed > self.timeout_seconds
        return False
    
    def can_retry(self) -> bool:
        """
        Check if the task can be retried.
        
        Returns:
            True if retry is possible, False otherwise
        """
        return self.retry_count < self.max_retries


class AgentMetrics(BaseModel):
    """
    Schema for agent performance metrics.
    
    Tracks various performance indicators for monitoring and optimization.
    
    Attributes:
        agent_name: Name of the agent
        period_start: Start of metrics period
        period_end: End of metrics period
        tasks_processed: Number of tasks processed
        tasks_succeeded: Number of successful tasks
        tasks_failed: Number of failed tasks
        tasks_timed_out: Number of timed out tasks
        average_processing_time_ms: Average processing time
        p95_processing_time_ms: 95th percentile processing time
        p99_processing_time_ms: 99th percentile processing time
        average_confidence: Average confidence score
        error_rate: Error rate as percentage
        uptime_percentage: Agent uptime percentage
        current_tasks: Currently active tasks
        peak_concurrent_tasks: Peak concurrent tasks
        memory_usage_mb: Memory usage in MB
        cpu_percentage: CPU usage percentage
        last_updated: Last metrics update time
    """
    
    agent_name: str = Field(
        ...,
        description="Name of the agent"
    )
    
    period_start: datetime = Field(
        ...,
        description="Start of metrics period"
    )
    
    period_end: datetime = Field(
        ...,
        description="End of metrics period"
    )
    
    tasks_processed: int = Field(
        0,
        ge=0,
        description="Total tasks processed"
    )
    
    tasks_succeeded: int = Field(
        0,
        ge=0,
        description="Successful tasks"
    )
    
    tasks_failed: int = Field(
        0,
        ge=0,
        description="Failed tasks"
    )
    
    tasks_timed_out: int = Field(
        0,
        ge=0,
        description="Timed out tasks"
    )
    
    average_processing_time_ms: float = Field(
        0.0,
        ge=0.0,
        description="Average processing time"
    )
    
    p95_processing_time_ms: float = Field(
        0.0,
        ge=0.0,
        description="95th percentile processing time"
    )
    
    p99_processing_time_ms: float = Field(
        0.0,
        ge=0.0,
        description="99th percentile processing time"
    )
    
    average_confidence: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Average confidence score"
    )
    
    error_rate: float = Field(
        0.0,
        ge=0.0,
        le=100.0,
        description="Error rate percentage"
    )
    
    uptime_percentage: float = Field(
        100.0,
        ge=0.0,
        le=100.0,
        description="Agent uptime percentage"
    )
    
    current_tasks: int = Field(
        0,
        ge=0,
        description="Currently active tasks"
    )
    
    peak_concurrent_tasks: int = Field(
        0,
        ge=0,
        description="Peak concurrent tasks"
    )
    
    memory_usage_mb: float = Field(
        0.0,
        ge=0.0,
        description="Memory usage in MB"
    )
    
    cpu_percentage: float = Field(
        0.0,
        ge=0.0,
        le=100.0,
        description="CPU usage percentage"
    )
    
    last_updated: datetime = Field(
        default_factory=datetime.now,
        description="Last metrics update time"
    )
    
    @validator("error_rate", always=True)
    def calculate_error_rate(cls, value: float, values: Dict) -> float:
        """
        Calculate error rate if not provided.
        
        Args:
            value: Provided error rate
            values: Other field values
            
        Returns:
            Calculated error rate
        """
        if value > 0 or "tasks_processed" not in values:
            return value
        
        tasks_processed = values.get("tasks_processed", 0)
        tasks_failed = values.get("tasks_failed", 0)
        
        if tasks_processed > 0:
            return (tasks_failed / tasks_processed) * 100
        
        return 0.0
    
    def get_success_rate(self) -> float:
        """
        Calculate success rate.
        
        Returns:
            Success rate as percentage
        """
        if self.tasks_processed > 0:
            return (self.tasks_succeeded / self.tasks_processed) * 100
        return 100.0