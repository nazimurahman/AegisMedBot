"""
Response schemas module for AegisMedBot.

This module defines standardized response formats for all API endpoints,
ensuring consistent error handling and data presentation across the platform.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Generic, TypeVar
from datetime import datetime
import uuid

# Define generic type variable for data payloads
T = TypeVar('T')


class ResponseMetadata(BaseModel):
    """
    Metadata for API responses.
    
    Contains information about the response itself, useful for debugging
    and monitoring.
    
    Attributes:
        request_id: Unique identifier for the request
        timestamp: When the response was generated
        processing_time_ms: Time taken to process the request
        api_version: Version of the API that generated the response
        agent_name: Name of agent that handled the request (if applicable)
    """
    
    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique request identifier for tracing"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Response generation timestamp"
    )
    
    processing_time_ms: Optional[float] = Field(
        None,
        ge=0.0,
        description="Request processing time in milliseconds"
    )
    
    api_version: str = Field(
        "1.0.0",
        description="API version that generated this response"
    )
    
    agent_name: Optional[str] = Field(
        None,
        description="Agent that handled the request if applicable"
    )


class ApiResponse(BaseModel, Generic[T]):
    """
    Generic API response wrapper.
    
    Provides a consistent structure for all API responses, including
    success status, data payload, and metadata.
    
    Attributes:
        success: Whether the request was successful
        data: Response data payload (generic type)
        message: Human-readable status message
        error: Error details if any
        metadata: Response metadata
    """
    
    success: bool = Field(
        ...,
        description="Whether the request was successful"
    )
    
    data: Optional[T] = Field(
        None,
        description="Response data payload"
    )
    
    message: Optional[str] = Field(
        None,
        description="Human-readable status message"
    )
    
    error: Optional[Dict[str, Any]] = Field(
        None,
        description="Error details if request failed"
    )
    
    metadata: ResponseMetadata = Field(
        default_factory=ResponseMetadata,
        description="Response metadata"
    )
    
    class Config:
        """Pydantic configuration."""
        
        schema_extra = {
            "example": {
                "success": True,
                "data": {"key": "value"},
                "message": "Request processed successfully",
                "metadata": {
                    "request_id": "req_123456",
                    "timestamp": "2025-03-14T12:00:00Z",
                    "processing_time_ms": 45.2,
                    "api_version": "1.0.0"
                }
            }
        }


class ErrorDetail(BaseModel):
    """
    Detailed error information for failed requests.
    
    Provides structured error data to help clients understand and
    handle errors appropriately.
    
    Attributes:
        code: Error code for programmatic handling
        message: Human-readable error message
        details: Additional error details
        field: Specific field that caused the error (for validation)
        suggestion: Suggested action to resolve the error
        help_url: URL to documentation for this error
    """
    
    code: str = Field(
        ...,
        description="Error code for programmatic handling"
    )
    
    message: str = Field(
        ...,
        description="Human-readable error message"
    )
    
    details: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional error details"
    )
    
    field: Optional[str] = Field(
        None,
        description="Specific field that caused the error"
    )
    
    suggestion: Optional[str] = Field(
        None,
        description="Suggested action to resolve the error"
    )
    
    help_url: Optional[str] = Field(
        None,
        description="URL to documentation for this error"
    )


class ErrorResponse(ApiResponse[None]):
    """
    Specialized response for error cases.
    
    Ensures consistent error format across all endpoints.
    
    Attributes:
        success: Always False for error responses
        error: Detailed error information
    """
    
    success: bool = Field(
        False,
        const=True,
        description="Always false for error responses"
    )
    
    data: None = Field(
        None,
        description="No data for error responses"
    )
    
    error: ErrorDetail = Field(
        ...,
        description="Detailed error information"
    )
    
    class Config:
        """Pydantic configuration."""
        
        schema_extra = {
            "example": {
                "success": False,
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "Invalid input data",
                    "field": "patient_id",
                    "suggestion": "Provide a valid patient ID",
                    "help_url": "https://docs.example.com/errors/validation"
                },
                "metadata": {
                    "request_id": "req_123456",
                    "timestamp": "2025-03-14T12:00:00Z",
                    "api_version": "1.0.0"
                }
            }
        }


class PaginatedResponse(BaseModel, Generic[T]):
    """
    Paginated response wrapper for list endpoints.
    
    Provides pagination metadata along with the data payload.
    
    Attributes:
        items: List of items for current page
        total: Total number of items across all pages
        page: Current page number
        page_size: Number of items per page
        total_pages: Total number of pages
        has_next: Whether there is a next page
        has_previous: Whether there is a previous page
        next_page_url: URL to next page if available
        previous_page_url: URL to previous page if available
    """
    
    items: List[T] = Field(
        ...,
        description="Items for current page"
    )
    
    total: int = Field(
        ...,
        ge=0,
        description="Total number of items across all pages"
    )
    
    page: int = Field(
        ...,
        ge=1,
        description="Current page number (1-based)"
    )
    
    page_size: int = Field(
        ...,
        ge=1,
        le=100,
        description="Number of items per page"
    )
    
    total_pages: int = Field(
        ...,
        ge=0,
        description="Total number of pages"
    )
    
    has_next: bool = Field(
        ...,
        description="Whether there is a next page"
    )
    
    has_previous: bool = Field(
        ...,
        description="Whether there is a previous page"
    )
    
    next_page_url: Optional[str] = Field(
        None,
        description="URL to next page if available"
    )
    
    previous_page_url: Optional[str] = Field(
        None,
        description="URL to previous page if available"
    )
    
    @validator("total_pages", always=True)
    def calculate_total_pages(cls, value: int, values: Dict) -> int:
        """
        Calculate total pages if not provided.
        
        Args:
            value: Provided total pages
            values: Other field values
            
        Returns:
            Calculated total pages
        """
        if value > 0:
            return value
        
        total = values.get("total", 0)
        page_size = values.get("page_size", 10)
        
        if total > 0 and page_size > 0:
            return (total + page_size - 1) // page_size
        
        return 0
    
    @validator("has_next", always=True)
    def calculate_has_next(cls, value: bool, values: Dict) -> bool:
        """
        Calculate has_next flag if not provided.
        
        Args:
            value: Provided has_next
            values: Other field values
            
        Returns:
            Calculated has_next flag
        """
        if value is not None:
            return value
        
        page = values.get("page", 1)
        total_pages = values.get("total_pages", 0)
        
        return page < total_pages
    
    @validator("has_previous", always=True)
    def calculate_has_previous(cls, value: bool, values: Dict) -> bool:
        """
        Calculate has_previous flag if not provided.
        
        Args:
            value: Provided has_previous
            values: Other field values
            
        Returns:
            Calculated has_previous flag
        """
        if value is not None:
            return value
        
        page = values.get("page", 1)
        
        return page > 1


class ChatResponse(BaseModel):
    """
    Response format for chat interactions.
    
    Specialized response structure for the main chat interface,
    including conversation context and metadata.
    
    Attributes:
        conversation_id: ID of the ongoing conversation
        response: Main response text
        agent: Name of agent that generated the response
        confidence: Confidence score of the response
        requires_human: Whether human intervention is needed
        processing_time_ms: Time taken to generate response
        suggestions: Suggested follow-up questions
        sources: Sources used to generate response
        warnings: Any warnings generated
        follow_up_questions: Suggested follow-up questions
        related_topics: Related topics for exploration
    """
    
    conversation_id: str = Field(
        ...,
        description="Conversation identifier for multi-turn interactions"
    )
    
    response: str = Field(
        ...,
        description="Main response text"
    )
    
    agent: str = Field(
        ...,
        description="Name of agent that generated the response"
    )
    
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score of the response"
    )
    
    requires_human: bool = Field(
        False,
        description="Whether human intervention is needed"
    )
    
    processing_time_ms: float = Field(
        ...,
        ge=0.0,
        description="Response generation time in milliseconds"
    )
    
    suggestions: List[str] = Field(
        default_factory=list,
        description="Suggested follow-up actions"
    )
    
    sources: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Sources used to generate the response"
    )
    
    warnings: List[str] = Field(
        default_factory=list,
        description="Any warnings generated"
    )
    
    follow_up_questions: List[str] = Field(
        default_factory=list,
        description="Suggested follow-up questions"
    )
    
    related_topics: List[str] = Field(
        default_factory=list,
        description="Related topics for exploration"
    )
    
    class Config:
        """Pydantic configuration."""
        
        schema_extra = {
            "example": {
                "conversation_id": "conv_123456",
                "response": "Based on current guidelines, sepsis management involves...",
                "agent": "clinical_knowledge_agent",
                "confidence": 0.94,
                "requires_human": False,
                "processing_time_ms": 234.5,
                "sources": [
                    {
                        "title": "Sepsis Guidelines 2025",
                        "source": "Society of Critical Care Medicine",
                        "url": "https://example.com/guidelines"
                    }
                ],
                "follow_up_questions": [
                    "What antibiotics are recommended?",
                    "When should I consider ICU transfer?"
                ]
            }
        }


class HealthCheckResponse(BaseModel):
    """
    Response format for health check endpoints.
    
    Provides comprehensive health status information for monitoring
    and orchestration systems.
    
    Attributes:
        status: Overall system status
        version: System version
        timestamp: Current server time
        uptime_seconds: System uptime in seconds
        services: Status of individual services
        database: Database connection status
        redis: Redis connection status
        qdrant: Qdrant connection status
        agents: Status of registered agents
        environment: Deployment environment
        commit_hash: Git commit hash of current deployment
    """
    
    status: str = Field(
        ...,
        regex="^(healthy|degraded|unhealthy)$",
        description="Overall system health status"
    )
    
    version: str = Field(
        ...,
        description="System version"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Current server time"
    )
    
    uptime_seconds: float = Field(
        ...,
        ge=0.0,
        description="System uptime in seconds"
    )
    
    services: Dict[str, str] = Field(
        ...,
        description="Status of individual services"
    )
    
    database: Dict[str, Any] = Field(
        ...,
        description="Database connection details"
    )
    
    redis: Dict[str, Any] = Field(
        ...,
        description="Redis connection details"
    )
    
    qdrant: Dict[str, Any] = Field(
        ...,
        description="Qdrant connection details"
    )
    
    agents: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Status of registered agents"
    )
    
    environment: str = Field(
        ...,
        description="Deployment environment"
    )
    
    commit_hash: Optional[str] = Field(
        None,
        description="Git commit hash of current deployment"
    )
    
    class Config:
        """Pydantic configuration."""
        
        schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "timestamp": "2025-03-14T12:00:00Z",
                "uptime_seconds": 86400,
                "services": {
                    "api": "healthy",
                    "auth": "healthy",
                    "agents": "healthy"
                },
                "database": {
                    "connected": True,
                    "latency_ms": 2.3
                },
                "environment": "production"
            }
        }


class AgentRegistrationResponse(BaseModel):
    """
    Response for agent registration endpoint.
    
    Provides confirmation and details after agent registration.
    
    Attributes:
        agent_id: Unique identifier for the registered agent
        name: Name of the registered agent
        status: Current agent status
        registration_time: When the agent was registered
        auth_token: Authentication token for the agent
        endpoints: API endpoints for the agent
        config: Assigned configuration
        capabilities: Registered capabilities
    """
    
    agent_id: str = Field(
        ...,
        description="Unique agent identifier"
    )
    
    name: str = Field(
        ...,
        description="Agent name"
    )
    
    status: str = Field(
        ...,
        description="Current agent status"
    )
    
    registration_time: datetime = Field(
        ...,
        description="Registration timestamp"
    )
    
    auth_token: Optional[str] = Field(
        None,
        description="Authentication token for agent API calls"
    )
    
    endpoints: Dict[str, str] = Field(
        default_factory=dict,
        description="API endpoints for the agent"
    )
    
    config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Assigned configuration"
    )
    
    capabilities: List[str] = Field(
        default_factory=list,
        description="Registered capabilities"
    )
    
    class Config:
        """Pydantic configuration."""
        
        schema_extra = {
            "example": {
                "agent_id": "agent_123456",
                "name": "clinical_knowledge_agent",
                "status": "active",
                "registration_time": "2025-03-14T12:00:00Z",
                "capabilities": ["drug_interaction", "guideline_retrieval"]
            }
        }


class AuditLogResponse(BaseModel):
    """
    Response format for audit log entries.
    
    Standardized format for all audit log entries to ensure
    consistent logging and easy searching.
    
    Attributes:
        log_id: Unique log entry identifier
        timestamp: When the event occurred
        user_id: ID of the user who performed the action
        user_role: Role of the user
        action: Action performed
        resource_type: Type of resource affected
        resource_id: ID of the affected resource
        details: Additional event details
        ip_address: IP address of the client
        user_agent: User agent of the client
        outcome: Outcome of the action (success/failure)
        error_code: Error code if action failed
        severity: Severity level of the event
    """
    
    log_id: str = Field(
        ...,
        description="Unique log entry identifier"
    )
    
    timestamp: datetime = Field(
        ...,
        description="When the event occurred"
    )
    
    user_id: str = Field(
        ...,
        description="User who performed the action"
    )
    
    user_role: str = Field(
        ...,
        description="Role of the user"
    )
    
    action: str = Field(
        ...,
        description="Action performed"
    )
    
    resource_type: str = Field(
        ...,
        description="Type of resource affected"
    )
    
    resource_id: Optional[str] = Field(
        None,
        description="ID of affected resource"
    )
    
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional event details"
    )
    
    ip_address: Optional[str] = Field(
        None,
        description="Client IP address"
    )
    
    user_agent: Optional[str] = Field(
        None,
        description="Client user agent"
    )
    
    outcome: str = Field(
        ...,
        regex="^(success|failure|pending)$",
        description="Action outcome"
    )
    
    error_code: Optional[str] = Field(
        None,
        description="Error code if action failed"
    )
    
    severity: str = Field(
        "info",
        regex="^(debug|info|warning|error|critical)$",
        description="Event severity"
    )
    
    class Config:
        """Pydantic configuration."""
        
        schema_extra = {
            "example": {
                "log_id": "log_123456",
                "timestamp": "2025-03-14T12:00:00Z",
                "user_id": "user_123",
                "user_role": "medical_director",
                "action": "CHAT_QUERY",
                "resource_type": "conversation",
                "resource_id": "conv_456",
                "outcome": "success",
                "severity": "info"
            }
        }