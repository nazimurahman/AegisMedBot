"""
Chat routes module for handling conversation endpoints.
This module provides REST and WebSocket endpoints for interacting with the AI agents.
"""

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, validator
from datetime import datetime
import asyncio
import json
import logging
import uuid

# Import core components
from ...core.config import settings
from ...core.security import get_current_user
from ...services.audit_service import AuditService
from ...services.notification_service import NotificationService
from ...models.schemas.response import ResponseModel, ErrorResponseModel

# Import agent orchestrator (to be implemented)
from agents.orchestrator.agent_orchestrator import AgentOrchestrator

# Configure logger for this module
logger = logging.getLogger(__name__)

# Create router instance with prefix and tags for API documentation
router = APIRouter(prefix="/chat", tags=["Chat"])

# Pydantic models for request/response validation

class ChatRequest(BaseModel):
    """
    Request model for chat endpoint.
    Validates incoming chat messages.
    """
    
    query: str = Field(
        ...,  # Ellipsis means required
        min_length=1,
        max_length=5000,
        description="The user's query or message"
    )
    
    conversation_id: Optional[str] = Field(
        None,
        description="Existing conversation ID for multi-turn conversations"
    )
    
    patient_id: Optional[str] = Field(
        None,
        description="Patient ID if query is patient-specific"
    )
    
    context: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional context for the query"
    )
    
    @validator('query')
    def validate_query(cls, value):
        """
        Custom validator to ensure query is not just whitespace.
        """
        if not value.strip():
            raise ValueError('Query cannot be empty or just whitespace')
        return value.strip()

class ChatResponse(BaseModel):
    """
    Response model for chat endpoint.
    Structures the AI agent's response.
    """
    
    conversation_id: str = Field(
        ..., 
        description="Unique identifier for the conversation"
    )
    
    message_id: str = Field(
        ...,
        description="Unique identifier for this specific message"
    )
    
    response: str = Field(
        ...,
        description="The AI agent's response text"
    )
    
    agent: str = Field(
        ...,
        description="Name of the agent that generated the response"
    )
    
    confidence: float = Field(
        ...,
        ge=0.0,  # Greater than or equal to 0
        le=1.0,  # Less than or equal to 1
        description="Confidence score of the response (0-1)"
    )
    
    requires_human: bool = Field(
        False,
        description="Whether human intervention is needed"
    )
    
    processing_time_ms: float = Field(
        ...,
        description="Time taken to process the request in milliseconds"
    )
    
    suggestions: Optional[List[str]] = Field(
        None,
        description="Suggested follow-up questions"
    )
    
    sources: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Sources used to generate the response"
    )
    
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the response"
    )

class ChatFeedback(BaseModel):
    """
    Model for collecting user feedback on chat responses.
    """
    
    conversation_id: str = Field(..., description="Conversation ID")
    message_id: str = Field(..., description="Message ID being rated")
    rating: int = Field(
        ...,
        ge=1,  # Minimum rating 1
        le=5,  # Maximum rating 5
        description="User rating (1-5 stars)"
    )
    feedback_text: Optional[str] = Field(
        None,
        max_length=1000,
        description="Optional text feedback"
    )
    tags: Optional[List[str]] = Field(
        None,
        description="Optional tags categorizing the feedback"
    )

# Dependency to get orchestrator instance
async def get_orchestrator() -> AgentOrchestrator:
    """
    Dependency injection for AgentOrchestrator.
    Creates or retrieves the orchestrator instance.
    """
    # This would typically be a singleton or from app state
    orchestrator = AgentOrchestrator()
    return orchestrator

@router.post("/", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator),
    current_user: Dict[str, Any] = Depends(get_current_user),
    audit_service: AuditService = Depends()
):
    """
    Main chat endpoint for sending messages to the AI agent system.
    
    This endpoint:
    1. Validates the incoming request
    2. Logs the interaction for audit
    3. Routes the query through the agent orchestrator
    4. Returns the structured response
    5. Records processing metrics
    
    Args:
        request: The validated chat request
        orchestrator: The agent orchestrator instance
        current_user: The authenticated user
        audit_service: Service for audit logging
    
    Returns:
        Structured chat response with agent output
        
    Raises:
        HTTPException: If processing fails
    """
    
    # Record start time for performance tracking
    start_time = datetime.now()
    
    # Generate unique IDs for tracking
    request_id = str(uuid.uuid4())
    message_id = str(uuid.uuid4())
    
    logger.info(f"Processing chat request {request_id} from user {current_user['id']}")
    
    try:
        # Audit the incoming request for compliance
        await audit_service.log_chat_request(
            user_id=current_user["id"],
            request_id=request_id,
            query=request.query,
            conversation_id=request.conversation_id,
            patient_id=request.patient_id,
            timestamp=datetime.now()
        )
        
        # Prepare message for orchestrator
        orchestrator_message = {
            "query": request.query,
            "patient_id": request.patient_id,
            "user_id": current_user["id"],
            "user_role": current_user.get("role", "unknown"),
            "context": request.context,
            "request_id": request_id,
            "message_id": message_id
        }
        
        # Process through agent orchestrator
        result = await orchestrator.process_message(
            message=orchestrator_message,
            conversation_id=request.conversation_id
        )
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Build structured response
        response = ChatResponse(
            conversation_id=result.get("conversation_id", str(uuid.uuid4())),
            message_id=message_id,
            response=result.get("response", {}).get("text", "No response generated"),
            agent=result.get("agent", "unknown"),
            confidence=result.get("confidence", 0.0),
            requires_human=result.get("requires_human", False),
            processing_time_ms=processing_time,
            suggestions=result.get("response", {}).get("suggestions", []),
            sources=result.get("response", {}).get("sources", []),
            metadata={
                "request_id": request_id,
                "model_used": result.get("model_used", "unknown"),
                "agent_chain": result.get("agent_chain", [])
            }
        )
        
        # Log successful response for audit
        await audit_service.log_chat_response(
            user_id=current_user["id"],
            conversation_id=response.conversation_id,
            message_id=message_id,
            response_summary=response.response[:200],  # Log only preview
            processing_time_ms=processing_time,
            timestamp=datetime.now()
        )
        
        logger.info(f"Chat request {request_id} completed in {processing_time:.2f}ms")
        
        return response
        
    except ValueError as ve:
        # Handle validation errors
        logger.warning(f"Validation error in chat request {request_id}: {str(ve)}")
        raise HTTPException(status_code=422, detail=str(ve))
        
    except TimeoutError as te:
        # Handle timeout errors
        logger.error(f"Timeout in chat request {request_id}: {str(te)}")
        raise HTTPException(status_code=504, detail="Request timeout")
        
    except Exception as e:
        # Handle unexpected errors
        logger.error(f"Unexpected error in chat request {request_id}: {str(e)}", exc_info=True)
        
        # Log error for audit
        await audit_service.log_error(
            user_id=current_user["id"],
            request_id=request_id,
            error=str(e),
            timestamp=datetime.now()
        )
        
        raise HTTPException(status_code=500, detail="Internal server error")

@router.websocket("/ws")
async def websocket_chat_endpoint(
    websocket: WebSocket,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator)
):
    """
    WebSocket endpoint for real-time chat with streaming responses.
    
    This endpoint:
    1. Accepts WebSocket connection
    2. Authenticates the user
    3. Streams responses token-by-token for real-time experience
    4. Handles disconnection gracefully
    
    Args:
        websocket: The WebSocket connection
        orchestrator: The agent orchestrator instance
    """
    
    # Accept the WebSocket connection
    await websocket.accept()
    
    # Generate connection ID for tracking
    connection_id = str(uuid.uuid4())
    logger.info(f"WebSocket connection {connection_id} established")
    
    try:
        # Authenticate the connection using token from query params
        token = websocket.query_params.get("token")
        if not token:
            await websocket.close(code=1008, reason="Missing authentication token")
            logger.warning(f"WebSocket {connection_id} closed: missing token")
            return
        
        # Validate user (simplified - use proper auth in production)
        # In production, you'd validate the JWT token here
        user_id = "authenticated_user"  # Extract from validated token
        
        # Get conversation ID if provided
        conversation_id = websocket.query_params.get("conversation_id")
        
        # Send connection acknowledgment
        await websocket.send_json({
            "type": "connection_established",
            "connection_id": connection_id,
            "conversation_id": conversation_id,
            "timestamp": datetime.now().isoformat()
        })
        
        # Main message loop
        while True:
            try:
                # Receive message from client with timeout
                data = await asyncio.wait_for(websocket.receive_text(), timeout=60.0)
                message = json.loads(data)
                
                # Validate message structure
                if "query" not in message:
                    await websocket.send_json({
                        "type": "error",
                        "error": "Missing 'query' field",
                        "timestamp": datetime.now().isoformat()
                    })
                    continue
                
                # Send acknowledgment
                await websocket.send_json({
                    "type": "ack",
                    "message_id": message.get("message_id", "unknown"),
                    "timestamp": datetime.now().isoformat()
                })
                
                # Prepare message for orchestrator
                orchestrator_message = {
                    "query": message["query"],
                    "user_id": user_id,
                    "context": message.get("context", {}),
                    "stream": True  # Indicate streaming mode
                }
                
                # Process through orchestrator (streaming version)
                result = await orchestrator.process_message_streaming(
                    message=orchestrator_message,
                    conversation_id=conversation_id
                )
                
                # Stream response token by token
                response_text = result.get("response", {}).get("text", "")
                words = response_text.split()
                
                for i, word in enumerate(words):
                    # Determine if this is the last token
                    is_last = (i == len(words) - 1)
                    
                    # Send token chunk
                    await websocket.send_json({
                        "type": "token",
                        "token": word + (" " if i < len(words) - 1 else ""),
                        "index": i,
                        "is_last": is_last,
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    # Small delay for natural streaming effect
                    await asyncio.sleep(0.03)
                
                # Send completion message with metadata
                await websocket.send_json({
                    "type": "complete",
                    "conversation_id": result.get("conversation_id"),
                    "agent": result.get("agent"),
                    "confidence": result.get("confidence"),
                    "requires_human": result.get("requires_human", False),
                    "suggestions": result.get("response", {}).get("suggestions", []),
                    "sources": result.get("response", {}).get("sources", []),
                    "processing_time_ms": result.get("processing_time_ms", 0),
                    "timestamp": datetime.now().isoformat()
                })
                
                # Update conversation ID for next message
                if result.get("conversation_id"):
                    conversation_id = result["conversation_id"]
                
            except asyncio.TimeoutError:
                # Send heartbeat to keep connection alive
                await websocket.send_json({
                    "type": "heartbeat",
                    "timestamp": datetime.now().isoformat()
                })
                
            except json.JSONDecodeError as je:
                # Handle malformed JSON
                logger.warning(f"Invalid JSON received: {je}")
                await websocket.send_json({
                    "type": "error",
                    "error": "Invalid JSON format",
                    "timestamp": datetime.now().isoformat()
                })
                
    except WebSocketDisconnect:
        # Handle client disconnection
        logger.info(f"WebSocket {connection_id} disconnected by client")
        
    except Exception as e:
        # Handle unexpected errors
        logger.error(f"WebSocket {connection_id} error: {str(e)}", exc_info=True)
        
        try:
            await websocket.send_json({
                "type": "error",
                "error": "Internal server error",
                "timestamp": datetime.now().isoformat()
            })
        except:
            # Connection might be closed already
            pass

@router.post("/feedback", response_model=ResponseModel)
async def submit_feedback(
    feedback: ChatFeedback,
    current_user: Dict[str, Any] = Depends(get_current_user),
    audit_service: AuditService = Depends()
):
    """
    Submit user feedback on chat responses for continuous improvement.
    
    This endpoint:
    1. Validates feedback data
    2. Stores feedback for model retraining
    3. Triggers alerts for negative feedback
    
    Args:
        feedback: User feedback data
        current_user: The authenticated user
        audit_service: Service for audit logging
    
    Returns:
        Confirmation of feedback receipt
    """
    
    feedback_id = str(uuid.uuid4())
    logger.info(f"Received feedback {feedback_id} from user {current_user['id']}")
    
    try:
        # Store feedback in database (implementation in audit_service)
        await audit_service.store_feedback(
            feedback_id=feedback_id,
            user_id=current_user["id"],
            conversation_id=feedback.conversation_id,
            message_id=feedback.message_id,
            rating=feedback.rating,
            feedback_text=feedback.feedback_text,
            tags=feedback.tags,
            timestamp=datetime.now()
        )
        
        # If rating is low (1-2), trigger notification for review
        if feedback.rating <= 2:
            notification_service = NotificationService()
            await notification_service.send_alert(
                type="negative_feedback",
                severity="medium",
                details={
                    "feedback_id": feedback_id,
                    "user_id": current_user["id"],
                    "conversation_id": feedback.conversation_id,
                    "rating": feedback.rating
                }
            )
        
        return ResponseModel(
            status="success",
            message="Feedback recorded successfully",
            data={"feedback_id": feedback_id}
        )
        
    except Exception as e:
        logger.error(f"Error storing feedback: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error storing feedback")

@router.get("/history/{conversation_id}")
async def get_conversation_history(
    conversation_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    audit_service: AuditService = Depends()
):
    """
    Retrieve conversation history for a specific conversation.
    
    Args:
        conversation_id: ID of the conversation to retrieve
        current_user: The authenticated user
        audit_service: Service for audit logging
    
    Returns:
        Conversation history with messages
    """
    
    logger.info(f"Retrieving conversation {conversation_id} for user {current_user['id']}")
    
    try:
        # Check if user has access to this conversation
        has_access = await audit_service.check_conversation_access(
            user_id=current_user["id"],
            conversation_id=conversation_id
        )
        
        if not has_access:
            raise HTTPException(status_code=403, detail="Access denied to this conversation")
        
        # Retrieve conversation history
        history = await audit_service.get_conversation_history(
            conversation_id=conversation_id
        )
        
        return ResponseModel(
            status="success",
            message="Conversation history retrieved",
            data=history
        )
        
    except HTTPException:
        raise
        
    except Exception as e:
        logger.error(f"Error retrieving conversation: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving conversation")