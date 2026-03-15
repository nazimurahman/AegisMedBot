"""
Logging middleware for AegisMedBot.
Provides structured logging for all requests and responses.
Integrates with ELK stack for centralized log management.
"""

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Callable, Dict, Any
import time
import logging
import uuid
import json
from datetime import datetime
from ..core.config import settings

# Configure structured logging
logger = logging.getLogger(__name__)

class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for logging all HTTP requests and responses.
    Creates structured logs with request/response details, timing, and errors.
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process each request through the logging middleware.
        
        Args:
            request: The incoming FastAPI request
            call_next: The next middleware or route handler
            
        Returns:
            The response from the next handler
        """
        # Generate a unique request ID for tracing
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Add request ID to response headers for client-side tracing
        request.state.request_id = request_id
        
        # Record start time for performance monitoring
        start_time = time.time()
        
        # Extract request metadata for logging
        request_metadata = await self._extract_request_metadata(request)
        
        # Log the incoming request
        logger.info(
            "Incoming request",
            extra={
                "request_id": request_id,
                "type": "request",
                "metadata": request_metadata
            }
        )
        
        try:
            # Process the request through the application
            response = await call_next(request)
            
            # Calculate processing time
            process_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            # Extract response metadata
            response_metadata = self._extract_response_metadata(response, process_time)
            
            # Log successful response
            logger.info(
                "Request completed",
                extra={
                    "request_id": request_id,
                    "type": "response",
                    "metadata": {
                        **request_metadata,
                        **response_metadata
                    }
                }
            )
            
            # Add processing time header
            response.headers["X-Process-Time-MS"] = str(int(process_time))
            response.headers["X-Request-ID"] = request_id
            
            return response
            
        except Exception as e:
            # Log any unhandled exceptions
            process_time = (time.time() - start_time) * 1000
            logger.error(
                "Request failed",
                extra={
                    "request_id": request_id,
                    "type": "error",
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "metadata": {
                        **request_metadata,
                        "process_time_ms": process_time
                    }
                },
                exc_info=True
            )
            # Re-raise the exception to be handled by FastAPI's exception handlers
            raise
    
    async def _extract_request_metadata(self, request: Request) -> Dict[str, Any]:
        """
        Extract relevant metadata from the request for logging.
        
        Args:
            request: FastAPI request object
            
        Returns:
            Dictionary with request metadata
        """
        metadata = {
            "method": request.method,
            "path": request.url.path,
            "query_string": str(request.query_params),
            "client_host": request.client.host if request.client else None,
            "user_agent": request.headers.get("user-agent"),
            "content_type": request.headers.get("content-type"),
            "content_length": request.headers.get("content-length")
        }
        
        # Add user info if authenticated
        user = getattr(request.state, "user", None)
        if user:
            metadata["user_id"] = user.get("sub")
            metadata["user_role"] = user.get("role")
        
        # Add rate limit info if available
        rate_headers = getattr(request.state, "rate_limit_headers", {})
        if rate_headers:
            metadata["rate_limit"] = rate_headers
        
        return metadata
    
    def _extract_response_metadata(self, response: Response, process_time: float) -> Dict[str, Any]:
        """
        Extract metadata from the response for logging.
        
        Args:
            response: FastAPI response object
            process_time: Request processing time in milliseconds
            
        Returns:
            Dictionary with response metadata
        """
        return {
            "status_code": response.status_code,
            "process_time_ms": process_time,
            "response_size": response.headers.get("content-length"),
            "response_headers": dict(response.headers)
        }

class StructuredLogFormatter(logging.Formatter):
    """
    Custom log formatter that outputs logs in JSON format for ELK stack.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format a log record as JSON.
        
        Args:
            record: The log record to format
            
        Returns:
            JSON string representation of the log
        """
        # Base log data
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add extra fields if they exist
        if hasattr(record, "extra"):
            log_data.update(record.extra)
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_data)

class AuditLogger:
    """
    Specialized logger for audit trails.
    Logs all sensitive operations for compliance and security monitoring.
    """
    
    def __init__(self):
        self.logger = logging.getLogger("audit")
        self.logger.setLevel(logging.INFO)
    
    async def log_action(
        self,
        user_id: str,
        action: str,
        resource: str,
        details: Dict[str, Any],
        result: str = "success",
        ip_address: str = None
    ):
        """
        Log an auditable action.
        
        Args:
            user_id: ID of the user performing the action
            action: Type of action (create, read, update, delete)
            resource: Resource being accessed
            details: Additional details about the action
            result: Result of the action (success/failure)
            ip_address: Client IP address
        """
        audit_data = {
            "type": "audit",
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "action": action,
            "resource": resource,
            "details": details,
            "result": result,
            "ip_address": ip_address
        }
        
        self.logger.info("Audit log", extra=audit_data)
    
    async def log_data_access(
        self,
        user_id: str,
        patient_id: str,
        data_type: str,
        access_reason: str,
        ip_address: str = None
    ):
        """
        Log access to patient data for HIPAA compliance.
        
        Args:
            user_id: ID of the user accessing data
            patient_id: ID of the patient whose data is accessed
            data_type: Type of data accessed (PHI, clinical notes, etc.)
            access_reason: Reason for accessing the data
            ip_address: Client IP address
        """
        await self.log_action(
            user_id=user_id,
            action="read",
            resource=f"patient_data/{patient_id}",
            details={
                "patient_id": patient_id,
                "data_type": data_type,
                "access_reason": access_reason
            },
            ip_address=ip_address
        )