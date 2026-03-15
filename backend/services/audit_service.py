"""
Audit logging service for AegisMedBot.

This module provides comprehensive audit logging capabilities for
tracking all system events, user actions, and data access for
compliance and security monitoring.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
import json
import logging
from uuid import uuid4
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import insert, select

from ..models.enums import AuditAction, Severity, ResourceType
from ..core.config import settings

# Configure module logger
logger = logging.getLogger(__name__)


class AuditService:
    """
    Service for comprehensive audit logging across the platform.
    
    This service handles:
    - Recording all user actions and system events
    - Structured logging for easy searching and analysis
    - Integration with multiple storage backends
    - Compliance reporting
    """
    
    def __init__(
        self,
        redis_client: Optional[Redis] = None,
        db_session: Optional[AsyncSession] = None
    ):
        """
        Initialize the audit service.
        
        Args:
            redis_client: Optional Redis client for caching
            db_session: Optional database session for persistent storage
        """
        self.redis_client = redis_client
        self.db_session = db_session
        self.logger = logging.getLogger("audit")
        
        # Ensure audit logger has a file handler for persistent storage
        self._setup_audit_logger()
        
        logger.info("AuditService initialized")
    
    def _setup_audit_logger(self):
        """Configure the audit logger with appropriate handlers."""
        # Create a file handler for audit logs if not already configured
        if not self.logger.handlers:
            try:
                file_handler = logging.FileHandler("logs/audit.log")
                formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)
                self.logger.setLevel(logging.INFO)
            except Exception as e:
                logger.error(f"Failed to setup audit logger: {str(e)}")
    
    async def log_event(
        self,
        action: AuditAction,
        user_id: str,
        resource_type: Optional[ResourceType] = None,
        resource_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        outcome: str = "success",
        error_code: Optional[str] = None,
        severity: Severity = Severity.INFO
    ) -> str:
        """
        Log an audit event.
        
        This is the main method for recording all audit events.
        
        Args:
            action: Type of action being logged
            user_id: ID of the user performing the action
            resource_type: Type of resource being accessed
            resource_id: ID of the specific resource
            details: Additional event details
            ip_address: Client IP address
            user_agent: Client user agent
            outcome: Outcome of the action (success/failure)
            error_code: Error code if action failed
            severity: Severity level of the event
            
        Returns:
            ID of the created audit log entry
        """
        log_id = str(uuid4())
        timestamp = datetime.utcnow()
        
        # Create structured log entry
        log_entry = {
            "log_id": log_id,
            "timestamp": timestamp.isoformat(),
            "action": action.value if isinstance(action, AuditAction) else action,
            "user_id": user_id,
            "resource_type": resource_type.value if resource_type else None,
            "resource_id": resource_id,
            "details": json.dumps(details) if details else None,
            "ip_address": ip_address,
            "user_agent": user_agent,
            "outcome": outcome,
            "error_code": error_code,
            "severity": severity.value if isinstance(severity, Severity) else severity,
            "environment": settings.ENVIRONMENT
        }
        
        # Log to file
        self.logger.info(json.dumps(log_entry))
        
        # Store in Redis for recent access
        if self.redis_client:
            await self._store_in_redis(log_id, log_entry)
        
        # Store in database for permanent retention
        if self.db_session:
            await self._store_in_database(log_entry)
        
        return log_id
    
    async def _store_in_redis(self, log_id: str, log_entry: Dict[str, Any]):
        """
        Store audit log in Redis for fast recent access.
        
        Args:
            log_id: Unique log identifier
            log_entry: Log entry data
        """
        try:
            # Store with 7-day expiration
            await self.redis_client.setex(
                f"audit:{log_id}",
                604800,  # 7 days in seconds
                json.dumps(log_entry)
            )
            
            # Add to recent logs list
            await self.redis_client.lpush(
                "audit:recent",
                log_id
            )
            await self.redis_client.ltrim("audit:recent", 0, 999)  # Keep last 1000
            
        except Exception as e:
            logger.error(f"Failed to store audit log in Redis: {str(e)}")
    
    async def _store_in_database(self, log_entry: Dict[str, Any]):
        """
        Store audit log in database for permanent retention.
        
        Args:
            log_entry: Log entry data
        """
        try:
            # Insert into audit_logs table
            stmt = insert(AuditLog).values(**log_entry)
            await self.db_session.execute(stmt)
            await self.db_session.commit()
        except Exception as e:
            logger.error(f"Failed to store audit log in database: {str(e)}")
            await self.db_session.rollback()
    
    async def log_chat_request(
        self,
        user_id: str,
        query: str,
        conversation_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> str:
        """
        Log a chat request from a user.
        
        Args:
            user_id: ID of the user making the request
            query: User's query text
            conversation_id: ID of the conversation
            ip_address: Client IP address
            user_agent: Client user agent
            
        Returns:
            ID of the created audit log entry
        """
        details = {
            "query_preview": query[:200] + "..." if len(query) > 200 else query,
            "query_length": len(query),
            "conversation_id": conversation_id
        }
        
        return await self.log_event(
            action=AuditAction.CHAT_QUERY,
            user_id=user_id,
            resource_type=ResourceType.CONVERSATION,
            resource_id=conversation_id,
            details=details,
            ip_address=ip_address,
            user_agent=user_agent,
            outcome="success",
            severity=Severity.INFO
        )
    
    async def log_chat_response(
        self,
        user_id: str,
        conversation_id: str,
        response: Dict[str, Any],
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> str:
        """
        Log a chat response sent to a user.
        
        Args:
            user_id: ID of the user receiving the response
            conversation_id: ID of the conversation
            response: Response data
            ip_address: Client IP address
            user_agent: Client user agent
            
        Returns:
            ID of the created audit log entry
        """
        details = {
            "conversation_id": conversation_id,
            "agent": response.get("agent"),
            "confidence": response.get("confidence"),
            "requires_human": response.get("requires_human", False),
            "processing_time_ms": response.get("processing_time_ms")
        }
        
        return await self.log_event(
            action=AuditAction.CHAT_RESPONSE,
            user_id=user_id,
            resource_type=ResourceType.CONVERSATION,
            resource_id=conversation_id,
            details=details,
            ip_address=ip_address,
            user_agent=user_agent,
            outcome="success",
            severity=Severity.INFO
        )
    
    async def log_data_access(
        self,
        user_id: str,
        resource_type: ResourceType,
        resource_id: str,
        action: AuditAction,
        success: bool = True,
        error: Optional[str] = None,
        ip_address: Optional[str] = None
    ) -> str:
        """
        Log access to sensitive data.
        
        Args:
            user_id: ID of the user accessing data
            resource_type: Type of resource being accessed
            resource_id: ID of the specific resource
            action: Action being performed
            success: Whether access was successful
            error: Error message if access failed
            ip_address: Client IP address
            
        Returns:
            ID of the created audit log entry
        """
        details = {
            "resource_type": resource_type.value,
            "resource_id": resource_id
        }
        
        if error:
            details["error"] = error
        
        return await self.log_event(
            action=action,
            user_id=user_id,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details,
            ip_address=ip_address,
            outcome="failure" if error else "success",
            error_code=error,
            severity=Severity.WARNING if error else Severity.INFO
        )
    
    async def log_agent_action(
        self,
        agent_name: str,
        action: str,
        task_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        success: bool = True,
        error: Optional[str] = None
    ) -> str:
        """
        Log an agent action.
        
        Args:
            agent_name: Name of the agent
            action: Action being performed
            task_id: ID of the task
            details: Additional details
            success: Whether action was successful
            error: Error message if action failed
            
        Returns:
            ID of the created audit log entry
        """
        log_details = {
            "agent_name": agent_name,
            "action": action,
            "task_id": task_id
        }
        
        if details:
            log_details.update(details)
        
        return await self.log_event(
            action=AuditAction.AGENT_TASK,
            user_id=f"agent:{agent_name}",  # Agents have synthetic user IDs
            resource_type=ResourceType.AGENT,
            resource_id=agent_name,
            details=log_details,
            outcome="failure" if error else "success",
            error_code=error,
            severity=Severity.WARNING if error else Severity.INFO
        )
    
    async def log_authentication(
        self,
        user_id: str,
        success: bool,
        failure_reason: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> str:
        """
        Log authentication attempts.
        
        Args:
            user_id: ID of the user attempting authentication
            success: Whether authentication was successful
            failure_reason: Reason for failure if unsuccessful
            ip_address: Client IP address
            user_agent: Client user agent
            
        Returns:
            ID of the created audit log entry
        """
        action = AuditAction.LOGIN if success else AuditAction.LOGIN_FAILED
        
        details = {}
        if failure_reason:
            details["failure_reason"] = failure_reason
        
        return await self.log_event(
            action=action,
            user_id=user_id,
            details=details,
            ip_address=ip_address,
            user_agent=user_agent,
            outcome="success" if success else "failure",
            error_code=failure_reason,
            severity=Severity.WARNING if not success else Severity.INFO
        )
    
    async def get_recent_logs(
        self,
        limit: int = 100,
        user_id: Optional[str] = None,
        action: Optional[AuditAction] = None
    ) -> List[Dict[str, Any]]:
        """
        Get recent audit logs.
        
        Args:
            limit: Maximum number of logs to return
            user_id: Filter by user ID
            action: Filter by action type
            
        Returns:
            List of recent audit log entries
        """
        logs = []
        
        # Try to get from Redis first
        if self.redis_client:
            recent_ids = await self.redis_client.lrange("audit:recent", 0, limit - 1)
            
            for log_id in recent_ids:
                log_data = await self.redis_client.get(f"audit:{log_id}")
                if log_data:
                    log_entry = json.loads(log_data)
                    
                    # Apply filters
                    if user_id and log_entry.get("user_id") != user_id:
                        continue
                    if action and log_entry.get("action") != action.value:
                        continue
                    
                    logs.append(log_entry)
        
        return logs
    
    async def search_logs(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        user_id: Optional[str] = None,
        action: Optional[AuditAction] = None,
        resource_type: Optional[ResourceType] = None,
        resource_id: Optional[str] = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Search audit logs with filters.
        
        Args:
            start_date: Filter logs after this date
            end_date: Filter logs before this date
            user_id: Filter by user ID
            action: Filter by action type
            resource_type: Filter by resource type
            resource_id: Filter by resource ID
            limit: Maximum number of logs to return
            
        Returns:
            List of matching audit log entries
        """
        # In production, this would query the database
        # For now, return empty list
        return []
    
    async def generate_compliance_report(
        self,
        start_date: datetime,
        end_date: datetime,
        report_type: str = "hipaa"
    ) -> Dict[str, Any]:
        """
        Generate a compliance report for a time period.
        
        Args:
            start_date: Start of report period
            end_date: End of report period
            report_type: Type of report (hipaa, gdpr, etc.)
            
        Returns:
            Compliance report data
        """
        # In production, this would analyze logs and generate comprehensive report
        report = {
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "report_type": report_type,
            "generated_at": datetime.utcnow().isoformat(),
            "summary": {
                "total_events": 0,
                "unique_users": 0,
                "data_access_events": 0,
                "failed_attempts": 0
            },
            "details": {}
        }
        
        return report


# SQLAlchemy model for audit logs (for reference)
"""
CREATE TABLE audit_logs (
    log_id VARCHAR(36) PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    action VARCHAR(50) NOT NULL,
    user_id VARCHAR(100) NOT NULL,
    resource_type VARCHAR(50),
    resource_id VARCHAR(100),
    details TEXT,
    ip_address VARCHAR(45),
    user_agent TEXT,
    outcome VARCHAR(20),
    error_code VARCHAR(100),
    severity VARCHAR(20),
    environment VARCHAR(20),
    INDEX idx_timestamp (timestamp),
    INDEX idx_user_id (user_id),
    INDEX_idx_action (action)
);
"""