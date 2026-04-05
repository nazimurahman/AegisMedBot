"""
Audit Repository Module for AegisMedBot

This module handles audit logging for all system operations to ensure
compliance with healthcare regulations and provide a complete
trace of all data access and modifications.
"""

from sqlalchemy.orm import Session
from sqlalchemy import and_, desc, func
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import json
import logging

from database.models.audit import AuditLog  # We'll define this model

logger = logging.getLogger(__name__)

class AuditRepository:
    """
    Repository for audit log operations
    
    This class manages all audit trail operations including recording
    data access, modifications, and user actions for compliance.
    
    Attributes:
        session: SQLAlchemy database session
    """
    
    def __init__(self, session: Session):
        """
        Initialize audit repository with database session
        
        Args:
            session: SQLAlchemy Session object for database operations
        """
        self.session = session
    
    def log_action(
        self,
        user_id: str,
        action: str,
        resource_type: str,
        resource_id: str,
        details: Dict[str, Any],
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> AuditLog:
        """
        Log an action to the audit trail
        
        This method records any user action or system event for
        compliance and security auditing.
        
        Args:
            user_id: ID of user performing action
            action: Type of action (CREATE, READ, UPDATE, DELETE, LOGIN, etc.)
            resource_type: Type of resource being accessed (Patient, Admission, etc.)
            resource_id: ID of the specific resource
            details: Additional details about the action
            ip_address: Source IP address of request
            user_agent: User agent string from request
            
        Returns:
            Created AuditLog object
        """
        audit_entry = AuditLog(
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details,
            ip_address=ip_address,
            user_agent=user_agent,
            timestamp=datetime.utcnow()
        )
        
        self.session.add(audit_entry)
        self.session.commit()
        self.session.refresh(audit_entry)
        
        logger.info(f"Audit log created: {action} on {resource_type} {resource_id} by {user_id}")
        return audit_entry
    
    def get_user_audit_trail(
        self,
        user_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[AuditLog]:
        """
        Get audit trail for a specific user
        
        Args:
            user_id: ID of user to audit
            start_date: Start of time range (optional)
            end_date: End of time range (optional)
            limit: Maximum number of records to return
            
        Returns:
            List of AuditLog objects for the user
        """
        query = self.session.query(AuditLog).filter(AuditLog.user_id == user_id)
        
        if start_date:
            query = query.filter(AuditLog.timestamp >= start_date)
        if end_date:
            query = query.filter(AuditLog.timestamp <= end_date)
        
        logs = query.order_by(desc(AuditLog.timestamp)).limit(limit).all()
        return logs
    
    def get_resource_audit_trail(
        self,
        resource_type: str,
        resource_id: str,
        limit: int = 100
    ) -> List[AuditLog]:
        """
        Get audit trail for a specific resource
        
        This method shows all access and modifications to a given resource.
        
        Args:
            resource_type: Type of resource (Patient, Admission, etc.)
            resource_id: ID of the specific resource
            limit: Maximum number of records to return
            
        Returns:
            List of AuditLog objects for the resource
        """
        logs = self.session.query(AuditLog).filter(
            and_(
                AuditLog.resource_type == resource_type,
                AuditLog.resource_id == resource_id
            )
        ).order_by(desc(AuditLog.timestamp)).limit(limit).all()
        
        return logs
    
    def get_recent_actions(
        self,
        action_type: Optional[str] = None,
        minutes: int = 60,
        limit: int = 100
    ) -> List[AuditLog]:
        """
        Get recent actions within time window
        
        Args:
            action_type: Filter by specific action type (optional)
            minutes: Number of minutes to look back
            limit: Maximum number of records to return
            
        Returns:
            List of recent AuditLog objects
        """
        cutoff_time = datetime.utcnow() - timedelta(minutes=minutes)
        
        query = self.session.query(AuditLog).filter(AuditLog.timestamp >= cutoff_time)
        
        if action_type:
            query = query.filter(AuditLog.action == action_type)
        
        logs = query.order_by(desc(AuditLog.timestamp)).limit(limit).all()
        return logs
    
    def get_action_statistics(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """
        Get statistics about actions over a time period
        
        This method provides aggregated metrics for security monitoring
        and compliance reporting.
        
        Args:
            start_date: Start of analysis period
            end_date: End of analysis period
            
        Returns:
            Dictionary with action statistics
        """
        query = self.session.query(AuditLog).filter(
            and_(
                AuditLog.timestamp >= start_date,
                AuditLog.timestamp <= end_date
            )
        )
        
        total_actions = query.count()
        
        # Count by action type
        action_counts = {}
        actions = query.all()
        
        for action in actions:
            action_counts[action.action] = action_counts.get(action.action, 0) + 1
        
        # Count by resource type
        resource_counts = {}
        for action in actions:
            resource_counts[action.resource_type] = resource_counts.get(action.resource_type, 0) + 1
        
        # Unique users
        unique_users = len(set([a.user_id for a in actions]))
        
        return {
            "total_actions": total_actions,
            "unique_users": unique_users,
            "action_counts": action_counts,
            "resource_counts": resource_counts,
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            }
        }
    
    def check_compliance(
        self,
        user_id: str,
        resource_type: str,
        resource_id: str
    ) -> bool:
        """
        Check if user has appropriate audit trail for compliance
        
        This method verifies that all required audit records exist
        for compliance with regulations like HIPAA.
        
        Args:
            user_id: ID of user to check
            resource_type: Type of resource accessed
            resource_id: ID of resource accessed
            
        Returns:
            True if compliance requirements are met, False otherwise
        """
        # Check that access was logged
        access_log = self.session.query(AuditLog).filter(
            and_(
                AuditLog.user_id == user_id,
                AuditLog.resource_type == resource_type,
                AuditLog.resource_id == resource_id,
                AuditLog.action.in_(["READ", "UPDATE", "DELETE"])
            )
        ).first()
        
        return access_log is not None
    
    def purge_old_logs(self, retention_days: int = 2555) -> int:
        """
        Delete audit logs older than retention period
        
        Healthcare regulations typically require 7 years of retention.
        This method implements secure deletion of old logs.
        
        Args:
            retention_days: Number of days to retain logs (default 7 years)
            
        Returns:
            Number of logs deleted
        """
        cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
        
        old_logs = self.session.query(AuditLog).filter(
            AuditLog.timestamp < cutoff_date
        ).all()
        
        count = len(old_logs)
        
        for log in old_logs:
            self.session.delete(log)
        
        self.session.commit()
        
        logger.info(f"Purged {count} audit logs older than {retention_days} days")
        return count