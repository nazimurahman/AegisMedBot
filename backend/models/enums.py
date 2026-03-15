"""
Enums module for AegisMedBot.

This module defines all enumeration types used throughout the platform,
providing type safety and consistent value sets for various attributes.
"""

from enum import Enum, unique
from typing import List, Tuple


@unique
class AgentType(str, Enum):
    """
    Enumeration of agent types in the system.
    
    Each agent type represents a specialized category of agent
    with distinct responsibilities and capabilities.
    """
    
    CLINICAL = "clinical"
    """Agent specializing in clinical knowledge and medical information"""
    
    RISK = "risk"
    """Agent specializing in patient risk assessment and prediction"""
    
    OPERATIONS = "operations"
    """Agent specializing in hospital operations and resource management"""
    
    DIRECTOR = "director"
    """Agent specializing in executive insights and strategic reporting"""
    
    COMPLIANCE = "compliance"
    """Agent specializing in regulatory compliance and privacy"""
    
    RESEARCH = "research"
    """Agent specializing in medical literature and research"""
    
    ORCHESTRATOR = "orchestrator"
    """Central agent coordinating other agents"""
    
    @classmethod
    def get_clinical_agents(cls) -> List[str]:
        """
        Get list of agent types focused on clinical tasks.
        
        Returns:
            List of clinical agent type values
        """
        return [cls.CLINICAL.value, cls.RISK.value, cls.RESEARCH.value]
    
    @classmethod
    def get_administrative_agents(cls) -> List[str]:
        """
        Get list of agent types focused on administrative tasks.
        
        Returns:
            List of administrative agent type values
        """
        return [cls.OPERATIONS.value, cls.DIRECTOR.value, cls.COMPLIANCE.value]


@unique
class AgentStatus(str, Enum):
    """
    Enumeration of possible agent statuses.
    
    Tracks the current operational state of each agent in the system.
    """
    
    IDLE = "idle"
    """Agent is online but not currently processing any tasks"""
    
    PROCESSING = "processing"
    """Agent is actively processing one or more tasks"""
    
    WAITING_FOR_HUMAN = "waiting_for_human"
    """Agent is waiting for human input or approval"""
    
    ERROR = "error"
    """Agent encountered an error and needs attention"""
    
    OFFLINE = "offline"
    """Agent is not currently available"""
    
    INITIALIZING = "initializing"
    """Agent is starting up and not yet ready"""
    
    SHUTTING_DOWN = "shutting_down"
    """Agent is in the process of shutting down"""
    
    DEGRADED = "degraded"
    """Agent is operational but with reduced functionality"""
    
    @classmethod
    def get_operational_statuses(cls) -> List[str]:
        """
        Get list of statuses indicating the agent is operational.
        
        Returns:
            List of operational status values
        """
        return [cls.IDLE.value, cls.PROCESSING.value, cls.WAITING_FOR_HUMAN.value]
    
    @classmethod
    def get_unavailable_statuses(cls) -> List[str]:
        """
        Get list of statuses indicating the agent is not available.
        
        Returns:
            List of unavailable status values
        """
        return [cls.OFFLINE.value, cls.ERROR.value, cls.SHUTTING_DOWN.value]


@unique
class MessageType(str, Enum):
    """
    Enumeration of message types for agent communication.
    
    Defines the different kinds of messages that can be exchanged
    between agents and between agents and the orchestrator.
    """
    
    REQUEST = "request"
    """Initial request for information or action"""
    
    RESPONSE = "response"
    """Response to a previous request"""
    
    ESCALATION = "escalation"
    """Message requiring escalation to human"""
    
    ERROR = "error"
    """Error notification"""
    
    STATUS_UPDATE = "status_update"
    """Agent status update"""
    
    HEARTBEAT = "heartbeat"
    """Agent heartbeat signal"""
    
    CONFIG_UPDATE = "config_update"
    """Configuration update notification"""
    
    TASK_ASSIGNMENT = "task_assignment"
    """Assignment of a new task"""
    
    TASK_COMPLETE = "task_complete"
    """Notification of task completion"""
    
    QUERY = "query"
    """Query for information"""
    
    @classmethod
    def get_control_messages(cls) -> List[str]:
        """
        Get list of control-related message types.
        
        Returns:
            List of control message type values
        """
        return [cls.HEARTBEAT.value, cls.STATUS_UPDATE.value, cls.CONFIG_UPDATE.value]
    
    @classmethod
    def get_task_messages(cls) -> List[str]:
        """
        Get list of task-related message types.
        
        Returns:
            List of task message type values
        """
        return [cls.TASK_ASSIGNMENT.value, cls.TASK_COMPLETE.value]


@unique
class ConfidenceLevel(str, Enum):
    """
    Enumeration of confidence levels for agent responses.
    
    Provides human-readable confidence categories based on the
    numerical confidence score.
    """
    
    VERY_HIGH = "very_high"
    """Confidence score 0.9-1.0 - Highly reliable response"""
    
    HIGH = "high"
    """Confidence score 0.7-0.9 - Generally reliable"""
    
    MEDIUM = "medium"
    """Confidence score 0.5-0.7 - Moderately reliable"""
    
    LOW = "low"
    """Confidence score 0.3-0.5 - Low reliability, verify information"""
    
    VERY_LOW = "very_low"
    """Confidence score 0.0-0.3 - Very low reliability, human review recommended"""
    
    @classmethod
    def from_score(cls, score: float) -> 'ConfidenceLevel':
        """
        Convert numerical confidence score to enum value.
        
        Args:
            score: Numerical confidence score between 0 and 1
            
        Returns:
            Appropriate ConfidenceLevel enum value
        """
        if score >= 0.9:
            return cls.VERY_HIGH
        elif score >= 0.7:
            return cls.HIGH
        elif score >= 0.5:
            return cls.MEDIUM
        elif score >= 0.3:
            return cls.LOW
        else:
            return cls.VERY_LOW


@unique
class UserRole(str, Enum):
    """
    Enumeration of user roles in the system.
    
    Defines the different user types for role-based access control.
    """
    
    MEDICAL_DIRECTOR = "medical_director"
    """Medical Director - Full access to all features"""
    
    PHYSICIAN = "physician"
    """Physician - Clinical access"""
    
    NURSE = "nurse"
    """Nurse - Patient care access"""
    
    ADMINISTRATOR = "administrator"
    """System Administrator - Technical access"""
    
    RESEARCHER = "researcher"
    """Researcher - Research data access"""
    
    COMPLIANCE_OFFICER = "compliance_officer"
    """Compliance Officer - Audit and compliance access"""
    
    @classmethod
    def get_clinical_roles(cls) -> List[str]:
        """
        Get list of clinical user roles.
        
        Returns:
            List of clinical role values
        """
        return [cls.MEDICAL_DIRECTOR.value, cls.PHYSICIAN.value, cls.NURSE.value]
    
    @classmethod
    def get_administrative_roles(cls) -> List[str]:
        """
        Get list of administrative user roles.
        
        Returns:
            List of administrative role values
        """
        return [cls.ADMINISTRATOR.value, cls.COMPLIANCE_OFFICER.value]


@unique
class Permission(str, Enum):
    """
    Enumeration of system permissions.
    
    Defines granular permissions for role-based access control.
    """
    
    # Patient data permissions
    VIEW_PATIENT = "view_patient"
    EDIT_PATIENT = "edit_patient"
    DELETE_PATIENT = "delete_patient"
    
    # Clinical permissions
    VIEW_CLINICAL_DATA = "view_clinical_data"
    EDIT_CLINICAL_DATA = "edit_clinical_data"
    ACCESS_SENSITIVE_DATA = "access_sensitive_data"
    
    # Agent permissions
    VIEW_AGENTS = "view_agents"
    CONFIGURE_AGENTS = "configure_agents"
    DEPLOY_AGENTS = "deploy_agents"
    
    # System permissions
    VIEW_LOGS = "view_logs"
    CONFIGURE_SYSTEM = "configure_system"
    MANAGE_USERS = "manage_users"
    
    # Audit permissions
    VIEW_AUDIT = "view_audit"
    EXPORT_AUDIT = "export_audit"
    
    @classmethod
    def get_patient_permissions(cls) -> List[str]:
        """
        Get list of patient-related permissions.
        
        Returns:
            List of patient permission values
        """
        return [cls.VIEW_PATIENT.value, cls.EDIT_PATIENT.value, cls.DELETE_PATIENT.value]


@unique
class AuditAction(str, Enum):
    """
    Enumeration of auditable actions.
    
    Defines the different actions that can be recorded in audit logs.
    """
    
    # Authentication actions
    LOGIN = "login"
    LOGOUT = "logout"
    LOGIN_FAILED = "login_failed"
    
    # Data actions
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    EXPORT = "export"
    
    # Agent actions
    AGENT_REGISTER = "agent_register"
    AGENT_DEREGISTER = "agent_deregister"
    AGENT_TASK = "agent_task"
    AGENT_RESPONSE = "agent_response"
    
    # System actions
    CONFIG_CHANGE = "config_change"
    SYSTEM_RESTART = "system_restart"
    BACKUP = "backup"
    
    # Chat actions
    CHAT_QUERY = "chat_query"
    CHAT_RESPONSE = "chat_response"
    CHAT_FEEDBACK = "chat_feedback"
    
    @classmethod
    def get_data_actions(cls) -> List[str]:
        """
        Get list of data-related actions.
        
        Returns:
            List of data action values
        """
        return [cls.CREATE.value, cls.READ.value, cls.UPDATE.value, cls.DELETE.value]


@unique
class TaskPriority(int, Enum):
    """
    Enumeration of task priority levels.
    
    Higher number indicates higher priority.
    """
    
    CRITICAL = 10
    """Critical priority - immediate attention required"""
    
    HIGH = 7
    """High priority - urgent"""
    
    MEDIUM = 5
    """Medium priority - normal"""
    
    LOW = 3
    """Low priority - can wait"""
    
    BACKGROUND = 1
    """Background priority - process when idle"""


@unique
class ResourceType(str, Enum):
    """
    Enumeration of resource types in the system.
    
    Identifies the type of resource for permission and audit purposes.
    """
    
    PATIENT = "patient"
    ADMISSION = "admission"
    VITAL_SIGN = "vital_sign"
    LAB_RESULT = "lab_result"
    MEDICATION = "medication"
    
    AGENT = "agent"
    TASK = "task"
    CONVERSATION = "conversation"
    
    USER = "user"
    ROLE = "role"
    PERMISSION = "permission"
    
    CONFIG = "config"
    AUDIT_LOG = "audit_log"


@unique
class Environment(str, Enum):
    """
    Enumeration of deployment environments.
    
    Identifies the current runtime environment for configuration.
    """
    
    DEVELOPMENT = "development"
    """Local development environment"""
    
    STAGING = "staging"
    """Staging/testing environment"""
    
    PRODUCTION = "production"
    """Production environment"""
    
    TEST = "test"
    """Automated test environment"""


@unique
class Severity(str, Enum):
    """
    Enumeration of severity levels for logs and alerts.
    """
    
    DEBUG = "debug"
    """Debug information - detailed diagnostic data"""
    
    INFO = "info"
    """Normal operational information"""
    
    WARNING = "warning"
    """Warning - potential issue detected"""
    
    ERROR = "error"
    """Error - operation failed but system continues"""
    
    CRITICAL = "critical"
    """Critical - system may be unable to continue"""