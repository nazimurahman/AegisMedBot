"""
Admin routes for AegisMedBot platform.
Handles system administration, user management, and monitoring endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from fastapi.security import OAuth2PasswordBearer
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, validator
from datetime import datetime, timedelta
import logging
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc
import json

# Import core dependencies
from core.config import settings
from core.database import get_db
from core.security import get_current_admin_user, hash_password, verify_password
from models.schemas.patient import PatientResponse, PatientListResponse
from models.schemas.agent import AgentStatusResponse, AgentMetricsResponse
from services.audit_service import AuditService
from services.notification_service import NotificationService

# Configure logging for admin operations
logger = logging.getLogger(__name__)

# Create admin router with prefix and tags
router = APIRouter(
    prefix="/api/v1/admin",
    tags=["Admin Operations"],
    responses={
        401: {"description": "Unauthorized - Admin access required"},
        403: {"description": "Forbidden - Insufficient permissions"},
        404: {"description": "Resource not found"}
    }
)

# ============================================================================
# Pydantic Models for Request/Response Validation
# ============================================================================

class UserCreateRequest(BaseModel):
    """
    Model for creating a new user in the system.
    Validates user data before database insertion.
    """
    username: str = Field(..., min_length=3, max_length=50, 
                          description="Unique username for the user")
    email: str = Field(..., description="Valid email address")
    password: str = Field(..., min_length=8, max_length=100,
                          description="Password with minimum 8 characters")
    full_name: str = Field(..., min_length=2, max_length=100,
                          description="User's full name")
    role: str = Field(..., description="User role: admin, doctor, nurse, or staff")
    department: Optional[str] = Field(None, max_length=100,
                                      description="Department assignment")
    is_active: bool = Field(True, description="Whether user account is active")
    
    @validator('email')
    def validate_email(cls, v):
        """
        Custom validator to ensure email format is correct.
        Uses simple regex pattern for demonstration.
        """
        import re
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, v):
            raise ValueError('Invalid email format')
        return v.lower()  # Normalize email to lowercase
    
    @validator('role')
    def validate_role(cls, v):
        """
        Ensure role is one of the allowed values.
        """
        allowed_roles = ['admin', 'doctor', 'nurse', 'staff', 'researcher']
        if v.lower() not in allowed_roles:
            raise ValueError(f'Role must be one of: {", ".join(allowed_roles)}')
        return v.lower()

class UserUpdateRequest(BaseModel):
    """
    Model for updating existing user information.
    All fields are optional since partial updates are allowed.
    """
    email: Optional[str] = Field(None, description="Updated email address")
    full_name: Optional[str] = Field(None, min_length=2, max_length=100,
                                     description="Updated full name")
    department: Optional[str] = Field(None, max_length=100,
                                      description="Updated department")
    is_active: Optional[bool] = Field(None, description="Updated active status")
    role: Optional[str] = Field(None, description="Updated role")
    
    @validator('email')
    def validate_email(cls, v):
        """Validate email if provided."""
        if v is not None:
            import re
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            if not re.match(email_pattern, v):
                raise ValueError('Invalid email format')
            return v.lower()
        return v

class UserResponse(BaseModel):
    """
    Model for user data returned to admin clients.
    Excludes sensitive information like password hash.
    """
    id: str
    username: str
    email: str
    full_name: str
    role: str
    department: Optional[str]
    is_active: bool
    created_at: datetime
    last_login: Optional[datetime]
    created_by: Optional[str]
    
    class Config:
        """Pydantic configuration for ORM mode."""
        orm_mode = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class SystemHealthResponse(BaseModel):
    """
    Comprehensive system health status response.
    Includes all critical service statuses.
    """
    status: str = Field(..., description="Overall system status: healthy, degraded, or down")
    timestamp: datetime = Field(default_factory=datetime.now)
    
    # Individual service statuses
    database: Dict[str, Any] = Field(..., description="Database connection status")
    redis: Dict[str, Any] = Field(..., description="Redis cache status")
    qdrant: Dict[str, Any] = Field(..., description="Vector database status")
    agents: Dict[str, Any] = Field(..., description="AI agents status")
    api: Dict[str, Any] = Field(..., description="API service status")
    
    # Performance metrics
    response_time_ms: float = Field(..., description="API response time in milliseconds")
    active_users: int = Field(..., description="Number of currently active users")
    total_requests_24h: int = Field(..., description="Total API requests in last 24 hours")
    
    class Config:
        """Allow ORM mode and custom JSON encoding."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class AuditLogResponse(BaseModel):
    """
    Model for audit log entries returned to admin.
    Tracks all security-relevant events.
    """
    id: str
    user_id: Optional[str]
    username: Optional[str]
    action: str
    resource: str
    resource_id: Optional[str]
    details: Dict[str, Any]
    ip_address: str
    user_agent: Optional[str]
    timestamp: datetime
    status: str  # success, failure, pending
    
    class Config:
        """Allow ORM mode."""
        orm_mode = True

class SystemSettingsUpdate(BaseModel):
    """
    Model for updating system-wide settings.
    Only accessible by admin users.
    """
    maintenance_mode: Optional[bool] = Field(None, description="Enable/disable maintenance mode")
    max_conversations_per_user: Optional[int] = Field(None, gt=0, le=1000,
                                                      description="Maximum conversations per user")
    agent_timeout_seconds: Optional[int] = Field(None, gt=5, le=120,
                                                 description="Agent processing timeout")
    rate_limit_requests: Optional[int] = Field(None, gt=10, le=10000,
                                               description="Rate limit per minute")
    enable_audit_logging: Optional[bool] = Field(None, description="Enable detailed audit logs")
    log_retention_days: Optional[int] = Field(None, gt=1, le=365,
                                              description="Days to retain logs")
    model_config: Optional[Dict[str, Any]] = Field(None, description="AI model configuration")

# ============================================================================
# Admin Endpoint Implementations
# ============================================================================

@router.get("/health", response_model=SystemHealthResponse)
async def get_system_health(
    current_user: Dict[str, Any] = Depends(get_current_admin_user),
    db: AsyncSession = Depends(get_db),
    audit_service: AuditService = Depends()
):
    """
    Get comprehensive system health status.
    Requires admin privileges.
    
    This endpoint checks all critical services and returns their status:
    - Database connectivity and performance
    - Redis cache availability
    - Vector database (Qdrant) health
    - AI agent status
    - API service metrics
    
    Args:
        current_user: Authenticated admin user from dependency
        db: Database session for health checks
        audit_service: Service for logging admin actions
        
    Returns:
        SystemHealthResponse with detailed status information
        
    Raises:
        HTTPException: If health check fails or user is not admin
    """
    start_time = datetime.now()
    
    try:
        # Log admin action for audit trail
        await audit_service.log_action(
            user_id=current_user["id"],
            action="SYSTEM_HEALTH_CHECK",
            resource="system",
            details={"timestamp": start_time.isoformat()},
            ip_address=current_user.get("ip_address", "unknown")
        )
        
        # Initialize health status dictionary
        health_status = {
            "database": {"status": "unknown", "latency_ms": 0},
            "redis": {"status": "unknown", "latency_ms": 0},
            "qdrant": {"status": "unknown", "latency_ms": 0},
            "agents": {"status": "unknown", "active_count": 0},
            "api": {"status": "healthy", "version": settings.VERSION}
        }
        
        # Check database connectivity
        try:
            db_start = datetime.now()
            # Execute simple query to check database
            result = await db.execute(select([func.count()]).select_from(table_name="users"))
            db_latency = (datetime.now() - db_start).total_seconds() * 1000
            health_status["database"] = {
                "status": "healthy",
                "latency_ms": round(db_latency, 2),
                "connections": await get_active_db_connections(db)
            }
        except Exception as e:
            logger.error(f"Database health check failed: {str(e)}")
            health_status["database"] = {
                "status": "unhealthy",
                "error": str(e),
                "latency_ms": 0
            }
        
        # Check Redis connectivity
        try:
            redis_start = datetime.now()
            redis_client = await get_redis_client()
            await redis_client.ping()
            redis_latency = (datetime.now() - redis_start).total_seconds() * 1000
            health_status["redis"] = {
                "status": "healthy",
                "latency_ms": round(redis_latency, 2),
                "memory_usage": await get_redis_memory_usage(redis_client)
            }
        except Exception as e:
            logger.error(f"Redis health check failed: {str(e)}")
            health_status["redis"] = {
                "status": "unhealthy",
                "error": str(e),
                "latency_ms": 0
            }
        
        # Check Qdrant vector database
        try:
            qdrant_start = datetime.now()
            qdrant_client = await get_qdrant_client()
            collection_info = await qdrant_client.get_collection(settings.QDRANT_COLLECTION)
            qdrant_latency = (datetime.now() - qdrant_start).total_seconds() * 1000
            health_status["qdrant"] = {
                "status": "healthy",
                "latency_ms": round(qdrant_latency, 2),
                "vectors_count": collection_info.vectors_count,
                "collection_status": collection_info.status
            }
        except Exception as e:
            logger.error(f"Qdrant health check failed: {str(e)}")
            health_status["qdrant"] = {
                "status": "unhealthy",
                "error": str(e),
                "latency_ms": 0
            }
        
        # Check AI agents status
        try:
            agent_status = await get_agent_status()
            health_status["agents"] = {
                "status": "healthy" if agent_status["healthy_count"] > 0 else "degraded",
                "active_count": agent_status["active_count"],
                "healthy_count": agent_status["healthy_count"],
                "total_count": agent_status["total_count"],
                "agents": agent_status["agents"]
            }
        except Exception as e:
            logger.error(f"Agent health check failed: {str(e)}")
            health_status["agents"] = {
                "status": "unhealthy",
                "error": str(e),
                "active_count": 0
            }
        
        # Calculate active users
        try:
            active_users = await get_active_users_count(db, minutes=15)
            health_status["api"]["active_users"] = active_users
        except Exception as e:
            logger.warning(f"Could not get active users: {str(e)}")
            active_users = 0
        
        # Calculate total requests in last 24 hours
        try:
            total_requests = await get_request_count_24h(db)
        except Exception as e:
            logger.warning(f"Could not get request count: {str(e)}")
            total_requests = 0
        
        # Determine overall system status
        overall_status = "healthy"
        for service, status in health_status.items():
            if service != "api" and status.get("status") == "unhealthy":
                overall_status = "degraded"
                break
        
        # Calculate total response time
        total_latency = (datetime.now() - start_time).total_seconds() * 1000
        
        # Build and return response
        return SystemHealthResponse(
            status=overall_status,
            timestamp=datetime.now(),
            database=health_status["database"],
            redis=health_status["redis"],
            qdrant=health_status["qdrant"],
            agents=health_status["agents"],
            api=health_status["api"],
            response_time_ms=round(total_latency, 2),
            active_users=active_users,
            total_requests_24h=total_requests
        )
        
    except Exception as e:
        logger.error(f"Health check failed with unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Health check failed: {str(e)}"
        )

@router.get("/users", response_model=List[UserResponse])
async def get_all_users(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Number of records to return"),
    role: Optional[str] = Query(None, description="Filter by user role"),
    is_active: Optional[bool] = Query(None, description="Filter by active status"),
    search: Optional[str] = Query(None, min_length=2, description="Search in username, email, or full name"),
    current_user: Dict[str, Any] = Depends(get_current_admin_user),
    db: AsyncSession = Depends(get_db),
    audit_service: AuditService = Depends()
):
    """
    Get all users with optional filtering and pagination.
    Admin only endpoint.
    
    Args:
        skip: Number of users to skip for pagination
        limit: Maximum number of users to return
        role: Optional filter by user role
        is_active: Optional filter by active status
        search: Optional search term for username, email, or full name
        current_user: Authenticated admin user
        db: Database session
        audit_service: Audit logging service
        
    Returns:
        List of UserResponse objects
        
    Raises:
        HTTPException: If query fails or user is not admin
    """
    try:
        # Log admin action
        await audit_service.log_action(
            user_id=current_user["id"],
            action="LIST_USERS",
            resource="users",
            details={"skip": skip, "limit": limit, "role": role, "search": search},
            ip_address=current_user.get("ip_address", "unknown")
        )
        
        # Build query with filters
        from models.user import User  # Import here to avoid circular imports
        
        query = select(User)
        
        # Apply role filter
        if role:
            query = query.where(User.role == role)
        
        # Apply active status filter
        if is_active is not None:
            query = query.where(User.is_active == is_active)
        
        # Apply search filter
        if search:
            search_pattern = f"%{search}%"
            query = query.where(
                (User.username.ilike(search_pattern)) |
                (User.email.ilike(search_pattern)) |
                (User.full_name.ilike(search_pattern))
            )
        
        # Apply pagination
        query = query.offset(skip).limit(limit).order_by(User.created_at.desc())
        
        # Execute query
        result = await db.execute(query)
        users = result.scalars().all()
        
        # Convert to response models
        user_responses = []
        for user in users:
            user_dict = {
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "full_name": user.full_name,
                "role": user.role,
                "department": user.department,
                "is_active": user.is_active,
                "created_at": user.created_at,
                "last_login": user.last_login,
                "created_by": user.created_by
            }
            user_responses.append(UserResponse(**user_dict))
        
        logger.info(f"Retrieved {len(user_responses)} users for admin {current_user['username']}")
        return user_responses
        
    except Exception as e:
        logger.error(f"Error retrieving users: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve users: {str(e)}"
        )

@router.post("/users", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(
    user_data: UserCreateRequest,
    current_user: Dict[str, Any] = Depends(get_current_admin_user),
    db: AsyncSession = Depends(get_db),
    audit_service: AuditService = Depends(),
    notification_service: NotificationService = Depends()
):
    """
    Create a new user in the system.
    Admin only endpoint.
    
    This endpoint creates a new user account with the specified role and permissions.
    The password is hashed before storage, and an audit log is created.
    
    Args:
        user_data: Validated user creation data
        current_user: Authenticated admin user creating this account
        db: Database session
        audit_service: Audit logging service
        notification_service: Service for sending notifications
        
    Returns:
        Created user data (excluding password)
        
    Raises:
        HTTPException: If username/email exists or validation fails
    """
    try:
        # Import User model
        from models.user import User
        
        # Check if username already exists
        existing_username = await db.execute(
            select(User).where(User.username == user_data.username)
        )
        if existing_username.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Username '{user_data.username}' already exists"
            )
        
        # Check if email already exists
        existing_email = await db.execute(
            select(User).where(User.email == user_data.email)
        )
        if existing_email.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Email '{user_data.email}' already registered"
            )
        
        # Hash the password
        hashed_password = hash_password(user_data.password)
        
        # Create new user instance
        new_user = User(
            username=user_data.username,
            email=user_data.email,
            hashed_password=hashed_password,
            full_name=user_data.full_name,
            role=user_data.role,
            department=user_data.department,
            is_active=user_data.is_active,
            created_at=datetime.now(),
            created_by=current_user["id"]
        )
        
        # Add to database
        db.add(new_user)
        await db.commit()
        await db.refresh(new_user)
        
        # Log the action
        await audit_service.log_action(
            user_id=current_user["id"],
            action="CREATE_USER",
            resource="users",
            resource_id=new_user.id,
            details={"username": user_data.username, "role": user_data.role},
            ip_address=current_user.get("ip_address", "unknown"),
            status="success"
        )
        
        # Send welcome notification
        await notification_service.send_welcome_email(
            email=user_data.email,
            name=user_data.full_name,
            username=user_data.username,
            created_by=current_user["full_name"]
        )
        
        logger.info(f"User {user_data.username} created by admin {current_user['username']}")
        
        # Return created user
        return UserResponse(
            id=new_user.id,
            username=new_user.username,
            email=new_user.email,
            full_name=new_user.full_name,
            role=new_user.role,
            department=new_user.department,
            is_active=new_user.is_active,
            created_at=new_user.created_at,
            last_login=new_user.last_login,
            created_by=new_user.created_by
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error creating user: {str(e)}", exc_info=True)
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create user: {str(e)}"
        )

@router.get("/users/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: str,
    current_user: Dict[str, Any] = Depends(get_current_admin_user),
    db: AsyncSession = Depends(get_db),
    audit_service: AuditService = Depends()
):
    """
    Get detailed information about a specific user.
    Admin only endpoint.
    
    Args:
        user_id: Unique identifier of the user
        current_user: Authenticated admin user
        db: Database session
        audit_service: Audit logging service
        
    Returns:
        Detailed user information
        
    Raises:
        HTTPException: If user not found or access denied
    """
    try:
        # Import User model
        from models.user import User
        
        # Query user by ID
        result = await db.execute(
            select(User).where(User.id == user_id)
        )
        user = result.scalar_one_or_none()
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User with ID {user_id} not found"
            )
        
        # Log the action
        await audit_service.log_action(
            user_id=current_user["id"],
            action="VIEW_USER",
            resource="users",
            resource_id=user_id,
            details={"username": user.username},
            ip_address=current_user.get("ip_address", "unknown")
        )
        
        # Return user data
        return UserResponse(
            id=user.id,
            username=user.username,
            email=user.email,
            full_name=user.full_name,
            role=user.role,
            department=user.department,
            is_active=user.is_active,
            created_at=user.created_at,
            last_login=user.last_login,
            created_by=user.created_by
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving user {user_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve user: {str(e)}"
        )

@router.put("/users/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: str,
    user_update: UserUpdateRequest,
    current_user: Dict[str, Any] = Depends(get_current_admin_user),
    db: AsyncSession = Depends(get_db),
    audit_service: AuditService = Depends()
):
    """
    Update an existing user's information.
    Admin only endpoint.
    
    Args:
        user_id: ID of user to update
        user_update: Partial update data
        current_user: Authenticated admin user
        db: Database session
        audit_service: Audit logging service
        
    Returns:
        Updated user information
        
    Raises:
        HTTPException: If user not found or update fails
    """
    try:
        # Import User model
        from models.user import User
        
        # Find user
        result = await db.execute(
            select(User).where(User.id == user_id)
        )
        user = result.scalar_one_or_none()
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User with ID {user_id} not found"
            )
        
        # Store original values for audit
        original_values = {
            "email": user.email,
            "full_name": user.full_name,
            "department": user.department,
            "is_active": user.is_active,
            "role": user.role
        }
        
        # Update fields if provided
        update_data = user_update.dict(exclude_unset=True)
        changed_fields = []
        
        for field, value in update_data.items():
            if hasattr(user, field):
                # Check if value actually changed
                if getattr(user, field) != value:
                    setattr(user, field, value)
                    changed_fields.append(field)
        
        # If email changed, check for uniqueness
        if "email" in update_data and update_data["email"] != original_values["email"]:
            # Check if new email already exists for another user
            existing = await db.execute(
                select(User).where(User.email == update_data["email"], User.id != user_id)
            )
            if existing.scalar_one_or_none():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Email '{update_data['email']}' already in use"
                )
        
        # Update timestamp
        user.updated_at = datetime.now()
        
        # Commit changes
        await db.commit()
        await db.refresh(user)
        
        # Log the action
        await audit_service.log_action(
            user_id=current_user["id"],
            action="UPDATE_USER",
            resource="users",
            resource_id=user_id,
            details={
                "username": user.username,
                "changed_fields": changed_fields,
                "original": original_values,
                "updated": update_data
            },
            ip_address=current_user.get("ip_address", "unknown"),
            status="success"
        )
        
        logger.info(f"User {user.username} updated by admin {current_user['username']}")
        
        # Return updated user
        return UserResponse(
            id=user.id,
            username=user.username,
            email=user.email,
            full_name=user.full_name,
            role=user.role,
            department=user.department,
            is_active=user.is_active,
            created_at=user.created_at,
            last_login=user.last_login,
            created_by=user.created_by
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating user {user_id}: {str(e)}", exc_info=True)
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update user: {str(e)}"
        )

@router.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(
    user_id: str,
    permanent: bool = Query(False, description="Permanently delete instead of deactivate"),
    current_user: Dict[str, Any] = Depends(get_current_admin_user),
    db: AsyncSession = Depends(get_db),
    audit_service: AuditService = Depends()
):
    """
    Delete or deactivate a user.
    Admin only endpoint.
    
    By default, users are deactivated (soft delete) rather than permanently deleted.
    Use permanent=true for hard deletion (removes all traces).
    
    Args:
        user_id: ID of user to delete
        permanent: If true, permanently delete user; if false, just deactivate
        current_user: Authenticated admin user
        db: Database session
        audit_service: Audit logging service
        
    Returns:
        204 No Content on success
        
    Raises:
        HTTPException: If user not found or cannot delete self
    """
    try:
        # Import User model
        from models.user import User
        
        # Prevent admin from deleting themselves
        if user_id == current_user["id"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot delete your own admin account"
            )
        
        # Find user
        result = await db.execute(
            select(User).where(User.id == user_id)
        )
        user = result.scalar_one_or_none()
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User with ID {user_id} not found"
            )
        
        if permanent:
            # Permanently delete user
            await db.delete(user)
            action = "PERMANENTLY_DELETE_USER"
            logger.warning(f"User {user.username} permanently deleted by admin {current_user['username']}")
        else:
            # Soft delete - just deactivate
            user.is_active = False
            user.deactivated_at = datetime.now()
            user.deactivated_by = current_user["id"]
            action = "DEACTIVATE_USER"
            logger.info(f"User {user.username} deactivated by admin {current_user['username']}")
        
        # Commit changes
        await db.commit()
        
        # Log the action
        await audit_service.log_action(
            user_id=current_user["id"],
            action=action,
            resource="users",
            resource_id=user_id,
            details={
                "username": user.username,
                "permanent": permanent,
                "email": user.email
            },
            ip_address=current_user.get("ip_address", "unknown"),
            status="success"
        )
        
        # Return no content
        return None
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting user {user_id}: {str(e)}", exc_info=True)
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete user: {str(e)}"
        )

@router.get("/audit-logs", response_model=List[AuditLogResponse])
async def get_audit_logs(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Number of records to return"),
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    action: Optional[str] = Query(None, description="Filter by action type"),
    resource: Optional[str] = Query(None, description="Filter by resource type"),
    start_date: Optional[datetime] = Query(None, description="Filter by start date"),
    end_date: Optional[datetime] = Query(None, description="Filter by end date"),
    status: Optional[str] = Query(None, description="Filter by status (success/failure)"),
    current_user: Dict[str, Any] = Depends(get_current_admin_user),
    db: AsyncSession = Depends(get_db),
    audit_service: AuditService = Depends()
):
    """
    Retrieve audit logs with filtering and pagination.
    Admin only endpoint for compliance and security monitoring.
    
    Args:
        skip: Pagination offset
        limit: Maximum records to return
        user_id: Filter by specific user
        action: Filter by action type (CREATE_USER, UPDATE_USER, etc.)
        resource: Filter by resource type (users, patients, agents, etc.)
        start_date: Filter logs after this date
        end_date: Filter logs before this date
        status: Filter by success or failure
        current_user: Authenticated admin user
        db: Database session
        audit_service: Audit logging service
        
    Returns:
        List of audit log entries
        
    Raises:
        HTTPException: If query fails
    """
    try:
        # Import AuditLog model
        from models.audit import AuditLog
        
        # Build query with filters
        query = select(AuditLog)
        
        if user_id:
            query = query.where(AuditLog.user_id == user_id)
        
        if action:
            query = query.where(AuditLog.action == action)
        
        if resource:
            query = query.where(AuditLog.resource == resource)
        
        if start_date:
            query = query.where(AuditLog.timestamp >= start_date)
        
        if end_date:
            query = query.where(AuditLog.timestamp <= end_date)
        
        if status:
            query = query.where(AuditLog.status == status)
        
        # Apply pagination and ordering
        query = query.order_by(desc(AuditLog.timestamp)).offset(skip).limit(limit)
        
        # Execute query
        result = await db.execute(query)
        logs = result.scalars().all()
        
        # Log this audit view action
        await audit_service.log_action(
            user_id=current_user["id"],
            action="VIEW_AUDIT_LOGS",
            resource="audit",
            details={
                "filters": {
                    "user_id": user_id,
                    "action": action,
                    "resource": resource,
                    "start_date": start_date.isoformat() if start_date else None,
                    "end_date": end_date.isoformat() if end_date else None,
                    "status": status
                },
                "result_count": len(logs)
            },
            ip_address=current_user.get("ip_address", "unknown"),
            status="success"
        )
        
        # Convert to response models
        log_responses = []
        for log in logs:
            log_dict = {
                "id": log.id,
                "user_id": log.user_id,
                "username": log.username,
                "action": log.action,
                "resource": log.resource,
                "resource_id": log.resource_id,
                "details": log.details,
                "ip_address": log.ip_address,
                "user_agent": log.user_agent,
                "timestamp": log.timestamp,
                "status": log.status
            }
            log_responses.append(AuditLogResponse(**log_dict))
        
        return log_responses
        
    except Exception as e:
        logger.error(f"Error retrieving audit logs: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve audit logs: {str(e)}"
        )

@router.get("/metrics/agents", response_model=AgentMetricsResponse)
async def get_agent_metrics(
    timeframe: str = Query("24h", regex="^(1h|6h|24h|7d|30d)$"),
    current_user: Dict[str, Any] = Depends(get_current_admin_user),
    audit_service: AuditService = Depends()
):
    """
    Get performance metrics for all AI agents.
    Admin only endpoint for monitoring agent effectiveness.
    
    Args:
        timeframe: Time period for metrics (1h, 6h, 24h, 7d, 30d)
        current_user: Authenticated admin user
        audit_service: Audit logging service
        
    Returns:
        Comprehensive agent metrics including:
        - Response times
        - Success rates
        - Confidence scores
        - Usage statistics
        - Error rates
        
    Raises:
        HTTPException: If metrics collection fails
    """
    try:
        # Log the request
        await audit_service.log_action(
            user_id=current_user["id"],
            action="VIEW_AGENT_METRICS",
            resource="agents",
            details={"timeframe": timeframe},
            ip_address=current_user.get("ip_address", "unknown")
        )
        
        # Calculate time range based on timeframe
        end_time = datetime.now()
        if timeframe == "1h":
            start_time = end_time - timedelta(hours=1)
        elif timeframe == "6h":
            start_time = end_time - timedelta(hours=6)
        elif timeframe == "24h":
            start_time = end_time - timedelta(days=1)
        elif timeframe == "7d":
            start_time = end_time - timedelta(days=7)
        elif timeframe == "30d":
            start_time = end_time - timedelta(days=30)
        else:
            start_time = end_time - timedelta(days=1)  # Default to 24h
        
        # In production, this would query a metrics database (Prometheus, InfluxDB, etc.)
        # For demonstration, we'll generate sample metrics
        
        # Get agent statistics from monitoring system
        agent_stats = await get_agent_statistics(start_time, end_time)
        
        # Calculate overall metrics
        total_requests = sum(agent["total_requests"] for agent in agent_stats.values())
        total_errors = sum(agent["error_count"] for agent in agent_stats.values())
        avg_response_time = sum(agent["avg_response_time_ms"] for agent in agent_stats.values()) / len(agent_stats) if agent_stats else 0
        
        # Calculate success rate
        success_rate = ((total_requests - total_errors) / total_requests * 100) if total_requests > 0 else 0
        
        # Build response
        metrics = AgentMetricsResponse(
            timeframe=timeframe,
            start_time=start_time,
            end_time=end_time,
            total_requests=total_requests,
            total_errors=total_errors,
            success_rate=round(success_rate, 2),
            avg_response_time_ms=round(avg_response_time, 2),
            agents=agent_stats,
            timestamp=datetime.now()
        )
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error retrieving agent metrics: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve agent metrics: {str(e)}"
        )

@router.get("/metrics/system")
async def get_system_metrics(
    current_user: Dict[str, Any] = Depends(get_current_admin_user),
    audit_service: AuditService = Depends()
):
    """
    Get comprehensive system performance metrics.
    Admin only endpoint for capacity planning and performance monitoring.
    
    Returns:
        Dictionary containing:
        - CPU and memory usage
        - Database performance
        - Cache hit rates
        - API request rates
        - Error rates by endpoint
        
    Raises:
        HTTPException: If metrics collection fails
    """
    try:
        # Log the request
        await audit_service.log_action(
            user_id=current_user["id"],
            action="VIEW_SYSTEM_METRICS",
            resource="system",
            ip_address=current_user.get("ip_address", "unknown")
        )
        
        # Collect metrics from various sources
        metrics = {
            "cpu": await get_cpu_metrics(),
            "memory": await get_memory_metrics(),
            "database": await get_database_metrics(),
            "cache": await get_cache_metrics(),
            "api": await get_api_metrics(),
            "timestamp": datetime.now().isoformat()
        }
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error retrieving system metrics: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve system metrics: {str(e)}"
        )

@router.post("/settings")
async def update_system_settings(
    settings_update: SystemSettingsUpdate,
    current_user: Dict[str, Any] = Depends(get_current_admin_user),
    audit_service: AuditService = Depends()
):
    """
    Update system-wide configuration settings.
    Admin only endpoint for platform configuration.
    
    Args:
        settings_update: New settings values to apply
        current_user: Authenticated admin user
        audit_service: Audit logging service
        
    Returns:
        Updated settings with confirmation
        
    Raises:
        HTTPException: If update fails
    """
    try:
        # Log the update attempt
        await audit_service.log_action(
            user_id=current_user["id"],
            action="UPDATE_SYSTEM_SETTINGS",
            resource="settings",
            details=settings_update.dict(exclude_unset=True),
            ip_address=current_user.get("ip_address", "unknown")
        )
        
        # Update settings in database or config file
        updated_settings = await apply_system_settings(settings_update.dict(exclude_unset=True))
        
        logger.info(f"System settings updated by admin {current_user['username']}: {settings_update.dict(exclude_unset=True)}")
        
        return {
            "status": "success",
            "message": "System settings updated successfully",
            "updated_settings": updated_settings,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error updating system settings: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update system settings: {str(e)}"
        )

@router.post("/backup")
async def create_system_backup(
    backup_type: str = Query("full", regex="^(full|incremental|config)$"),
    current_user: Dict[str, Any] = Depends(get_current_admin_user),
    audit_service: AuditService = Depends()
):
    """
    Create a system backup.
    Admin only endpoint for disaster recovery.
    
    Args:
        backup_type: Type of backup (full, incremental, config)
        current_user: Authenticated admin user
        audit_service: Audit logging service
        
    Returns:
        Backup information including location and size
        
    Raises:
        HTTPException: If backup fails
    """
    try:
        # Log backup initiation
        await audit_service.log_action(
            user_id=current_user["id"],
            action="CREATE_BACKUP",
            resource="system",
            details={"backup_type": backup_type},
            ip_address=current_user.get("ip_address", "unknown")
        )
        
        # Trigger backup process (would connect to backup service in production)
        backup_info = await trigger_system_backup(backup_type)
        
        logger.info(f"System backup created by admin {current_user['username']}: {backup_type}")
        
        return {
            "status": "success",
            "message": f"{backup_type.capitalize()} backup initiated",
            "backup_id": backup_info.get("backup_id"),
            "location": backup_info.get("location"),
            "size_mb": backup_info.get("size_mb"),
            "estimated_completion": backup_info.get("estimated_completion"),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error creating backup: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create backup: {str(e)}"
        )

# ============================================================================
# Helper Functions
# ============================================================================

async def get_active_db_connections(db: AsyncSession) -> int:
    """
    Get number of active database connections.
    Helper function for health checks.
    """
    try:
        result = await db.execute("SELECT count(*) FROM pg_stat_activity WHERE state = 'active'")
        return result.scalar() or 0
    except:
        return 0

async def get_redis_memory_usage(redis_client) -> Dict[str, Any]:
    """
    Get Redis memory usage statistics.
    """
    try:
        info = await redis_client.info("memory")
        return {
            "used_memory_human": info.get("used_memory_human", "unknown"),
            "peak_memory_human": info.get("peak_memory_human", "unknown"),
            "maxmemory_human": info.get("maxmemory_human", "unknown"),
            "memory_fragmentation_ratio": info.get("mem_fragmentation_ratio", 0)
        }
    except:
        return {"error": "Could not retrieve memory usage"}

async def get_active_users_count(db: AsyncSession, minutes: int = 15) -> int:
    """
    Get number of users active in the last N minutes.
    """
    try:
        cutoff = datetime.now() - timedelta(minutes=minutes)
        from models.user import User
        result = await db.execute(
            select(func.count(User.id)).where(User.last_activity >= cutoff)
        )
        return result.scalar() or 0
    except:
        return 0

async def get_request_count_24h(db: AsyncSession) -> int:
    """
    Get total API requests in last 24 hours.
    """
    try:
        cutoff = datetime.now() - timedelta(days=1)
        from models.audit import AuditLog
        result = await db.execute(
            select(func.count(AuditLog.id)).where(AuditLog.timestamp >= cutoff)
        )
        return result.scalar() or 0
    except:
        return 0

async def get_agent_status() -> Dict[str, Any]:
    """
    Get status of all AI agents.
    Queries the agent orchestrator for current status.
    """
    # In production, this would query the agent orchestrator
    # For demonstration, return sample data
    return {
        "active_count": 6,
        "healthy_count": 6,
        "total_count": 6,
        "agents": {
            "clinical_agent": {"status": "healthy", "tasks_processed": 1250},
            "risk_agent": {"status": "healthy", "tasks_processed": 890},
            "operations_agent": {"status": "healthy", "tasks_processed": 2340},
            "director_agent": {"status": "healthy", "tasks_processed": 450},
            "compliance_agent": {"status": "healthy", "tasks_processed": 6780},
            "research_agent": {"status": "healthy", "tasks_processed": 560}
        }
    }

async def get_agent_statistics(start_time: datetime, end_time: datetime) -> Dict[str, Any]:
    """
    Get detailed statistics for each agent.
    """
    # In production, this would query a metrics database
    # For demonstration, return sample data
    return {
        "clinical_agent": {
            "total_requests": 1250,
            "error_count": 12,
            "avg_response_time_ms": 345,
            "avg_confidence": 0.89,
            "p95_response_time_ms": 520
        },
        "risk_agent": {
            "total_requests": 890,
            "error_count": 5,
            "avg_response_time_ms": 420,
            "avg_confidence": 0.92,
            "p95_response_time_ms": 680
        },
        "operations_agent": {
            "total_requests": 2340,
            "error_count": 28,
            "avg_response_time_ms": 280,
            "avg_confidence": 0.85,
            "p95_response_time_ms": 450
        },
        "director_agent": {
            "total_requests": 450,
            "error_count": 3,
            "avg_response_time_ms": 560,
            "avg_confidence": 0.94,
            "p95_response_time_ms": 820
        },
        "compliance_agent": {
            "total_requests": 6780,
            "error_count": 15,
            "avg_response_time_ms": 120,
            "avg_confidence": 0.99,
            "p95_response_time_ms": 180
        },
        "research_agent": {
            "total_requests": 560,
            "error_count": 8,
            "avg_response_time_ms": 890,
            "avg_confidence": 0.87,
            "p95_response_time_ms": 1450
        }
    }

async def get_cpu_metrics() -> Dict[str, Any]:
    """Get CPU usage metrics."""
    import psutil
    return {
        "percent": psutil.cpu_percent(interval=1),
        "count": psutil.cpu_count(),
        "frequency_mhz": psutil.cpu_freq().current if psutil.cpu_freq() else None
    }

async def get_memory_metrics() -> Dict[str, Any]:
    """Get memory usage metrics."""
    import psutil
    mem = psutil.virtual_memory()
    return {
        "total_gb": mem.total / (1024**3),
        "available_gb": mem.available / (1024**3),
        "percent": mem.percent
    }

async def get_database_metrics() -> Dict[str, Any]:
    """Get database performance metrics."""
    # Would query PostgreSQL statistics
    return {
        "connections": 25,
        "queries_per_second": 150,
        "cache_hit_ratio": 0.97,
        "replication_lag_seconds": 0
    }

async def get_cache_metrics() -> Dict[str, Any]:
    """Get Redis cache metrics."""
    return {
        "hit_rate": 0.89,
        "miss_rate": 0.11,
        "keys_count": 12500,
        "memory_used_mb": 256
    }

async def get_api_metrics() -> Dict[str, Any]:
    """Get API performance metrics."""
    return {
        "requests_per_second": 45,
        "avg_response_time_ms": 180,
        "p95_response_time_ms": 320,
        "p99_response_time_ms": 550,
        "error_rate_percent": 0.5
    }

async def apply_system_settings(settings: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply system settings changes.
    Updates configuration in database and reloads affected services.
    """
    # In production, this would update settings in database
    # and potentially trigger service reloads
    return {
        **settings,
        "applied_at": datetime.now().isoformat(),
        "version": settings.VERSION
    }

async def trigger_system_backup(backup_type: str) -> Dict[str, Any]:
    """
    Trigger a system backup.
    In production, this would call backup service.
    """
    import uuid
    return {
        "backup_id": str(uuid.uuid4()),
        "location": f"/backups/{backup_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "size_mb": 1250,
        "estimated_completion": (datetime.now() + timedelta(minutes=5)).isoformat()
    }