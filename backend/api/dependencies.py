"""
FastAPI dependencies for AegisMedBot.
Provides reusable dependency injection for common components.
These dependencies are used across multiple endpoints.
"""

from fastapi import Request, Depends, HTTPException, status
from typing import Optional, AsyncGenerator, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
import logging
from ..core.database import get_db
from ..core.cache import get_redis
from ..core.config import settings
from ..services.auth_service import AuthService
from ..services.audit_service import AuditService
from ..services.notification_service import NotificationService
from .middleware.auth import get_current_user, require_roles
import redis.asyncio as redis

logger = logging.getLogger(__name__)

# Database dependency
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency that provides a database session.
    Uses async generator for proper session cleanup.
    
    Yields:
        Async SQLAlchemy session
    """
    async for session in get_db():
        yield session

# Redis dependency
async def get_redis_client() -> AsyncGenerator[redis.Redis, None]:
    """
    Dependency that provides a Redis client.
    
    Yields:
        Async Redis client
    """
    client = await get_redis()
    try:
        yield client
    finally:
        await client.close()

# Service dependencies
async def get_auth_service(
    db: AsyncSession = Depends(get_db_session),
    redis_client: redis.Redis = Depends(get_redis_client)
) -> AuthService:
    """
    Dependency that provides an authenticated AuthService instance.
    
    Args:
        db: Database session
        redis_client: Redis client
        
    Returns:
        Configured AuthService
    """
    return AuthService(db, redis_client, settings)

async def get_audit_service(
    db: AsyncSession = Depends(get_db_session)
) -> AuditService:
    """
    Dependency that provides an AuditService instance.
    
    Args:
        db: Database session
        
    Returns:
        Configured AuditService
    """
    return AuditService(db)

async def get_notification_service(
    redis_client: redis.Redis = Depends(get_redis_client)
) -> NotificationService:
    """
    Dependency that provides a NotificationService instance.
    
    Args:
        redis_client: Redis client for pub/sub
        
    Returns:
        Configured NotificationService
    """
    return NotificationService(redis_client)

# Common pagination parameters
async def get_pagination_params(
    skip: int = 0,
    limit: int = 100
) -> Dict[str, int]:
    """
    Dependency for pagination parameters with validation.
    
    Args:
        skip: Number of records to skip
        limit: Maximum number of records to return
        
    Returns:
        Dictionary with validated pagination parameters
    """
    # Validate skip (must be non-negative)
    if skip < 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Skip parameter must be non-negative"
        )
    
    # Validate limit (must be between 1 and 1000)
    if limit < 1 or limit > 1000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Limit must be between 1 and 1000"
        )
    
    return {"skip": skip, "limit": limit}

# Date range parameters
async def get_date_range_params(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Dict[str, Optional[str]]:
    """
    Dependency for date range parameters with validation.
    
    Args:
        start_date: Start date in ISO format
        end_date: End date in ISO format
        
    Returns:
        Dictionary with validated date parameters
    """
    from datetime import datetime
    
    # Validate date formats if provided
    if start_date:
        try:
            datetime.fromisoformat(start_date)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid start_date format. Use ISO format (YYYY-MM-DD)"
            )
    
    if end_date:
        try:
            datetime.fromisoformat(end_date)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid end_date format. Use ISO format (YYYY-MM-DD)"
            )
    
    return {"start_date": start_date, "end_date": end_date}

# Request ID dependency
async def get_request_id(request: Request) -> str:
    """
    Dependency that returns the current request ID.
    
    Args:
        request: FastAPI request object
        
    Returns:
        Request ID string
    """
    return getattr(request.state, "request_id", "unknown")

# Client info dependency
async def get_client_info(request: Request) -> Dict[str, Any]:
    """
    Dependency that returns client information.
    
    Args:
        request: FastAPI request object
        
    Returns:
        Dictionary with client IP and user agent
    """
    return {
        "ip_address": request.client.host if request.client else None,
        "user_agent": request.headers.get("user-agent"),
        "accept_language": request.headers.get("accept-language")
    }

# Rate limit status dependency
async def get_rate_limit_status(
    request: Request,
    rate_limiter = Depends()  # This would be injected from middleware
) -> Dict[str, Any]:
    """
    Dependency that returns current rate limit status.
    
    Args:
        request: FastAPI request object
        
    Returns:
        Rate limit status dictionary
    """
    rate_info = getattr(request.state, "rate_limit_headers", {})
    return {
        "limit": rate_info.get("X-RateLimit-Limit"),
        "remaining": rate_info.get("X-RateLimit-Remaining"),
        "reset": rate_info.get("X-RateLimit-Reset")
    }

# Role-based access control shortcuts
def require_admin():
    """
    Dependency that requires admin role.
    """
    return require_roles(["admin", "medical_director"])

def require_doctor():
    """
    Dependency that requires doctor role.
    """
    return require_roles(["doctor", "admin", "medical_director"])

def require_nurse():
    """
    Dependency that requires nurse role.
    """
    return require_roles(["nurse", "doctor", "admin", "medical_director"])

# Feature flag dependency
async def check_feature_flag(
    feature_name: str,
    user: Dict[str, Any] = Depends(get_current_user),
    redis_client: redis.Redis = Depends(get_redis_client)
) -> bool:
    """
    Dependency that checks if a feature is enabled for the user.
    
    Args:
        feature_name: Name of the feature to check
        user: Current user information
        redis_client: Redis client for feature flags
        
    Returns:
        True if feature is enabled
        
    Raises:
        HTTPException: If feature is disabled
    """
    # Check if feature is globally enabled
    feature_key = f"feature_flag:{feature_name}"
    is_enabled = await redis_client.get(feature_key)
    
    if is_enabled is None:
        # Feature flag not set, check default from config
        is_enabled = getattr(settings, f"FEATURE_{feature_name.upper()}", False)
    else:
        is_enabled = is_enabled.lower() == "true"
    
    if not is_enabled:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Feature '{feature_name}' is currently disabled"
        )
    
    # Check if user has access (based on role, subscription, etc.)
    user_role = user.get("role")
    role_access_key = f"feature_role:{feature_name}:{user_role}"
    has_role_access = await redis_client.get(role_access_key)
    
    if has_role_access is not None and has_role_access.lower() == "false":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Your role does not have access to feature '{feature_name}'"
        )
    
    return True

# Configuration dependency
async def get_settings():
    """
    Dependency that returns application settings.
    
    Returns:
        Settings object
    """
    return settings

# Combined dependencies for common patterns
class CommonDependencies:
    """
    Container for common dependency combinations.
    """
    
    @staticmethod
    async def authenticated_user(
        request: Request,
        user: Dict[str, Any] = Depends(get_current_user)
    ) -> Dict[str, Any]:
        """Get authenticated user with request context."""
        return {
            "user": user,
            "request_id": getattr(request.state, "request_id"),
            "client_info": await get_client_info(request)
        }
    
    @staticmethod
    async def admin_user(
        user: Dict[str, Any] = Depends(require_admin())
    ) -> Dict[str, Any]:
        """Get admin user."""
        return user
    
    @staticmethod
    async def with_database(
        db: AsyncSession = Depends(get_db_session),
        user: Dict[str, Any] = Depends(get_current_user)
    ) -> Dict[str, Any]:
        """Get database session with authenticated user."""
        return {
            "db": db,
            "user": user
        }
    
    @staticmethod
    async def with_audit(
        db: AsyncSession = Depends(get_db_session),
        audit: AuditService = Depends(get_audit_service),
        user: Dict[str, Any] = Depends(get_current_user)
    ) -> Dict[str, Any]:
        """Get database, audit service, and user."""
        return {
            "db": db,
            "audit": audit,
            "user": user
        }