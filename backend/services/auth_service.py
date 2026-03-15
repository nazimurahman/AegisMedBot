"""
Authentication service for AegisMedBot.

This module handles all authentication-related functionality including
JWT token generation and validation, password hashing, and user session management.
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Union
import jwt
from passlib.context import CryptContext
from fastapi import HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uuid
import logging
from redis.asyncio import Redis

from ..core.config import settings
from ..models.enums import UserRole, Permission

# Configure logging
logger = logging.getLogger(__name__)

# Password hashing context
pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto",
    bcrypt__rounds=12  # Higher rounds for better security
)


class AuthService:
    """
    Authentication service handling user authentication and authorization.
    
    This service provides methods for:
    - Password hashing and verification
    - JWT token generation and validation
    - User session management
    - Role-based permission checking
    """
    
    def __init__(self, redis_client: Optional[Redis] = None):
        """
        Initialize the authentication service.
        
        Args:
            redis_client: Optional Redis client for session storage
        """
        self.redis_client = redis_client
        self.secret_key = settings.SECRET_KEY
        self.algorithm = settings.ALGORITHM
        self.access_token_expire_minutes = settings.ACCESS_TOKEN_EXPIRE_MINUTES
        self.refresh_token_expire_days = settings.REFRESH_TOKEN_EXPIRE_DAYS
        
        logger.info("AuthService initialized")
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """
        Verify a plain password against a hashed password.
        
        Args:
            plain_password: Password in plain text
            hashed_password: Hashed password from database
            
        Returns:
            True if password matches, False otherwise
        """
        try:
            return pwd_context.verify(plain_password, hashed_password)
        except Exception as e:
            logger.error(f"Password verification error: {str(e)}")
            return False
    
    def hash_password(self, password: str) -> str:
        """
        Hash a password for secure storage.
        
        Args:
            password: Plain text password
            
        Returns:
            Hashed password string
        """
        try:
            return pwd_context.hash(password)
        except Exception as e:
            logger.error(f"Password hashing error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error processing password"
            )
    
    def create_access_token(
        self,
        data: Dict[str, Any],
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """
        Create a JWT access token.
        
        Args:
            data: Data to encode in the token
            expires_delta: Optional custom expiration time
            
        Returns:
            JWT token string
        """
        to_encode = data.copy()
        
        # Set expiration time
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        
        to_encode.update({
            "exp": expire,
            "iat": datetime.utcnow(),
            "jti": str(uuid.uuid4()),  # Unique token ID
            "type": "access"
        })
        
        try:
            encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
            return encoded_jwt
        except Exception as e:
            logger.error(f"Token creation error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error creating access token"
            )
    
    def create_refresh_token(self, user_id: str) -> str:
        """
        Create a JWT refresh token.
        
        Args:
            user_id: User ID to encode in the token
            
        Returns:
            JWT refresh token string
        """
        expire = datetime.utcnow() + timedelta(days=self.refresh_token_expire_days)
        
        to_encode = {
            "sub": user_id,
            "exp": expire,
            "iat": datetime.utcnow(),
            "jti": str(uuid.uuid4()),
            "type": "refresh"
        }
        
        try:
            encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
            return encoded_jwt
        except Exception as e:
            logger.error(f"Refresh token creation error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error creating refresh token"
            )
    
    def decode_token(self, token: str) -> Dict[str, Any]:
        """
        Decode and validate a JWT token.
        
        Args:
            token: JWT token string
            
        Returns:
            Decoded token payload
            
        Raises:
            HTTPException: If token is invalid or expired
        """
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm]
            )
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired",
                headers={"WWW-Authenticate": "Bearer"}
            )
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
                headers={"WWW-Authenticate": "Bearer"}
            )
    
    async def validate_token(self, token: str) -> Dict[str, Any]:
        """
        Validate a token and check if it's been revoked.
        
        Args:
            token: JWT token string
            
        Returns:
            Validated token payload
        """
        # Decode the token
        payload = self.decode_token(token)
        
        # Check if token has been revoked (if Redis is available)
        if self.redis_client:
            token_id = payload.get("jti")
            if token_id:
                revoked = await self.redis_client.get(f"revoked_token:{token_id}")
                if revoked:
                    logger.warning(f"Revoked token used: {token_id}")
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Token has been revoked",
                        headers={"WWW-Authenticate": "Bearer"}
                    )
        
        return payload
    
    async def revoke_token(self, token: str) -> bool:
        """
        Revoke a token by adding it to the revocation list.
        
        Args:
            token: JWT token to revoke
            
        Returns:
            True if token was revoked, False otherwise
        """
        if not self.redis_client:
            logger.warning("Redis not available, cannot revoke token")
            return False
        
        try:
            payload = self.decode_token(token)
            token_id = payload.get("jti")
            exp = payload.get("exp")
            
            if token_id and exp:
                # Calculate time until token expires
                ttl = max(0, exp - datetime.utcnow().timestamp())
                if ttl > 0:
                    await self.redis_client.setex(
                        f"revoked_token:{token_id}",
                        int(ttl),
                        "revoked"
                    )
                    logger.info(f"Token {token_id} revoked")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error revoking token: {str(e)}")
            return False
    
    def get_token_from_request(self, request: Request) -> Optional[str]:
        """
        Extract token from request headers.
        
        Args:
            request: FastAPI request object
            
        Returns:
            Token string or None if not found
        """
        authorization = request.headers.get("Authorization")
        if not authorization:
            return None
        
        parts = authorization.split()
        if len(parts) != 2 or parts[0].lower() != "bearer":
            return None
        
        return parts[1]
    
    def check_permission(
        self,
        user_permissions: List[str],
        required_permission: Union[str, List[str]],
        require_all: bool = False
    ) -> bool:
        """
        Check if user has required permissions.
        
        Args:
            user_permissions: List of permissions the user has
            required_permission: Single permission or list of required permissions
            require_all: If True, user must have all required permissions
                         If False, user needs at least one
            
        Returns:
            True if user has required permissions
        """
        if isinstance(required_permission, str):
            required_permission = [required_permission]
        
        user_permission_set = set(user_permissions)
        required_set = set(required_permission)
        
        if require_all:
            return required_set.issubset(user_permission_set)
        else:
            return len(required_set.intersection(user_permission_set)) > 0
    
    def get_role_permissions(self, role: UserRole) -> List[str]:
        """
        Get default permissions for a role.
        
        Args:
            role: User role
            
        Returns:
            List of permissions for the role
        """
        # Define role-based permission mappings
        role_permissions = {
            UserRole.MEDICAL_DIRECTOR: [
                Permission.VIEW_PATIENT.value,
                Permission.EDIT_PATIENT.value,
                Permission.VIEW_CLINICAL_DATA.value,
                Permission.EDIT_CLINICAL_DATA.value,
                Permission.ACCESS_SENSITIVE_DATA.value,
                Permission.VIEW_AGENTS.value,
                Permission.VIEW_LOGS.value,
                Permission.VIEW_AUDIT.value
            ],
            UserRole.PHYSICIAN: [
                Permission.VIEW_PATIENT.value,
                Permission.EDIT_PATIENT.value,
                Permission.VIEW_CLINICAL_DATA.value,
                Permission.EDIT_CLINICAL_DATA.value,
                Permission.ACCESS_SENSITIVE_DATA.value
            ],
            UserRole.NURSE: [
                Permission.VIEW_PATIENT.value,
                Permission.VIEW_CLINICAL_DATA.value,
                Permission.EDIT_CLINICAL_DATA.value
            ],
            UserRole.ADMINISTRATOR: [
                Permission.VIEW_AGENTS.value,
                Permission.CONFIGURE_AGENTS.value,
                Permission.VIEW_LOGS.value,
                Permission.CONFIGURE_SYSTEM.value,
                Permission.MANAGE_USERS.value,
                Permission.VIEW_AUDIT.value,
                Permission.EXPORT_AUDIT.value
            ],
            UserRole.RESEARCHER: [
                Permission.VIEW_PATIENT.value,
                Permission.VIEW_CLINICAL_DATA.value
            ],
            UserRole.COMPLIANCE_OFFICER: [
                Permission.VIEW_AUDIT.value,
                Permission.EXPORT_AUDIT.value,
                Permission.VIEW_LOGS.value
            ]
        }
        
        return role_permissions.get(role, [])
    
    async def create_user_session(
        self,
        user_id: str,
        user_data: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        Create a new user session with access and refresh tokens.
        
        Args:
            user_id: User identifier
            user_data: Additional user data for token payload
            
        Returns:
            Dictionary containing access_token and refresh_token
        """
        # Create access token
        access_token_payload = {
            "sub": user_id,
            "user_id": user_id,
            "role": user_data.get("role"),
            "permissions": user_data.get("permissions", [])
        }
        
        access_token = self.create_access_token(access_token_payload)
        
        # Create refresh token
        refresh_token = self.create_refresh_token(user_id)
        
        # Store session in Redis if available
        if self.redis_client:
            session_id = str(uuid.uuid4())
            session_data = {
                "user_id": user_id,
                "access_token": access_token,
                "refresh_token": refresh_token,
                "created_at": datetime.utcnow().isoformat()
            }
            
            await self.redis_client.setex(
                f"session:{session_id}",
                self.refresh_token_expire_days * 24 * 3600,
                str(session_data)  # In production, use proper serialization
            )
        
        logger.info(f"Session created for user {user_id}")
        
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer"
        }
    
    async def refresh_access_token(self, refresh_token: str) -> Dict[str, str]:
        """
        Create a new access token using a refresh token.
        
        Args:
            refresh_token: Valid refresh token
            
        Returns:
            New access token
        """
        # Validate refresh token
        payload = self.decode_token(refresh_token)
        
        if payload.get("type") != "refresh":
            logger.warning("Invalid token type for refresh")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )
        
        user_id = payload.get("sub")
        if not user_id:
            logger.warning("No user ID in refresh token")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )
        
        # Create new access token
        # In production, fetch user data from database
        access_token_payload = {
            "sub": user_id,
            "user_id": user_id
        }
        
        access_token = self.create_access_token(access_token_payload)
        
        logger.info(f"Access token refreshed for user {user_id}")
        
        return {
            "access_token": access_token,
            "token_type": "bearer"
        }


class JWTBearer(HTTPBearer):
    """
    FastAPI dependency for JWT authentication.
    
    This class can be used as a dependency in FastAPI routes
    to require valid JWT authentication.
    """
    
    def __init__(
        self,
        auth_service: AuthService,
        auto_error: bool = True,
        required_permissions: Optional[List[str]] = None
    ):
        """
        Initialize the JWT bearer.
        
        Args:
            auth_service: AuthService instance
            auto_error: Whether to automatically raise HTTPException
            required_permissions: Optional list of required permissions
        """
        super().__init__(auto_error=auto_error)
        self.auth_service = auth_service
        self.required_permissions = required_permissions or []
    
    async def __call__(self, request: Request) -> Dict[str, Any]:
        """
        Validate the JWT token from the request.
        
        Args:
            request: FastAPI request object
            
        Returns:
            Validated token payload
            
        Raises:
            HTTPException: If authentication fails
        """
        credentials: HTTPAuthorizationCredentials = await super().__call__(request)
        
        if not credentials:
            if self.auto_error:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Invalid authorization code"
                )
            return None
        
        if credentials.scheme != "Bearer":
            if self.auto_error:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Invalid authentication scheme"
                )
            return None
        
        # Validate the token
        payload = await self.auth_service.validate_token(credentials.credentials)
        
        # Check permissions if required
        if self.required_permissions:
            user_permissions = payload.get("permissions", [])
            has_permission = self.auth_service.check_permission(
                user_permissions,
                self.required_permissions,
                require_all=True
            )
            
            if not has_permission:
                if self.auto_error:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Insufficient permissions"
                    )
                return None
        
        return payload