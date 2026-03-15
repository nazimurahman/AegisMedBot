"""
Authentication middleware for AegisMedBot.
Handles JWT validation, user authentication, and request authorization.
This middleware runs before each request to verify the user's identity.
"""

from fastapi import Request, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional, Dict, Any
import logging
from jose import jwt, JWTError
from datetime import datetime
import json

# Configure logging for security events
logger = logging.getLogger(__name__)

class AuthMiddleware:
    """
    Authentication middleware that validates JWT tokens and extracts user information.
    This middleware runs on every request to ensure proper authentication.
    """
    
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        """
        Initialize the authentication middleware with security parameters.
        
        Args:
            secret_key: The secret key used to verify JWT signatures
            algorithm: The JWT signing algorithm (default: HS256)
        """
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.security = HTTPBearer()  # HTTPBearer extracts the token from Authorization header
        
    async def __call__(self, request: Request) -> Optional[Dict[str, Any]]:
        """
        Process each incoming request to authenticate the user.
        This method is called automatically by FastAPI for each request.
        
        Args:
            request: The incoming FastAPI request object
            
        Returns:
            Dictionary containing user information if authenticated
            
        Raises:
            HTTPException: If authentication fails
        """
        try:
            # Extract the authorization header
            auth_header = request.headers.get("Authorization")
            if not auth_header:
                # No authorization header means the user is not authenticated
                # For public endpoints, we return None instead of raising an exception
                return None
            
            # Parse the bearer token
            # Expected format: "Bearer <token>"
            scheme, token = auth_header.split()
            if scheme.lower() != "bearer":
                # Invalid authentication scheme
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid authentication scheme. Use Bearer token."
                )
            
            # Validate and decode the JWT token
            payload = await self._validate_token(token)
            
            # Attach user information to request state for later use
            request.state.user = payload
            
            # Log successful authentication for audit purposes
            logger.info(f"User authenticated: {payload.get('sub', 'unknown')}")
            
            return payload
            
        except ValueError as e:
            # Handle malformed authorization header
            logger.warning(f"Malformed authorization header: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authorization header format"
            )
        except JWTError as e:
            # Handle invalid or expired tokens
            logger.warning(f"JWT validation failed: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token"
            )
        except Exception as e:
            # Handle unexpected errors
            logger.error(f"Authentication error: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Authentication service error"
            )
    
    async def _validate_token(self, token: str) -> Dict[str, Any]:
        """
        Validate and decode a JWT token.
        
        Args:
            token: The JWT token string to validate
            
        Returns:
            Dictionary containing the decoded token payload
            
        Raises:
            JWTError: If token validation fails
        """
        try:
            # Decode and verify the JWT token
            # This checks signature, expiration, and other claims
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm]
            )
            
            # Check if token has expired
            exp = payload.get("exp")
            if exp and datetime.fromtimestamp(exp) < datetime.now():
                raise JWTError("Token has expired")
            
            # Ensure required claims are present
            required_claims = ["sub", "role", "exp"]
            for claim in required_claims:
                if claim not in payload:
                    raise JWTError(f"Missing required claim: {claim}")
            
            return payload
            
        except JWTError as e:
            # Re-raise JWT errors for specific handling
            raise
        except Exception as e:
            # Wrap unexpected errors
            raise JWTError(f"Token validation failed: {str(e)}")
    
    async def verify_role(self, request: Request, required_roles: list) -> bool:
        """
        Verify that the authenticated user has one of the required roles.
        
        Args:
            request: The FastAPI request object
            required_roles: List of roles that are allowed to access the resource
            
        Returns:
            True if user has required role, False otherwise
        """
        # Get user from request state (set during authentication)
        user = getattr(request.state, "user", None)
        if not user:
            return False
        
        # Check if user's role is in the list of required roles
        user_role = user.get("role")
        return user_role in required_roles

# Dependency for FastAPI to inject the auth middleware
async def get_current_user(request: Request) -> Dict[str, Any]:
    """
    FastAPI dependency to get the currently authenticated user.
    This can be used in endpoint functions to access user information.
    
    Args:
        request: The FastAPI request object
        
    Returns:
        Dictionary containing user information
        
    Raises:
        HTTPException: If user is not authenticated
    """
    user = getattr(request.state, "user", None)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated"
        )
    return user

# Role-based access control dependency
def require_roles(required_roles: list):
    """
    Factory function that creates a dependency to check user roles.
    
    Args:
        required_roles: List of roles allowed to access the endpoint
        
    Returns:
        A dependency function that verifies the user has the required role
    """
    async def role_checker(request: Request):
        user = await get_current_user(request)
        user_role = user.get("role")
        if user_role not in required_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access forbidden. Required roles: {required_roles}"
            )
        return user
    return role_checker