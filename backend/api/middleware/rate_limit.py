"""
Rate limiting middleware for AegisMedBot.
Prevents API abuse by limiting the number of requests per user/IP.
Uses Redis for distributed rate limiting across multiple instances.
"""

from fastapi import Request, HTTPException, status
from typing import Optional, Dict, Tuple
import time
import logging
import asyncio
from datetime import datetime, timedelta
import redis.asyncio as redis
from ..core.config import settings

logger = logging.getLogger(__name__)

class RateLimitMiddleware:
    """
    Rate limiting middleware using the token bucket algorithm.
    Limits requests based on user ID or IP address.
    """
    
    def __init__(self, redis_client: redis.Redis):
        """
        Initialize the rate limiter with Redis connection.
        
        Args:
            redis_client: Async Redis client for distributed rate limiting
        """
        self.redis = redis_client
        # Default rate limits: requests per time window
        self.default_limits = {
            "anonymous": {"requests": 60, "window": 60},  # 60 requests per minute for anonymous users
            "authenticated": {"requests": 1000, "window": 60},  # 1000 requests per minute for authenticated users
            "premium": {"requests": 5000, "window": 60}  # 5000 requests per minute for premium users
        }
        
        # Store rate limit info in request state
        self.limit_headers = [
            "X-RateLimit-Limit",      # Maximum requests allowed in the window
            "X-RateLimit-Remaining",   # Remaining requests in the current window
            "X-RateLimit-Reset"        # Time when the window resets (Unix timestamp)
        ]
    
    async def __call__(self, request: Request) -> Optional[Dict]:
        """
        Process each request to check and update rate limits.
        
        Args:
            request: The incoming FastAPI request
            
        Returns:
            Rate limit information if checked
            
        Raises:
            HTTPException: If rate limit is exceeded
        """
        try:
            # Determine the client identifier (user ID or IP)
            client_id = await self._get_client_identifier(request)
            
            # Get rate limit tier based on user role
            tier = await self._get_user_tier(request)
            limits = self.default_limits.get(tier, self.default_limits["anonymous"])
            
            # Check rate limit
            is_allowed, rate_info = await self._check_rate_limit(
                client_id,
                limits["requests"],
                limits["window"]
            )
            
            # Add rate limit headers to response
            # These will be applied when the response is sent
            request.state.rate_limit_headers = {
                "X-RateLimit-Limit": str(limits["requests"]),
                "X-RateLimit-Remaining": str(rate_info["remaining"]),
                "X-RateLimit-Reset": str(rate_info["reset_time"])
            }
            
            if not is_allowed:
                # Rate limit exceeded
                logger.warning(f"Rate limit exceeded for client: {client_id}")
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail={
                        "error": "Rate limit exceeded",
                        "limit": limits["requests"],
                        "window_seconds": limits["window"],
                        "reset_time": rate_info["reset_time"]
                    },
                    headers=request.state.rate_limit_headers
                )
            
            return rate_info
            
        except HTTPException:
            # Re-raise HTTP exceptions
            raise
        except Exception as e:
            # If rate limiting fails, log but allow the request
            # This ensures the service remains available even if Redis is down
            logger.error(f"Rate limiting error: {str(e)}")
            return None
    
    async def _get_client_identifier(self, request: Request) -> str:
        """
        Get a unique identifier for the client.
        Uses user ID for authenticated users, IP for anonymous.
        
        Args:
            request: FastAPI request object
            
        Returns:
            String identifier for the client
        """
        # Check if user is authenticated
        user = getattr(request.state, "user", None)
        if user:
            # Use user ID for authenticated users
            return f"user:{user.get('sub')}"
        
        # For anonymous users, use IP address
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            # Get the client IP from proxy headers
            client_ip = forwarded_for.split(",")[0].strip()
        else:
            # Fall back to direct client IP
            client_ip = request.client.host if request.client else "unknown"
        
        return f"ip:{client_ip}"
    
    async def _get_user_tier(self, request: Request) -> str:
        """
        Determine the rate limit tier for the user.
        
        Args:
            request: FastAPI request object
            
        Returns:
            Tier name: "anonymous", "authenticated", or "premium"
        """
        user = getattr(request.state, "user", None)
        if not user:
            return "anonymous"
        
        # Check user tier from JWT claims or database
        # This could be based on subscription level, role, etc.
        role = user.get("role", "")
        if role == "premium" or role == "medical_director":
            return "premium"
        elif role:
            return "authenticated"
        else:
            return "anonymous"
    
    async def _check_rate_limit(
        self,
        client_id: str,
        max_requests: int,
        window_seconds: int
    ) -> Tuple[bool, Dict]:
        """
        Check if the client has exceeded their rate limit using Redis.
        Implements the sliding window algorithm for accurate rate limiting.
        
        Args:
            client_id: Unique identifier for the client
            max_requests: Maximum number of requests allowed in the window
            window_seconds: Time window in seconds
            
        Returns:
            Tuple of (is_allowed, rate_info dictionary)
        """
        # Create Redis key for this client
        # Format: rate_limit:{client_id}:{window_start}
        current_window = int(time.time() / window_seconds) * window_seconds
        key = f"rate_limit:{client_id}:{current_window}"
        
        # Use Redis pipeline for atomic operations
        pipe = self.redis.pipeline()
        
        # Increment the request count and set expiration
        pipe.incr(key)
        pipe.expire(key, window_seconds)
        
        # Execute pipeline
        results = await pipe.execute()
        current_count = results[0]  # Result of INCR
        
        # Calculate rate limit information
        remaining = max(0, max_requests - current_count)
        reset_time = current_window + window_seconds
        
        rate_info = {
            "current": current_count,
            "remaining": remaining,
            "limit": max_requests,
            "reset_time": reset_time,
            "window": window_seconds
        }
        
        # Check if request is allowed
        is_allowed = current_count <= max_requests
        
        return is_allowed, rate_info
    
    async def get_rate_limit_status(self, request: Request) -> Dict:
        """
        Get current rate limit status for the client.
        Useful for clients to check their limits before making requests.
        
        Args:
            request: FastAPI request object
            
        Returns:
            Dictionary with rate limit status
        """
        client_id = await self._get_client_identifier(request)
        tier = await self._get_user_tier(request)
        limits = self.default_limits.get(tier, self.default_limits["anonymous"])
        
        _, rate_info = await self._check_rate_limit(
            client_id,
            limits["requests"],
            limits["window"]
        )
        
        return {
            "tier": tier,
            "limit": rate_info["limit"],
            "remaining": rate_info["remaining"],
            "reset_time": rate_info["reset_time"],
            "window_seconds": rate_info["window"]
        }