"""
Cache management module for AegisMedBot backend.
Provides Redis-based caching, session management, and distributed locking.
"""

import json
import pickle
from typing import Any, Optional, Union, Dict, List
from datetime import timedelta, datetime
import asyncio
from functools import wraps
import hashlib
import logging

# Redis imports
import redis.asyncio as redis
from redis.exceptions import RedisError, ConnectionError

# Pydantic for data validation
from pydantic import BaseModel

# Configure logging for cache operations
logger = logging.getLogger(__name__)

class CacheConfig:
    """
    Configuration settings for the cache system.
    Centralizes all cache-related configuration parameters.
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        password: Optional[str] = None,
        db: int = 0,
        default_ttl: int = 3600,  # Default time-to-live in seconds (1 hour)
        max_connections: int = 10,
        socket_timeout: float = 5.0,
        socket_connect_timeout: float = 5.0,
        retry_on_timeout: bool = True,
        health_check_interval: int = 30
    ):
        """
        Initialize cache configuration.
        
        Args:
            host: Redis server hostname
            port: Redis server port
            password: Redis authentication password
            db: Redis database number
            default_ttl: Default expiration time in seconds
            max_connections: Maximum connection pool size
            socket_timeout: Socket operation timeout
            socket_connect_timeout: Connection timeout
            retry_on_timeout: Whether to retry on timeout
            health_check_interval: Health check interval in seconds
        """
        self.host = host
        self.port = port
        self.password = password
        self.db = db
        self.default_ttl = default_ttl
        self.max_connections = max_connections
        self.socket_timeout = socket_timeout
        self.socket_connect_timeout = socket_connect_timeout
        self.retry_on_timeout = retry_on_timeout
        self.health_check_interval = health_check_interval
    
    @property
    def connection_url(self) -> str:
        """
        Build Redis connection URL from configuration.
        
        Returns:
            Redis connection URL string
        """
        if self.password:
            return f"redis://:{self.password}@{self.host}:{self.port}/{self.db}"
        return f"redis://{self.host}:{self.port}/{self.db}"


class CacheManager:
    """
    Central cache manager handling all Redis operations.
    Provides high-level caching abstractions with error handling and serialization.
    """
    
    def __init__(self, config: Optional[CacheConfig] = None):
        """
        Initialize the cache manager with configuration.
        
        Args:
            config: CacheConfig instance, uses defaults if not provided
        """
        self.config = config or CacheConfig()
        self._redis: Optional[redis.Redis] = None
        self._connection_pool: Optional[redis.ConnectionPool] = None
        self._initialized = False
        
        # Cache statistics tracking
        self.stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
            "errors": 0
        }
    
    async def initialize(self) -> bool:
        """
        Initialize Redis connection pool.
        Must be called before using the cache.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Create connection pool
            self._connection_pool = redis.ConnectionPool(
                host=self.config.host,
                port=self.config.port,
                password=self.config.password,
                db=self.config.db,
                max_connections=self.config.max_connections,
                socket_timeout=self.config.socket_timeout,
                socket_connect_timeout=self.config.socket_connect_timeout,
                retry_on_timeout=self.config.retry_on_timeout,
                health_check_interval=self.config.health_check_interval
            )
            
            # Create Redis client with connection pool
            self._redis = redis.Redis(
                connection_pool=self._connection_pool,
                decode_responses=False  # Keep binary for flexibility
            )
            
            # Test connection with ping
            await self._redis.ping()
            self._initialized = True
            logger.info("Cache manager initialized successfully")
            return True
            
        except ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {str(e)}")
            self._initialized = False
            return False
        except Exception as e:
            logger.error(f"Unexpected error initializing cache: {str(e)}")
            self._initialized = False
            return False
    
    async def close(self):
        """
        Close Redis connections and cleanup.
        Should be called during application shutdown.
        """
        if self._redis:
            await self._redis.close()
        if self._connection_pool:
            await self._connection_pool.disconnect()
        self._initialized = False
        logger.info("Cache manager closed")
    
    def _ensure_initialized(self):
        """
        Check if cache is initialized before operations.
        Raises RuntimeError if not initialized.
        """
        if not self._initialized or not self._redis:
            raise RuntimeError("Cache manager not initialized. Call initialize() first.")
    
    def _serialize(self, value: Any) -> bytes:
        """
        Serialize Python objects to bytes for Redis storage.
        Handles different types appropriately.
        
        Args:
            value: Python object to serialize
            
        Returns:
            Bytes representation for Redis
        """
        if value is None:
            return b""
        elif isinstance(value, (str, int, float, bool)):
            # Simple types can be converted to string
            return str(value).encode("utf-8")
        elif isinstance(value, (dict, list, tuple)):
            # Complex types use JSON
            return json.dumps(value).encode("utf-8")
        elif isinstance(value, BaseModel):
            # Pydantic models have built-in JSON serialization
            return value.model_dump_json().encode("utf-8")
        elif hasattr(value, "__dict__"):
            # Custom objects use pickle (use with caution)
            return pickle.dumps(value)
        else:
            # Fallback to pickle
            return pickle.dumps(value)
    
    def _deserialize(self, data: bytes, as_type: Optional[type] = None) -> Any:
        """
        Deserialize bytes from Redis back to Python objects.
        
        Args:
            data: Bytes from Redis
            as_type: Expected return type for validation
            
        Returns:
            Deserialized Python object
        """
        if not data:
            return None
        
        try:
            # Try JSON first (most common)
            decoded = data.decode("utf-8")
            try:
                return json.loads(decoded)
            except json.JSONDecodeError:
                # Not JSON, try other formats
                pass
            
            # If as_type is provided and it's a Pydantic model
            if as_type and issubclass(as_type, BaseModel):
                return as_type.model_validate_json(decoded)
            
            # Return as string if it's simple text
            return decoded
            
        except UnicodeDecodeError:
            # Binary data, try pickle
            try:
                return pickle.loads(data)
            except:
                # Return raw bytes if all else fails
                return data
    
    async def get(
        self,
        key: str,
        as_type: Optional[type] = None,
        default: Any = None
    ) -> Any:
        """
        Retrieve a value from cache.
        
        Args:
            key: Cache key string
            as_type: Expected type for deserialization
            default: Default value if key not found
            
        Returns:
            Cached value or default
        """
        self._ensure_initialized()
        
        try:
            value = await self._redis.get(key)
            if value is None:
                self.stats["misses"] += 1
                return default
            
            self.stats["hits"] += 1
            return self._deserialize(value, as_type)
            
        except RedisError as e:
            logger.error(f"Redis error getting key {key}: {str(e)}")
            self.stats["errors"] += 1
            return default
        except Exception as e:
            logger.error(f"Unexpected error getting key {key}: {str(e)}")
            self.stats["errors"] += 1
            return default
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        nx: bool = False  # Only set if key does not exist
    ) -> bool:
        """
        Store a value in cache.
        
        Args:
            key: Cache key string
            value: Value to store
            ttl: Time-to-live in seconds (uses default if None)
            nx: If True, only set if key doesn't exist
            
        Returns:
            True if successful, False otherwise
        """
        self._ensure_initialized()
        
        try:
            serialized = self._serialize(value)
            ttl = ttl or self.config.default_ttl
            
            if nx:
                # SET with NX option (only if not exists)
                result = await self._redis.set(key, serialized, ex=ttl, nx=True)
            else:
                # Regular SET
                result = await self._redis.setex(key, ttl, serialized)
            
            if result:
                self.stats["sets"] += 1
                logger.debug(f"Cached key: {key} with TTL: {ttl}")
                return True
            return False
            
        except RedisError as e:
            logger.error(f"Redis error setting key {key}: {str(e)}")
            self.stats["errors"] += 1
            return False
        except Exception as e:
            logger.error(f"Unexpected error setting key {key}: {str(e)}")
            self.stats["errors"] += 1
            return False
    
    async def delete(self, *keys: str) -> int:
        """
        Delete one or more keys from cache.
        
        Args:
            *keys: Variable number of keys to delete
            
        Returns:
            Number of keys deleted
        """
        self._ensure_initialized()
        
        try:
            count = await self._redis.delete(*keys)
            self.stats["deletes"] += count
            if count > 0:
                logger.debug(f"Deleted {count} keys")
            return count
            
        except RedisError as e:
            logger.error(f"Redis error deleting keys: {str(e)}")
            self.stats["errors"] += 1
            return 0
    
    async def exists(self, key: str) -> bool:
        """
        Check if a key exists in cache.
        
        Args:
            key: Key to check
            
        Returns:
            True if key exists, False otherwise
        """
        self._ensure_initialized()
        
        try:
            return await self._redis.exists(key) > 0
        except RedisError as e:
            logger.error(f"Redis error checking key {key}: {str(e)}")
            self.stats["errors"] += 1
            return False
    
    async def expire(self, key: str, ttl: int) -> bool:
        """
        Set expiration on an existing key.
        
        Args:
            key: Key to set expiration on
            ttl: Time-to-live in seconds
            
        Returns:
            True if expiration set, False otherwise
        """
        self._ensure_initialized()
        
        try:
            return await self._redis.expire(key, ttl)
        except RedisError as e:
            logger.error(f"Redis error setting expire on {key}: {str(e)}")
            self.stats["errors"] += 1
            return False
    
    async def ttl(self, key: str) -> int:
        """
        Get remaining time-to-live for a key.
        
        Args:
            key: Key to check
            
        Returns:
            Remaining TTL in seconds, -1 if no expiry, -2 if key doesn't exist
        """
        self._ensure_initialized()
        
        try:
            return await self._redis.ttl(key)
        except RedisError as e:
            logger.error(f"Redis error getting TTL for {key}: {str(e)}")
            self.stats["errors"] += 1
            return -2
    
    async def incr(self, key: str, amount: int = 1) -> Optional[int]:
        """
        Increment a numeric value in cache.
        
        Args:
            key: Key to increment
            amount: Amount to increment by
            
        Returns:
            New value after increment, or None on error
        """
        self._ensure_initialized()
        
        try:
            return await self._redis.incrby(key, amount)
        except RedisError as e:
            logger.error(f"Redis error incrementing {key}: {str(e)}")
            self.stats["errors"] += 1
            return None
    
    async def get_many(self, keys: List[str], as_type: Optional[type] = None) -> Dict[str, Any]:
        """
        Retrieve multiple keys at once.
        
        Args:
            keys: List of keys to retrieve
            as_type: Expected type for deserialization
            
        Returns:
            Dictionary mapping keys to values (missing keys omitted)
        """
        self._ensure_initialized()
        
        try:
            # MGET returns values in same order as keys
            values = await self._redis.mget(keys)
            result = {}
            
            for key, value in zip(keys, values):
                if value is not None:
                    result[key] = self._deserialize(value, as_type)
                    self.stats["hits"] += 1
                else:
                    self.stats["misses"] += 1
            
            return result
            
        except RedisError as e:
            logger.error(f"Redis error getting multiple keys: {str(e)}")
            self.stats["errors"] += 1
            return {}
    
    async def set_many(self, mapping: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """
        Store multiple key-value pairs at once.
        
        Args:
            mapping: Dictionary of keys to values
            ttl: Time-to-live for all keys
            
        Returns:
            True if all sets successful, False otherwise
        """
        self._ensure_initialized()
        
        try:
            ttl = ttl or self.config.default_ttl
            pipe = self._redis.pipeline()
            
            for key, value in mapping.items():
                serialized = self._serialize(value)
                pipe.setex(key, ttl, serialized)
            
            # Execute pipeline
            results = await pipe.execute()
            
            # All commands should return True
            success = all(results)
            if success:
                self.stats["sets"] += len(mapping)
                logger.debug(f"Cached {len(mapping)} keys")
            
            return success
            
        except RedisError as e:
            logger.error(f"Redis error setting multiple keys: {str(e)}")
            self.stats["errors"] += 1
            return False
    
    async def clear_pattern(self, pattern: str) -> int:
        """
        Delete all keys matching a pattern.
        
        Args:
            pattern: Redis pattern (e.g., "session:*")
            
        Returns:
            Number of keys deleted
        """
        self._ensure_initialized()
        
        try:
            # Find all matching keys
            keys = []
            async for key in self._redis.scan_iter(match=pattern):
                keys.append(key)
            
            if keys:
                deleted = await self._redis.delete(*keys)
                self.stats["deletes"] += deleted
                logger.info(f"Cleared {deleted} keys matching pattern: {pattern}")
                return deleted
            
            return 0
            
        except RedisError as e:
            logger.error(f"Redis error clearing pattern {pattern}: {str(e)}")
            self.stats["errors"] += 1
            return 0
    
    async def get_stats(self) -> Dict[str, Union[int, float]]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache stats including hit rate
        """
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = (self.stats["hits"] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            **self.stats,
            "total_requests": total_requests,
            "hit_rate_percent": round(hit_rate, 2)
        }
    
    def reset_stats(self):
        """Reset cache statistics counters."""
        self.stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
            "errors": 0
        }
        logger.info("Cache statistics reset")


class SessionManager:
    """
    Manages user sessions using Redis.
    Provides session creation, validation, and cleanup.
    """
    
    def __init__(self, cache_manager: CacheManager, session_ttl: int = 3600):
        """
        Initialize session manager.
        
        Args:
            cache_manager: CacheManager instance
            session_ttl: Session time-to-live in seconds
        """
        self.cache = cache_manager
        self.session_ttl = session_ttl
        self.prefix = "session:"  # Prefix for all session keys
    
    def _session_key(self, session_id: str) -> str:
        """
        Build full session key with prefix.
        
        Args:
            session_id: Raw session ID
            
        Returns:
            Prefixed session key
        """
        return f"{self.prefix}{session_id}"
    
    async def create_session(self, user_id: str, data: Dict[str, Any]) -> Optional[str]:
        """
        Create a new session for a user.
        
        Args:
            user_id: User identifier
            data: Session data to store
            
        Returns:
            Session ID if successful, None otherwise
        """
        import uuid
        
        # Generate unique session ID
        session_id = str(uuid.uuid4())
        session_key = self._session_key(session_id)
        
        # Add metadata to session data
        session_data = {
            **data,
            "user_id": user_id,
            "created_at": datetime.now().isoformat(),
            "last_accessed": datetime.now().isoformat()
        }
        
        # Store in cache
        success = await self.cache.set(session_key, session_data, ttl=self.session_ttl)
        
        if success:
            logger.info(f"Created session {session_id} for user {user_id}")
            return session_id
        return None
    
    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve session data.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session data if valid, None otherwise
        """
        session_key = self._session_key(session_id)
        session_data = await self.cache.get(session_key)
        
        if session_data:
            # Update last accessed time
            session_data["last_accessed"] = datetime.now().isoformat()
            await self.cache.set(session_key, session_data, ttl=self.session_ttl)
            
            logger.debug(f"Retrieved session {session_id}")
        
        return session_data
    
    async def update_session(self, session_id: str, data: Dict[str, Any]) -> bool:
        """
        Update existing session data.
        
        Args:
            session_id: Session identifier
            data: New data to merge with existing
            
        Returns:
            True if update successful, False otherwise
        """
        session_key = self._session_key(session_id)
        current = await self.cache.get(session_key)
        
        if not current:
            logger.warning(f"Attempted to update non-existent session {session_id}")
            return False
        
        # Merge new data with existing
        current.update(data)
        current["last_accessed"] = datetime.now().isoformat()
        
        success = await self.cache.set(session_key, current, ttl=self.session_ttl)
        
        if success:
            logger.debug(f"Updated session {session_id}")
        
        return success
    
    async def delete_session(self, session_id: str) -> bool:
        """
        Delete a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if deleted, False otherwise
        """
        session_key = self._session_key(session_id)
        deleted = await self.cache.delete(session_key)
        
        if deleted:
            logger.info(f"Deleted session {session_id}")
            return True
        return False
    
    async def validate_session(self, session_id: str, user_id: str) -> bool:
        """
        Validate that a session belongs to a specific user.
        
        Args:
            session_id: Session identifier
            user_id: Expected user ID
            
        Returns:
            True if session is valid and belongs to user
        """
        session_data = await self.get_session(session_id)
        
        if not session_data:
            return False
        
        return session_data.get("user_id") == user_id
    
    async def cleanup_expired(self) -> int:
        """
        Clean up expired sessions.
        Redis automatically handles expiration, but this method
        can be used for additional cleanup logic if needed.
        
        Returns:
            Number of sessions cleaned (always 0 as Redis handles it)
        """
        # Redis automatically removes expired keys
        # This method exists for interface consistency
        return 0


class RateLimiter:
    """
    Rate limiting implementation using Redis.
    Prevents API abuse by limiting request frequency.
    """
    
    def __init__(self, cache_manager: CacheManager):
        """
        Initialize rate limiter.
        
        Args:
            cache_manager: CacheManager instance
        """
        self.cache = cache_manager
        self.prefix = "ratelimit:"
    
    def _key(self, identifier: str, action: str) -> str:
        """
        Build rate limit key.
        
        Args:
            identifier: User or IP identifier
            action: Action being rate limited
            
        Returns:
            Rate limit key
        """
        return f"{self.prefix}{action}:{identifier}"
    
    async def check_rate_limit(
        self,
        identifier: str,
        action: str,
        max_requests: int,
        window_seconds: int
    ) -> tuple[bool, Dict[str, Any]]:
        """
        Check if an action is rate limited.
        
        Args:
            identifier: User or IP identifier
            action: Action being rate limited
            max_requests: Maximum requests allowed in window
            window_seconds: Time window in seconds
            
        Returns:
            Tuple of (allowed, limit_info)
            limit_info contains current count, remaining, reset time
        """
        key = self._key(identifier, action)
        now = datetime.now()
        
        # Use Redis transaction to ensure atomicity
        pipe = self.cache._redis.pipeline()
        
        # Increment counter and set expiration
        pipe.incr(key)
        pipe.expire(key, window_seconds)
        
        try:
            results = await pipe.execute()
            current_count = results[0]
            
            # Calculate limit info
            remaining = max(0, max_requests - current_count)
            reset_at = now.timestamp() + window_seconds
            
            limit_info = {
                "limit": max_requests,
                "remaining": remaining,
                "reset_at": reset_at,
                "current": current_count
            }
            
            # Check if over limit
            allowed = current_count <= max_requests
            
            if not allowed:
                logger.warning(f"Rate limit exceeded for {identifier} on {action}")
            
            return allowed, limit_info
            
        except Exception as e:
            logger.error(f"Rate limit check failed: {str(e)}")
            # On error, allow the request but log the issue
            return True, {"limit": max_requests, "remaining": 0, "error": str(e)}


class DistributedLock:
    """
    Distributed lock implementation using Redis.
    Ensures mutual exclusion across multiple service instances.
    """
    
    def __init__(self, cache_manager: CacheManager, lock_ttl: int = 30):
        """
        Initialize distributed lock.
        
        Args:
            cache_manager: CacheManager instance
            lock_ttl: Lock time-to-live in seconds
        """
        self.cache = cache_manager
        self.lock_ttl = lock_ttl
        self.prefix = "lock:"
    
    def _key(self, name: str) -> str:
        """
        Build lock key.
        
        Args:
            name: Lock name
            
        Returns:
            Lock key
        """
        return f"{self.prefix}{name}"
    
    async def acquire(self, name: str, owner_id: str, ttl: Optional[int] = None) -> bool:
        """
        Acquire a distributed lock.
        
        Args:
            name: Lock name
            owner_id: Unique identifier for lock owner
            ttl: Lock TTL in seconds (uses default if None)
            
        Returns:
            True if lock acquired, False otherwise
        """
        key = self._key(name)
        ttl = ttl or self.lock_ttl
        
        # Use SET NX to atomically set if not exists
        lock_value = json.dumps({
            "owner": owner_id,
            "acquired_at": datetime.now().isoformat()
        })
        
        success = await self.cache.set(key, lock_value, ttl=ttl, nx=True)
        
        if success:
            logger.info(f"Lock acquired: {name} by {owner_id}")
        
        return success
    
    async def release(self, name: str, owner_id: str) -> bool:
        """
        Release a distributed lock.
        Only releases if called by the lock owner.
        
        Args:
            name: Lock name
            owner_id: Owner identifier for verification
            
        Returns:
            True if lock released, False otherwise
        """
        key = self._key(name)
        
        # Get current lock data
        lock_data = await self.cache.get(key)
        
        if not lock_data:
            # Lock doesn't exist
            return True
        
        # Verify ownership
        if isinstance(lock_data, dict) and lock_data.get("owner") == owner_id:
            deleted = await self.cache.delete(key)
            if deleted:
                logger.info(f"Lock released: {name} by {owner_id}")
                return True
        
        logger.warning(f"Failed to release lock {name} - not owner or invalid")
        return False
    
    async def is_locked(self, name: str) -> bool:
        """
        Check if a lock exists.
        
        Args:
            name: Lock name
            
        Returns:
            True if locked, False otherwise
        """
        key = self._key(name)
        return await self.cache.exists(key)
    
    async def get_owner(self, name: str) -> Optional[str]:
        """
        Get the current lock owner.
        
        Args:
            name: Lock name
            
        Returns:
            Owner ID if locked, None otherwise
        """
        key = self._key(name)
        lock_data = await self.cache.get(key)
        
        if lock_data and isinstance(lock_data, dict):
            return lock_data.get("owner")
        
        return None
    
    async def refresh(self, name: str, owner_id: str, ttl: Optional[int] = None) -> bool:
        """
        Refresh lock expiration.
        
        Args:
            name: Lock name
            owner_id: Owner identifier for verification
            ttl: New TTL in seconds
            
        Returns:
            True if refresh successful, False otherwise
        """
        key = self._key(name)
        ttl = ttl or self.lock_ttl
        
        # Verify ownership
        lock_data = await self.cache.get(key)
        
        if not lock_data:
            return False
        
        if isinstance(lock_data, dict) and lock_data.get("owner") == owner_id:
            # Reset expiration
            return await self.cache.expire(key, ttl)
        
        return False


# Cache decorator for function result caching
def cached(
    ttl: int = 300,
    key_prefix: str = "cache",
    as_type: Optional[type] = None
):
    """
    Decorator to cache function results in Redis.
    
    Args:
        ttl: Time-to-live in seconds
        key_prefix: Prefix for cache key
        as_type: Expected return type for deserialization
    
    Returns:
        Decorated function with caching
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            # Generate cache key from function name and arguments
            key_parts = [key_prefix, func.__name__]
            
            # Add arguments to key
            for arg in args:
                key_parts.append(str(arg))
            for k, v in sorted(kwargs.items()):
                key_parts.append(f"{k}:{v}")
            
            cache_key = ":".join(key_parts)
            # Hash long keys to avoid Redis key length limits
            if len(cache_key) > 200:
                cache_key = f"{key_prefix}:{hashlib.md5(cache_key.encode()).hexdigest()}"
            
            # Get cache manager instance
            cache_manager = getattr(self, "_cache", None)
            if not cache_manager:
                # No cache available, execute function
                return await func(self, *args, **kwargs)
            
            # Try to get from cache
            cached_result = await cache_manager.get(cache_key, as_type=as_type)
            if cached_result is not None:
                logger.debug(f"Cache hit for {cache_key}")
                return cached_result
            
            # Execute function and cache result
            result = await func(self, *args, **kwargs)
            if result is not None:
                await cache_manager.set(cache_key, result, ttl=ttl)
                logger.debug(f"Cached result for {cache_key}")
            
            return result
        
        return wrapper
    
    return decorator


# Global cache manager instance
_cache_manager: Optional[CacheManager] = None

async def get_cache_manager() -> CacheManager:
    """
    Get or create global cache manager instance.
    Implements singleton pattern for cache manager.
    
    Returns:
        CacheManager instance
    """
    global _cache_manager
    
    if _cache_manager is None:
        # Load configuration from environment
        import os
        from dotenv import load_dotenv
        
        load_dotenv()
        
        config = CacheConfig(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            password=os.getenv("REDIS_PASSWORD", None),
            db=int(os.getenv("REDIS_DB", "0")),
            default_ttl=int(os.getenv("CACHE_TTL", "3600"))
        )
        
        _cache_manager = CacheManager(config)
        await _cache_manager.initialize()
    
    return _cache_manager


async def close_cache():
    """Close global cache manager on shutdown."""
    global _cache_manager
    if _cache_manager:
        await _cache_manager.close()
        _cache_manager = None