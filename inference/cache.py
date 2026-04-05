"""
Inference caching system for AegisMedBot to optimize performance.

This module provides intelligent caching mechanisms to avoid redundant
inference computations, significantly improving response times for
frequent or repetitive queries in the hospital environment.
"""

import hashlib
import json
import time
import pickle
from typing import Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import OrderedDict
import logging
import asyncio
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class CacheStrategy(Enum):
    """
    Caching strategies for inference results.
    
    Different strategies optimize for different use cases:
    - LRU: Least Recently Used - best for general purpose
    - LFU: Least Frequently Used - best for frequently accessed data
    - FIFO: First In First Out - simple time-based expiration
    - TTL: Time To Live - expiration based on time
    """
    LRU = "lru"      # Least Recently Used eviction policy
    LFU = "lfu"      # Least Frequently Used eviction policy
    FIFO = "fifo"    # First In First Out eviction policy
    TTL = "ttl"      # Time-based expiration only

@dataclass
class CacheEntry:
    """
    Individual cache entry with metadata.
    
    This dataclass stores not just the cached value but also
    metadata for cache management policies.
    
    Attributes:
        key: Cache key string
        value: Cached value (any picklable object)
        created_at: Timestamp when entry was created
        last_accessed: Timestamp of last access
        access_count: Number of times this entry has been accessed
        size_bytes: Estimated size of cached value in bytes
        ttl_seconds: Time to live in seconds (None for no expiration)
    """
    key: str
    value: Any
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    size_bytes: int = 0
    ttl_seconds: Optional[int] = None
    
    def __post_init__(self):
        """Calculate size estimate after initialization."""
        try:
            # Estimate size using pickle
            self.size_bytes = len(pickle.dumps(self.value))
        except:
            self.size_bytes = 1024  # Default size if estimation fails
    
    def is_expired(self) -> bool:
        """
        Check if cache entry has expired based on TTL.
        
        Returns:
            True if entry is expired, False otherwise
        """
        if self.ttl_seconds is None:
            return False
        return (time.time() - self.created_at) > self.ttl_seconds
    
    def update_access(self):
        """Update access metadata when entry is retrieved."""
        self.last_accessed = time.time()
        self.access_count += 1

class CacheKeyGenerator:
    """
    Generates consistent cache keys from inputs.
    
    This class ensures that identical inputs produce identical cache keys,
    while considering different hashing strategies for different input types.
    """
    
    @staticmethod
    def generate_key(
        model_name: str,
        inputs: Any,
        **kwargs
    ) -> str:
        """
        Generate a deterministic cache key.
        
        Args:
            model_name: Name of the model
            inputs: Input data
            **kwargs: Additional parameters affecting inference
            
        Returns:
            Hash string to use as cache key
        """
        # Create a dictionary of all components that affect the result
        key_components = {
            'model_name': model_name,
            'inputs_hash': CacheKeyGenerator._hash_input(inputs),
            'kwargs_hash': CacheKeyGenerator._hash_kwargs(kwargs)
        }
        
        # Convert to JSON and hash
        key_string = json.dumps(key_components, sort_keys=True)
        return hashlib.sha256(key_string.encode()).hexdigest()
    
    @staticmethod
    def _hash_input(inputs: Any) -> str:
        """
        Create a hash of the input data.
        
        Args:
            inputs: Input data in any format
            
        Returns:
            Hash string
        """
        try:
            # Try to convert to tensor first
            if hasattr(inputs, 'cpu'):
                # PyTorch tensor
                if inputs.is_cuda:
                    inputs = inputs.cpu()
                input_list = inputs.numpy().tolist()
                input_string = json.dumps(input_list, sort_keys=True)
            elif isinstance(inputs, (list, dict, tuple)):
                # Standard Python collections
                input_string = json.dumps(inputs, sort_keys=True)
            else:
                # Fallback to string representation
                input_string = str(inputs)
            
            return hashlib.md5(input_string.encode()).hexdigest()
            
        except Exception as e:
            logger.warning(f"Could not hash input: {e}, using fallback")
            # Fallback to simple hash of string representation
            return hashlib.md5(str(inputs).encode()).hexdigest()
    
    @staticmethod
    def _hash_kwargs(kwargs: Dict[str, Any]) -> str:
        """
        Create a hash of keyword arguments.
        
        Args:
            kwargs: Keyword arguments dictionary
            
        Returns:
            Hash string
        """
        # Filter out non-deterministic parameters
        filtered_kwargs = {k: v for k, v in kwargs.items() 
                          if k not in ['seed', 'temperature', 'timestamp']}
        
        try:
            return hashlib.md5(json.dumps(filtered_kwargs, sort_keys=True).encode()).hexdigest()
        except:
            return hashlib.md5(str(filtered_kwargs).encode()).hexdigest()

class InferenceCache:
    """
    Intelligent caching system for inference results.
    
    This class implements multiple caching strategies to optimize
    inference performance by storing and reusing previous results.
    
    Features:
    - Multiple eviction policies (LRU, LFU, FIFO, TTL)
    - Memory-aware caching with size limits
    - Automatic expiration based on TTL
    - Thread-safe operations with async support
    - Performance metrics and monitoring
    """
    
    def __init__(
        self,
        max_size_mb: int = 1024,  # 1GB default
        strategy: CacheStrategy = CacheStrategy.LRU,
        default_ttl_seconds: Optional[int] = 3600,  # 1 hour
        enable_metrics: bool = True
    ):
        """
        Initialize the inference cache.
        
        Args:
            max_size_mb: Maximum cache size in megabytes
            strategy: Cache eviction strategy
            default_ttl_seconds: Default time-to-live for entries
            enable_metrics: Whether to collect performance metrics
        """
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.strategy = strategy
        self.default_ttl = default_ttl_seconds
        self.enable_metrics = enable_metrics
        
        # Cache storage
        self._cache: Dict[str, CacheEntry] = {}
        
        # For LRU: maintain order
        self._lru_order = OrderedDict()
        
        # For LFU: maintain frequency counts
        self._lfu_frequencies: Dict[str, int] = {}
        
        # Metrics
        self._metrics = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'total_size_bytes': 0,
            'entry_count': 0
        }
        
        # Lock for thread safety
        self._lock = asyncio.Lock()
        
        logger.info(f"Cache initialized: max_size={max_size_mb}MB, strategy={strategy.value}")
    
    async def get(
        self,
        model_name: str,
        inputs: Any,
        **kwargs
    ) -> Optional[Any]:
        """
        Retrieve a value from cache.
        
        Args:
            model_name: Model identifier
            inputs: Input data
            **kwargs: Additional parameters affecting inference
            
        Returns:
            Cached value if found and not expired, None otherwise
        """
        # Generate cache key
        key = CacheKeyGenerator.generate_key(model_name, inputs, **kwargs)
        
        async with self._lock:
            # Check if key exists
            if key not in self._cache:
                self._record_miss()
                return None
            
            entry = self._cache[key]
            
            # Check expiration
            if entry.is_expired():
                await self._evict_entry(key)
                self._record_miss()
                return None
            
            # Update access metadata
            entry.update_access()
            
            # Update strategy-specific tracking
            if self.strategy == CacheStrategy.LRU:
                # Move to end (most recently used)
                self._lru_order.move_to_end(key)
            elif self.strategy == CacheStrategy.LFU:
                # Increment frequency
                self._lfu_frequencies[key] = self._lfu_frequencies.get(key, 0) + 1
            
            self._record_hit()
            
            logger.debug(f"Cache hit for key: {key[:8]}...")
            return entry.value
    
    async def put(
        self,
        model_name: str,
        inputs: Any,
        value: Any,
        ttl_seconds: Optional[int] = None,
        **kwargs
    ) -> bool:
        """
        Store a value in cache.
        
        Args:
            model_name: Model identifier
            inputs: Input data
            value: Value to cache
            ttl_seconds: Time to live for this entry
            **kwargs: Additional parameters affecting inference
            
        Returns:
            True if stored successfully, False otherwise
        """
        # Generate cache key
        key = CacheKeyGenerator.generate_key(model_name, inputs, **kwargs)
        
        async with self._lock:
            # Check if key already exists
            if key in self._cache:
                # Update existing entry
                entry = self._cache[key]
                entry.value = value
                entry.created_at = time.time()
                entry.ttl_seconds = ttl_seconds or self.default_ttl
                entry.update_access()
                
                # Recalculate size
                try:
                    entry.size_bytes = len(pickle.dumps(value))
                except:
                    entry.size_bytes = 1024
                
                logger.debug(f"Updated cache entry: {key[:8]}...")
                return True
            
            # Create new entry
            entry = CacheEntry(
                key=key,
                value=value,
                ttl_seconds=ttl_seconds or self.default_ttl
            )
            
            # Check if we need to evict
            await self._ensure_space(entry.size_bytes)
            
            # Store the entry
            self._cache[key] = entry
            self._metrics['total_size_bytes'] += entry.size_bytes
            self._metrics['entry_count'] += 1
            
            # Update strategy-specific structures
            if self.strategy == CacheStrategy.LRU:
                self._lru_order[key] = None
            elif self.strategy == CacheStrategy.LFU:
                self._lfu_frequencies[key] = 1
            
            logger.debug(f"Stored cache entry: {key[:8]}..., size={entry.size_bytes} bytes")
            return True
    
    async def _ensure_space(self, required_bytes: int):
        """
        Ensure enough space in cache by evicting entries if necessary.
        
        Args:
            required_bytes: Bytes needed for new entry
        """
        while (self._metrics['total_size_bytes'] + required_bytes) > self.max_size_bytes:
            if not self._cache:
                break
            
            # Select entry to evict based on strategy
            evict_key = self._select_eviction_candidate()
            
            if evict_key:
                await self._evict_entry(evict_key)
                self._metrics['evictions'] += 1
                logger.debug(f"Evicted entry to free space: {evict_key[:8]}...")
            else:
                break
    
    def _select_eviction_candidate(self) -> Optional[str]:
        """
        Select an entry to evict based on current strategy.
        
        Returns:
            Key of entry to evict, or None if no entry can be evicted
        """
        if not self._cache:
            return None
        
        if self.strategy == CacheStrategy.LRU:
            # Evict least recently used
            if self._lru_order:
                return next(iter(self._lru_order))
        
        elif self.strategy == CacheStrategy.LFU:
            # Evict least frequently used
            if self._lfu_frequencies:
                return min(self._lfu_frequencies.items(), key=lambda x: x[1])[0]
        
        elif self.strategy == CacheStrategy.FIFO:
            # Evict oldest (by creation time)
            return min(self._cache.items(), key=lambda x: x[1].created_at)[0]
        
        elif self.strategy == CacheStrategy.TTL:
            # Evict entry with earliest expiration
            candidates = [(k, v) for k, v in self._cache.items() if v.ttl_seconds is not None]
            if candidates:
                return min(candidates, key=lambda x: x[1].created_at + x[1].ttl_seconds)[0]
            # Fallback to LRU if no TTL entries
            if self._lru_order:
                return next(iter(self._lru_order))
        
        # Default: evict first entry
        return next(iter(self._cache.keys()))
    
    async def _evict_entry(self, key: str):
        """
        Evict a specific entry from cache.
        
        Args:
            key: Cache key to evict
        """
        if key in self._cache:
            entry = self._cache.pop(key)
            self._metrics['total_size_bytes'] -= entry.size_bytes
            self._metrics['entry_count'] -= 1
            
            # Clean up strategy-specific structures
            if self.strategy == CacheStrategy.LRU and key in self._lru_order:
                del self._lru_order[key]
            elif self.strategy == CacheStrategy.LFU and key in self._lfu_frequencies:
                del self._lfu_frequencies[key]
    
    async def clear(self, model_name: Optional[str] = None):
        """
        Clear cache entries.
        
        Args:
            model_name: Optional model name to clear entries for
        """
        async with self._lock:
            if model_name:
                # Clear only entries for specific model
                keys_to_remove = [
                    key for key, entry in self._cache.items()
                    if key.startswith(f"{model_name}_")
                ]
                for key in keys_to_remove:
                    await self._evict_entry(key)
                logger.info(f"Cleared {len(keys_to_remove)} entries for model '{model_name}'")
            else:
                # Clear everything
                self._cache.clear()
                self._lru_order.clear()
                self._lfu_frequencies.clear()
                self._metrics['total_size_bytes'] = 0
                self._metrics['entry_count'] = 0
                logger.info("Cleared entire cache")
    
    def _record_hit(self):
        """Record a cache hit."""
        if self.enable_metrics:
            self._metrics['hits'] += 1
    
    def _record_miss(self):
        """Record a cache miss."""
        if self.enable_metrics:
            self._metrics['misses'] += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get cache performance metrics.
        
        Returns:
            Dictionary with cache statistics
        """
        hit_rate = 0
        if (self._metrics['hits'] + self._metrics['misses']) > 0:
            hit_rate = self._metrics['hits'] / (self._metrics['hits'] + self._metrics['misses'])
        
        return {
            'hits': self._metrics['hits'],
            'misses': self._metrics['misses'],
            'hit_rate': hit_rate,
            'evictions': self._metrics['evictions'],
            'entry_count': self._metrics['entry_count'],
            'total_size_mb': self._metrics['total_size_bytes'] / (1024 * 1024),
            'max_size_mb': self.max_size_bytes / (1024 * 1024),
            'utilization_percent': (self._metrics['total_size_bytes'] / self.max_size_bytes) * 100,
            'strategy': self.strategy.value
        }
    
    async def cleanup_expired(self):
        """
        Remove all expired entries from cache.
        """
        async with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired()
            ]
            
            for key in expired_keys:
                await self._evict_entry(key)
            
            if expired_keys:
                logger.info(f"Cleaned up {len(expired_keys)} expired entries")
            
            return len(expired_keys)
    
    async def get_size(self) -> Dict[str, Any]:
        """
        Get detailed size information about cache.
        
        Returns:
            Dictionary with size statistics
        """
        async with self._lock:
            entries_by_model = {}
            for key, entry in self._cache.items():
                model_name = key.split('_')[0] if '_' in key else 'unknown'
                if model_name not in entries_by_model:
                    entries_by_model[model_name] = {
                        'count': 0,
                        'size_bytes': 0
                    }
                entries_by_model[model_name]['count'] += 1
                entries_by_model[model_name]['size_bytes'] += entry.size_bytes
            
            return {
                'total_entries': len(self._cache),
                'total_size_mb': self._metrics['total_size_bytes'] / (1024 * 1024),
                'entries_by_model': {
                    model: {
                        'count': data['count'],
                        'size_mb': data['size_bytes'] / (1024 * 1024)
                    }
                    for model, data in entries_by_model.items()
                }
            }