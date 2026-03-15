"""
Context Manager Module for Conversation State Management

This module contains the ContextManager class which is responsible for maintaining
conversation context across multiple turns in a dialogue.

The context manager handles:
1. Creating and storing conversation contexts
2. Retrieving context by conversation ID
3. Updating context with new information
4. Managing conversation history
5. Extracting patient-specific information
6. Context expiration and cleanup

Proper context management is crucial for maintaining coherent multi-turn
conversations and providing personalized responses based on conversation history.
"""

import json
import logging
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

# Try to import Redis for production caching
# If not available, we'll fall back to in-memory storage
try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logging.warning("Redis not available, using in-memory context storage")

# Configure module-level logger
logger = logging.getLogger(__name__)

class ContextManager:
    """
    Manages conversation context across multiple turns in a dialogue.
    
    This class provides a centralized store for conversation state, allowing
    agents to access historical information and maintain coherent multi-turn
    conversations. It supports both in-memory storage (for development) and
    Redis-backed storage (for production scaling).
    
    Each conversation context contains:
    - Conversation metadata (ID, timestamps)
    - Message history
    - Patient context (if patient-specific)
    - Agent context (agent-specific state)
    - Custom metadata
    
    Attributes:
        redis_client: Optional Redis client for distributed caching
        context_ttl: Time-to-live for contexts in seconds
        max_history: Maximum number of messages to keep in history
        local_cache: Fallback in-memory cache when Redis is unavailable
    """
    
    def __init__(
        self,
        redis_client: Optional[Any] = None,
        context_ttl: int = 3600,
        max_history: int = 50
    ):
        """
        Initialize the ContextManager with optional Redis client.
        
        Args:
            redis_client: Optional Redis client for production deployments
            context_ttl: Time-to-live for contexts in seconds (default: 1 hour)
            max_history: Maximum number of messages to keep in history
        """
        self.redis_client = redis_client
        self.context_ttl = context_ttl
        self.max_history = max_history
        
        # Fallback in-memory cache for development or when Redis is unavailable
        self.local_cache: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"ContextManager initialized with TTL: {context_ttl}s, "
                   f"max_history: {max_history}")
        
        if redis_client:
            logger.info("Using Redis for context storage")
        else:
            logger.info("Using in-memory storage for context")
    
    def create_conversation(self, initial_metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new conversation and return its unique ID.
        
        This method initializes a new conversation context with default structure
        and optional initial metadata. The conversation ID is a UUID that uniquely
        identifies this conversation across the system.
        
        Args:
            initial_metadata: Optional dictionary of initial metadata to store
            
        Returns:
            Unique conversation ID string (UUID format)
        """
        # Generate a unique conversation ID using UUID4
        # UUID4 is randomly generated and has extremely low collision probability
        conversation_id = str(uuid.uuid4())
        
        # Create initial context structure
        context = {
            'conversation_id': conversation_id,
            'created_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
            'message_count': 0,
            'history': [],  # Will store message exchange history
            'metadata': initial_metadata or {},  # User-provided metadata
            'patient_context': {},  # Patient-specific information
            'agent_context': {},    # Agent-specific state
            'custom_data': {}       # Flexible storage for any additional data
        }
        
        # Store in appropriate storage backend
        self._store_context(conversation_id, context)
        
        logger.info(f"Created new conversation: {conversation_id}")
        
        return conversation_id
    
    def get_context(self, conversation_id: str) -> Dict[str, Any]:
        """
        Retrieve the full context for a conversation.
        
        This method fetches the complete conversation context including all
        history and metadata. If the conversation doesn't exist, it returns
        a minimal context with just the conversation ID.
        
        Args:
            conversation_id: The unique identifier of the conversation
            
        Returns:
            Dictionary containing the full conversation context, or a minimal
            context if the conversation doesn't exist
        """
        if not conversation_id:
            logger.warning("Attempted to get context with empty conversation_id")
            return {}
        
        # Try to retrieve from storage
        context = self._retrieve_context(conversation_id)
        
        # If not found, return minimal context
        if not context:
            logger.debug(f"Conversation {conversation_id} not found, returning minimal context")
            return {
                'conversation_id': conversation_id,
                'history': [],
                'message_count': 0
            }
        
        return context
    
    def update_context(
        self,
        conversation_id: str,
        updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Update conversation context with new information.
        
        This method handles various types of updates:
        - Adding new messages to history
        - Updating patient context
        - Updating agent context
        - Updating metadata
        - Setting custom data
        
        Args:
            conversation_id: The unique identifier of the conversation
            updates: Dictionary containing updates to apply. Special keys:
                    - 'history': Add a message to history
                    - 'patient_context': Merge into patient context
                    - 'agent_context': Merge into agent context
                    - Any other key will update the top-level context
            
        Returns:
            Updated context dictionary
        """
        # Retrieve current context
        context = self.get_context(conversation_id)
        
        # If context doesn't exist, create it
        if not context.get('created_at'):
            context = {
                'conversation_id': conversation_id,
                'created_at': datetime.now().isoformat(),
                'history': [],
                'message_count': 0,
                'metadata': {},
                'patient_context': {},
                'agent_context': {},
                'custom_data': {}
            }
        
        # Apply updates based on type
        for key, value in updates.items():
            if key == 'history' and isinstance(value, dict):
                # Add a new message to history
                # Each history entry should be a dict with at least 'content' and 'role'
                self._add_to_history(context, value)
                
            elif key == 'patient_context' and isinstance(value, dict):
                # Merge patient context (doesn't overwrite, just adds new keys)
                context['patient_context'].update(value)
                
            elif key == 'agent_context' and isinstance(value, dict):
                # Merge agent context
                context['agent_context'].update(value)
                
            elif key == 'metadata' and isinstance(value, dict):
                # Merge metadata
                context['metadata'].update(value)
                
            elif key == 'custom_data' and isinstance(value, dict):
                # Merge custom data
                context['custom_data'].update(value)
                
            else:
                # Direct top-level update (use with caution)
                context[key] = value
        
        # Update timestamp
        context['last_updated'] = datetime.now().isoformat()
        
        # Store updated context
        self._store_context(conversation_id, context)
        
        logger.debug(f"Updated context for conversation {conversation_id}")
        
        return context
    
    def _add_to_history(self, context: Dict[str, Any], message: Dict[str, Any]) -> None:
        """
        Add a message to conversation history with timestamp.
        
        This internal method handles adding messages to history while maintaining
        the maximum history limit.
        
        Args:
            context: The conversation context to update
            message: Message dictionary to add (should contain 'role' and 'content')
        """
        # Add timestamp if not present
        if 'timestamp' not in message:
            message['timestamp'] = datetime.now().isoformat()
        
        # Add to history
        context['history'].append(message)
        context['message_count'] += 1
        
        # Trim history if it exceeds maximum length
        if len(context['history']) > self.max_history:
            # Remove oldest messages
            excess = len(context['history']) - self.max_history
            context['history'] = context['history'][excess:]
            logger.debug(f"Trimmed {excess} old messages from history")
    
    def _store_context(self, conversation_id: str, context: Dict[str, Any]) -> None:
        """
        Store context in the appropriate backend.
        
        This internal method handles the actual storage operation, choosing
        between Redis and local cache based on availability.
        
        Args:
            conversation_id: The conversation identifier
            context: The context dictionary to store
        """
        try:
            if self.redis_client and REDIS_AVAILABLE:
                # Store in Redis with expiration
                self.redis_client.setex(
                    f"context:{conversation_id}",
                    self.context_ttl,
                    json.dumps(context, default=str)  # Convert non-serializable objects
                )
            else:
                # Store in local cache
                self.local_cache[conversation_id] = context
        except Exception as e:
            logger.error(f"Error storing context for {conversation_id}: {str(e)}")
            # Fall back to local cache
            self.local_cache[conversation_id] = context
    
    def _retrieve_context(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve context from the appropriate backend.
        
        This internal method handles fetching context from either Redis or
        local cache.
        
        Args:
            conversation_id: The conversation identifier
            
        Returns:
            Context dictionary if found, None otherwise
        """
        try:
            if self.redis_client and REDIS_AVAILABLE:
                # Try Redis first
                context_data = self.redis_client.get(f"context:{conversation_id}")
                if context_data:
                    return json.loads(context_data)
            
            # Fall back to local cache
            return self.local_cache.get(conversation_id)
            
        except Exception as e:
            logger.error(f"Error retrieving context for {conversation_id}: {str(e)}")
            # Fall back to local cache
            return self.local_cache.get(conversation_id)
    
    def clear_context(self, conversation_id: str) -> bool:
        """
        Clear a conversation's context from storage.
        
        Args:
            conversation_id: The conversation identifier to clear
            
        Returns:
            True if context was cleared, False if not found
        """
        try:
            if self.redis_client and REDIS_AVAILABLE:
                # Delete from Redis
                result = self.redis_client.delete(f"context:{conversation_id}")
                if result > 0:
                    logger.info(f"Cleared Redis context for {conversation_id}")
                    return True
            else:
                # Delete from local cache
                if conversation_id in self.local_cache:
                    del self.local_cache[conversation_id]
                    logger.info(f"Cleared local context for {conversation_id}")
                    return True
            
            logger.debug(f"No context found for {conversation_id}")
            return False
            
        except Exception as e:
            logger.error(f"Error clearing context for {conversation_id}: {str(e)}")
            return False
    
    def get_relevant_history(
        self,
        conversation_id: str,
        current_query: str,
        max_messages: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant conversation history for context window.
        
        This method returns the most recent messages from the conversation
        history, which can be used to provide context to agents.
        
        Args:
            conversation_id: The conversation identifier
            current_query: The current query (for future relevance filtering)
            max_messages: Maximum number of messages to return
            
        Returns:
            List of recent messages (oldest first)
        """
        context = self.get_context(conversation_id)
        history = context.get('history', [])
        
        if not history:
            return []
        
        # Return the most recent messages (keeping chronological order)
        return history[-max_messages:]
    
    def extract_patient_context(self, conversation_id: str) -> Dict[str, Any]:
        """
        Extract patient-specific information from conversation.
        
        This method retrieves the patient context that has been accumulated
        during the conversation.
        
        Args:
            conversation_id: The conversation identifier
            
        Returns:
            Dictionary containing patient-specific information
        """
        context = self.get_context(conversation_id)
        return context.get('patient_context', {})
    
    def set_patient_context(
        self,
        conversation_id: str,
        patient_data: Dict[str, Any]
    ) -> None:
        """
        Set or update patient context for a conversation.
        
        This is a convenience method for updating patient information.
        
        Args:
            conversation_id: The conversation identifier
            patient_data: Patient information to store
        """
        self.update_context(conversation_id, {
            'patient_context': patient_data
        })
    
    def get_agent_context(
        self,
        conversation_id: str,
        agent_name: str
    ) -> Dict[str, Any]:
        """
        Get context specific to a particular agent.
        
        Args:
            conversation_id: The conversation identifier
            agent_name: Name of the agent whose context to retrieve
            
        Returns:
            Agent-specific context dictionary
        """
        context = self.get_context(conversation_id)
        agent_context = context.get('agent_context', {})
        return agent_context.get(agent_name, {})
    
    def set_agent_context(
        self,
        conversation_id: str,
        agent_name: str,
        agent_data: Dict[str, Any]
    ) -> None:
        """
        Set context for a specific agent.
        
        Args:
            conversation_id: The conversation identifier
            agent_name: Name of the agent
            agent_data: Agent-specific data to store
        """
        context = self.get_context(conversation_id)
        if 'agent_context' not in context:
            context['agent_context'] = {}
        
        context['agent_context'][agent_name] = agent_data
        self._store_context(conversation_id, context)
    
    def cleanup_expired_contexts(self) -> int:
        """
        Clean up expired contexts (only applicable for in-memory storage).
        
        For Redis, expiration is handled automatically. For in-memory storage,
        this method removes contexts older than the TTL.
        
        Returns:
            Number of contexts cleaned up
        """
        if self.redis_client:
            # Redis handles expiration automatically
            return 0
        
        # For in-memory, manually clean up expired contexts
        cleanup_count = 0
        now = datetime.now()
        expired_ids = []
        
        for conv_id, context in self.local_cache.items():
            last_updated = context.get('last_updated')
            if last_updated:
                try:
                    updated_time = datetime.fromisoformat(last_updated)
                    if now - updated_time > timedelta(seconds=self.context_ttl):
                        expired_ids.append(conv_id)
                except (ValueError, TypeError):
                    # If timestamp is invalid, consider it expired
                    expired_ids.append(conv_id)
        
        for conv_id in expired_ids:
            del self.local_cache[conv_id]
            cleanup_count += 1
        
        if cleanup_count > 0:
            logger.info(f"Cleaned up {cleanup_count} expired contexts")
        
        return cleanup_count
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about stored contexts.
        
        Returns:
            Dictionary with context storage statistics
        """
        if self.redis_client:
            # For Redis, we can't easily get count without scanning
            return {
                'storage_type': 'redis',
                'ttl_seconds': self.context_ttl
            }
        else:
            return {
                'storage_type': 'memory',
                'active_contexts': len(self.local_cache),
                'ttl_seconds': self.context_ttl,
                'max_history': self.max_history
            }