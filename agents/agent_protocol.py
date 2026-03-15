"""
Agent communication protocol definition for AegisMedBot.
Defines standardized communication patterns between agents for reliable
and predictable multi-agent interactions.
"""

from typing import Dict, Any, Optional, List, Callable, Awaitable
from enum import Enum
from pydantic import BaseModel, Field
from datetime import datetime
import asyncio
import json
import logging

logger = logging.getLogger(__name__)

class ProtocolVersion(str, Enum):
    """
    Version tracking for the agent communication protocol.
    Ensures backward compatibility as the system evolves.
    """
    V1_0 = "1.0"
    V1_1 = "1.1"

class MessagePriority(str, Enum):
    """
    Priority levels for message handling.
    Higher priority messages are processed first.
    """
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class MessageIntent(str, Enum):
    """
    Defines the intent of a message to help with routing and processing.
    """
    QUERY = "query"  # General information request
    COMMAND = "command"  # Execute an action
    NOTIFICATION = "notification"  # Inform about an event
    ERROR = "error"  # Error reporting
    ESCALATION = "escalation"  # Escalate to human
    FEEDBACK = "feedback"  # Provide feedback
    STATUS = "status"  # Status check
    SYNTHESIS = "synthesis"  # Combine multiple responses

class AgentCapability(BaseModel):
    """
    Describes an agent's capability for discovery and routing.
    """
    name: str = Field(..., description="Capability name")
    description: str = Field(..., description="Capability description")
    confidence_threshold: float = Field(0.7, description="Minimum confidence for this capability")
    required_tools: List[str] = Field(default_factory=list, description="Tools needed")
    estimated_latency_ms: int = Field(100, description="Estimated processing time")

class ProtocolMessage(BaseModel):
    """
    Extended message format with protocol-specific fields.
    Wraps the base AgentMessage with protocol metadata.
    """
    protocol_version: str = Field(ProtocolVersion.V1_0, description="Protocol version")
    priority: MessagePriority = Field(MessagePriority.MEDIUM, description="Message priority")
    intent: MessageIntent = Field(MessageIntent.QUERY, description="Message intent")
    ttl_seconds: int = Field(30, description="Time to live in seconds")
    requires_ack: bool = Field(True, description="Whether acknowledgment is required")
    max_retries: int = Field(3, description="Maximum retry attempts")
    
    class Config:
        use_enum_values = True

class AgentDiscovery:
    """
    Handles agent discovery and capability registration.
    Allows agents to discover each other's capabilities dynamically.
    """
    
    def __init__(self):
        self.agents: Dict[str, Dict[str, Any]] = {}
        self.capabilities: Dict[str, List[AgentCapability]] = {}
        self.discovery_lock = asyncio.Lock()
    
    async def register_agent(
        self,
        agent_name: str,
        agent_type: str,
        capabilities: List[AgentCapability],
        endpoint: Optional[str] = None
    ):
        """
        Register an agent with its capabilities.
        
        Args:
            agent_name: Unique agent identifier
            agent_type: Type/category of agent
            capabilities: List of agent capabilities
            endpoint: Optional communication endpoint
        """
        async with self.discovery_lock:
            self.agents[agent_name] = {
                "name": agent_name,
                "type": agent_type,
                "capabilities": capabilities,
                "endpoint": endpoint,
                "registered_at": datetime.now().isoformat(),
                "last_heartbeat": datetime.now().isoformat(),
                "status": "active"
            }
            
            self.capabilities[agent_name] = capabilities
            
            logger.info(f"Registered agent: {agent_name} with {len(capabilities)} capabilities")
    
    async def discover_agents(
        self,
        required_capability: str,
        min_confidence: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Discover agents that have a specific capability.
        
        Args:
            required_capability: Capability to look for
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of matching agents with their details
        """
        matching_agents = []
        
        for agent_name, capabilities in self.capabilities.items():
            for capability in capabilities:
                if capability.name == required_capability:
                    if capability.confidence_threshold >= min_confidence:
                        agent_info = self.agents.get(agent_name, {})
                        if agent_info.get("status") == "active":
                            matching_agents.append({
                                "agent_name": agent_name,
                                "capability": capability.dict(),
                                "agent_info": agent_info
                            })
        
        return matching_agents
    
    async def update_heartbeat(self, agent_name: str):
        """Update agent heartbeat timestamp."""
        if agent_name in self.agents:
            self.agents[agent_name]["last_heartbeat"] = datetime.now().isoformat()
    
    async def get_agent_status(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """Get current status of an agent."""
        return self.agents.get(agent_name)

class MessageRouter:
    """
    Routes messages between agents based on content and capabilities.
    Implements intelligent routing strategies.
    """
    
    def __init__(self, discovery: AgentDiscovery):
        self.discovery = discovery
        self.message_queues: Dict[str, asyncio.Queue] = {}
        self.routing_table: Dict[str, str] = {}  # Maps message types to agents
        self.router_lock = asyncio.Lock()
    
    async def route_message(
        self,
        message: ProtocolMessage,
        content: Dict[str, Any]
    ) -> Optional[str]:
        """
        Determine the best agent to route a message to.
        
        Args:
            message: Protocol message with intent
            content: Message content
            
        Returns:
            Target agent name or None if no suitable agent found
        """
        # Extract query for analysis
        query = content.get("query", "")
        
        # Determine required capability based on intent
        required_capability = self._map_intent_to_capability(message.intent, query)
        
        # Find agents with this capability
        matching_agents = await self.discovery.discover_agents(
            required_capability,
            min_confidence=0.6
        )
        
        if not matching_agents:
            logger.warning(f"No agents found for capability: {required_capability}")
            return None
        
        # Select best agent (simplified - could use more sophisticated selection)
        best_agent = matching_agents[0]["agent_name"]
        
        logger.info(f"Routed message to agent: {best_agent} for capability: {required_capability}")
        
        return best_agent
    
    def _map_intent_to_capability(self, intent: MessageIntent, query: str) -> str:
        """
        Map message intent and query to required capability.
        
        Args:
            intent: Message intent
            query: Query text
            
        Returns:
            Required capability name
        """
        # Intent-based mapping
        intent_map = {
            MessageIntent.QUERY: "information_retrieval",
            MessageIntent.COMMAND: "action_execution",
            MessageIntent.NOTIFICATION: "event_handling",
            MessageIntent.ERROR: "error_handling",
            MessageIntent.ESCALATION: "human_escalation",
            MessageIntent.FEEDBACK: "feedback_processing",
            MessageIntent.STATUS: "status_monitoring",
            MessageIntent.SYNTHESIS: "response_synthesis"
        }
        
        base_capability = intent_map.get(intent, "general_processing")
        
        # Enhance with query-based classification
        query_lower = query.lower()
        
        # Clinical queries
        if any(word in query_lower for word in ["diagnosis", "treatment", "medication", "disease"]):
            return "clinical_knowledge"
        
        # Risk-related queries
        if any(word in query_lower for word in ["risk", "predict", "probability", "chance"]):
            return "risk_prediction"
        
        # Operations queries
        if any(word in query_lower for word in ["bed", "occupancy", "flow", "resource"]):
            return "operations_management"
        
        # Director-level queries
        if any(word in query_lower for word in ["kpi", "performance", "report", "metric"]):
            return "strategic_intelligence"
        
        return base_capability
    
    async def create_message_queue(self, agent_name: str):
        """Create a message queue for an agent."""
        async with self.router_lock:
            if agent_name not in self.message_queues:
                self.message_queues[agent_name] = asyncio.Queue()
                logger.info(f"Created message queue for agent: {agent_name}")
    
    async def send_message(
        self,
        target_agent: str,
        message: ProtocolMessage,
        content: Dict[str, Any]
    ) -> bool:
        """
        Send a message to a specific agent.
        
        Args:
            target_agent: Target agent name
            message: Protocol message
            content: Message content
            
        Returns:
            True if message was queued successfully
        """
        if target_agent not in self.message_queues:
            await self.create_message_queue(target_agent)
        
        try:
            # Wrap content with protocol message
            full_message = {
                "protocol": message.dict(),
                "content": content,
                "timestamp": datetime.now().isoformat()
            }
            
            await self.message_queues[target_agent].put(full_message)
            
            logger.debug(f"Message sent to {target_agent}: {message.intent}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send message to {target_agent}: {str(e)}")
            return False
    
    async def receive_message(self, agent_name: str, timeout: Optional[float] = None) -> Optional[Dict]:
        """
        Receive a message for an agent.
        
        Args:
            agent_name: Agent name
            timeout: Optional timeout in seconds
            
        Returns:
            Message or None if timeout
        """
        if agent_name not in self.message_queues:
            return None
        
        try:
            if timeout:
                message = await asyncio.wait_for(
                    self.message_queues[agent_name].get(),
                    timeout=timeout
                )
            else:
                message = await self.message_queues[agent_name].get()
            
            return message
            
        except asyncio.TimeoutError:
            return None
        except Exception as e:
            logger.error(f"Error receiving message for {agent_name}: {str(e)}")
            return None

class AgentHandshake:
    """
    Handles agent-to-agent handshake for establishing communication.
    Ensures both agents are ready and compatible.
    """
    
    async def perform_handshake(
        self,
        agent_a: str,
        agent_b: str,
        timeout: int = 5
    ) -> bool:
        """
        Perform handshake between two agents.
        
        Args:
            agent_a: First agent name
            agent_b: Second agent name
            timeout: Handshake timeout in seconds
            
        Returns:
            True if handshake successful
        """
        try:
            logger.info(f"Initiating handshake between {agent_a} and {agent_b}")
            
            # Simulate handshake process
            handshake_id = str(uuid.uuid4())
            
            # In production, this would involve actual message exchange
            # For now, we'll simulate success
            
            logger.info(f"Handshake successful between {agent_a} and {agent_b}")
            return True
            
        except Exception as e:
            logger.error(f"Handshake failed between {agent_a} and {agent_b}: {str(e)}")
            return False
    
    async def verify_compatibility(
        self,
        agent_a_capabilities: List[AgentCapability],
        agent_b_capabilities: List[AgentCapability]
    ) -> bool:
        """
        Verify that two agents are compatible for communication.
        
        Args:
            agent_a_capabilities: Capabilities of first agent
            agent_b_capabilities: Capabilities of second agent
            
        Returns:
            True if agents are compatible
        """
        # Check if agents share any common capability domains
        a_domains = {cap.name.split("_")[0] for cap in agent_a_capabilities}
        b_domains = {cap.name.split("_")[0] for cap in agent_b_capabilities}
        
        return len(a_domains.intersection(b_domains)) > 0