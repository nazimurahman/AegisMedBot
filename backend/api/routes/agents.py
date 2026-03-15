"""
Agents routes module for managing and monitoring AI agents.
This module provides endpoints for agent status, metrics, and configuration.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
import logging
import psutil
import uuid

from ...core.config import settings
from ...core.security import get_current_user, require_role
from ...services.audit_service import AuditService
from ...models.schemas.response import ResponseModel

# Import agent orchestrator and base agent
from agents.orchestrator.agent_orchestrator import AgentOrchestrator
from agents.base_agent import AgentStatus

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agents", tags=["Agents"])

# Pydantic models for request/response validation

class AgentInfo(BaseModel):
    """
    Model for agent information response.
    """
    
    name: str = Field(..., description="Agent name")
    role: str = Field(..., description="Agent role description")
    status: str = Field(..., description="Current agent status")
    current_task: Optional[str] = Field(None, description="ID of current task if processing")
    uptime_seconds: float = Field(..., description="Agent uptime in seconds")
    tasks_completed: int = Field(..., description="Number of tasks completed")
    average_response_time_ms: float = Field(..., description="Average response time")
    error_rate: float = Field(..., description="Error rate (0-1)")
    last_active: datetime = Field(..., description="Last activity timestamp")

class AgentMetrics(BaseModel):
    """
    Model for detailed agent metrics.
    """
    
    agent_name: str = Field(..., description="Agent name")
    timestamp: datetime = Field(..., description="Metrics timestamp")
    
    # Performance metrics
    response_times: List[float] = Field(..., description="Recent response times")
    confidence_scores: List[float] = Field(..., description="Recent confidence scores")
    
    # Volume metrics
    requests_processed: int = Field(..., description="Total requests processed")
    tokens_processed: int = Field(..., description="Total tokens processed")
    
    # Quality metrics
    human_escalations: int = Field(..., description="Number of human escalations")
    user_feedback_avg: float = Field(..., description="Average user feedback score")
    
    # Resource metrics
    cpu_usage_percent: float = Field(..., description="CPU usage percentage")
    memory_usage_mb: float = Field(..., description="Memory usage in MB")
    
    # Error metrics
    errors_count: int = Field(..., description="Number of errors")
    timeout_count: int = Field(..., description="Number of timeouts")

class AgentConfiguration(BaseModel):
    """
    Model for agent configuration updates.
    """
    
    enabled: Optional[bool] = Field(None, description="Enable/disable agent")
    confidence_threshold: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold"
    )
    max_concurrent_tasks: Optional[int] = Field(
        None,
        ge=1,
        description="Maximum concurrent tasks"
    )
    timeout_seconds: Optional[int] = Field(
        None,
        ge=1,
        description="Task timeout in seconds"
    )
    model_params: Optional[Dict[str, Any]] = Field(
        None,
        description="Model parameters"
    )

# Dependency to get orchestrator
async def get_orchestrator() -> AgentOrchestrator:
    """
    Get or create agent orchestrator instance.
    """
    # This would typically be a singleton
    return AgentOrchestrator()

@router.get("/", response_model=List[AgentInfo])
async def list_agents(
    status: Optional[str] = Query(None, description="Filter by status"),
    orchestrator: AgentOrchestrator = Depends(get_orchestrator),
    current_user: Dict[str, Any] = Depends(require_role(["admin", "medical_director"]))
):
    """
    List all registered agents with their current status.
    Requires admin or medical director role.
    """
    
    logger.info(f"Listing agents for user {current_user['id']}")
    
    try:
        # Get all agents from orchestrator
        agents = orchestrator.list_agents()
        
        # Filter by status if provided
        if status:
            agents = [a for a in agents if a.status == status]
        
        # Build response
        agent_infos = []
        for agent in agents:
            metrics = orchestrator.get_agent_metrics(agent.name)
            
            agent_infos.append(AgentInfo(
                name=agent.name,
                role=agent.role,
                status=agent.status.value,
                current_task=agent.current_task,
                uptime_seconds=metrics.get("uptime_seconds", 0),
                tasks_completed=metrics.get("tasks_completed", 0),
                average_response_time_ms=metrics.get("avg_response_time_ms", 0),
                error_rate=metrics.get("error_rate", 0),
                last_active=metrics.get("last_active", datetime.now())
            ))
        
        return agent_infos
        
    except Exception as e:
        logger.error(f"Error listing agents: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving agents")

@router.get("/{agent_name}", response_model=AgentInfo)
async def get_agent_details(
    agent_name: str,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator),
    current_user: Dict[str, Any] = Depends(require_role(["admin", "medical_director"]))
):
    """
    Get detailed information about a specific agent.
    """
    
    logger.info(f"Getting details for agent {agent_name}")
    
    try:
        # Get agent instance
        agent = orchestrator.get_agent(agent_name)
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent {agent_name} not found")
        
        # Get metrics
        metrics = orchestrator.get_agent_metrics(agent_name)
        
        return AgentInfo(
            name=agent.name,
            role=agent.role,
            status=agent.status.value,
            current_task=agent.current_task,
            uptime_seconds=metrics.get("uptime_seconds", 0),
            tasks_completed=metrics.get("tasks_completed", 0),
            average_response_time_ms=metrics.get("avg_response_time_ms", 0),
            error_rate=metrics.get("error_rate", 0),
            last_active=metrics.get("last_active", datetime.now())
        )
        
    except HTTPException:
        raise
        
    except Exception as e:
        logger.error(f"Error getting agent details: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving agent details")

@router.get("/{agent_name}/metrics", response_model=AgentMetrics)
async def get_agent_metrics(
    agent_name: str,
    timeframe: str = Query("1h", description="Timeframe: 1h, 6h, 24h, 7d"),
    orchestrator: AgentOrchestrator = Depends(get_orchestrator),
    current_user: Dict[str, Any] = Depends(require_role(["admin"]))
):
    """
    Get detailed metrics for a specific agent.
    Requires admin role.
    """
    
    logger.info(f"Getting metrics for agent {agent_name}, timeframe={timeframe}")
    
    try:
        # Parse timeframe
        timeframe_map = {
            "1h": timedelta(hours=1),
            "6h": timedelta(hours=6),
            "24h": timedelta(hours=24),
            "7d": timedelta(days=7)
        }
        
        delta = timeframe_map.get(timeframe, timedelta(hours=1))
        since = datetime.now() - delta
        
        # Get metrics from orchestrator
        metrics = await orchestrator.get_agent_metrics_detailed(
            agent_name=agent_name,
            since=since
        )
        
        if not metrics:
            raise HTTPException(status_code=404, detail=f"No metrics found for agent {agent_name}")
        
        # Get system resource usage
        process = psutil.Process()
        cpu_percent = process.cpu_percent(interval=0.1)
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        
        return AgentMetrics(
            agent_name=agent_name,
            timestamp=datetime.now(),
            response_times=metrics.get("response_times", []),
            confidence_scores=metrics.get("confidence_scores", []),
            requests_processed=metrics.get("requests_processed", 0),
            tokens_processed=metrics.get("tokens_processed", 0),
            human_escalations=metrics.get("human_escalations", 0),
            user_feedback_avg=metrics.get("user_feedback_avg", 0.0),
            cpu_usage_percent=cpu_percent,
            memory_usage_mb=memory_mb,
            errors_count=metrics.get("errors_count", 0),
            timeout_count=metrics.get("timeout_count", 0)
        )
        
    except HTTPException:
        raise
        
    except Exception as e:
        logger.error(f"Error getting agent metrics: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving agent metrics")

@router.patch("/{agent_name}/config")
async def update_agent_config(
    agent_name: str,
    config: AgentConfiguration,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator),
    current_user: Dict[str, Any] = Depends(require_role(["admin"]))
):
    """
    Update configuration for a specific agent.
    Requires admin role.
    """
    
    logger.info(f"Updating config for agent {agent_name}")
    
    try:
        # Get agent
        agent = orchestrator.get_agent(agent_name)
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent {agent_name} not found")
        
        # Update configuration
        update_dict = config.dict(exclude_unset=True)
        success = await orchestrator.update_agent_config(agent_name, update_dict)
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to update agent configuration")
        
        # Log configuration change
        audit_service = AuditService()
        await audit_service.log_config_change(
            user_id=current_user["id"],
            agent_name=agent_name,
            changes=update_dict,
            timestamp=datetime.now()
        )
        
        return ResponseModel(
            status="success",
            message=f"Configuration updated for agent {agent_name}",
            data={"applied_changes": update_dict}
        )
        
    except HTTPException:
        raise
        
    except Exception as e:
        logger.error(f"Error updating agent config: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error updating agent configuration")

@router.post("/{agent_name}/reset")
async def reset_agent(
    agent_name: str,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator),
    current_user: Dict[str, Any] = Depends(require_role(["admin"]))
):
    """
    Reset an agent to its initial state.
    Requires admin role.
    """
    
    logger.warning(f"Resetting agent {agent_name} by user {current_user['id']}")
    
    try:
        # Get agent
        agent = orchestrator.get_agent(agent_name)
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent {agent_name} not found")
        
        # Reset agent
        success = await orchestrator.reset_agent(agent_name)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to reset agent")
        
        # Log reset action
        audit_service = AuditService()
        await audit_service.log_agent_reset(
            user_id=current_user["id"],
            agent_name=agent_name,
            timestamp=datetime.now()
        )
        
        return ResponseModel(
            status="success",
            message=f"Agent {agent_name} reset successfully"
        )
        
    except HTTPException:
        raise
        
    except Exception as e:
        logger.error(f"Error resetting agent: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error resetting agent")

@router.get("/system/health")
async def get_system_health(
    orchestrator: AgentOrchestrator = Depends(get_orchestrator),
    current_user: Dict[str, Any] = Depends(require_role(["admin", "medical_director"]))
):
    """
    Get overall system health status including all agents.
    """
    
    logger.info("Getting system health status")
    
    try:
        # Get all agents
        agents = orchestrator.list_agents()
        
        # Calculate aggregate metrics
        total_agents = len(agents)
        active_agents = sum(1 for a in agents if a.status == AgentStatus.PROCESSING)
        idle_agents = sum(1 for a in agents if a.status == AgentStatus.IDLE)
        error_agents = sum(1 for a in agents if a.status == AgentStatus.ERROR)
        waiting_agents = sum(1 for a in agents if a.status == AgentStatus.WAITING_FOR_HUMAN)
        
        # Get system resource usage
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Get database health (simplified)
        db_healthy = True  # Would check actual connection
        
        # Determine overall status
        if error_agents > 0:
            overall_status = "degraded"
        elif waiting_agents > 0:
            overall_status = "attention_needed"
        else:
            overall_status = "healthy"
        
        return {
            "status": overall_status,
            "timestamp": datetime.now().isoformat(),
            "agents": {
                "total": total_agents,
                "active": active_agents,
                "idle": idle_agents,
                "error": error_agents,
                "waiting_for_human": waiting_agents
            },
            "system": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": memory.available / (1024**3),
                "disk_percent": disk.percent,
                "disk_free_gb": disk.free / (1024**3)
            },
            "services": {
                "database": "healthy" if db_healthy else "unhealthy",
                "redis": "healthy",  # Would check actual connection
                "vector_db": "healthy"  # Would check actual connection
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting system health: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving system health")