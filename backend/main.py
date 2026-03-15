"""
Main application entry point for AegisMedBot backend.

This module initializes and configures the FastAPI application with all
routes, middleware, and dependencies. It handles startup/shutdown events,
database connections, and service initialization.
"""

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional
import logging
import time
from datetime import datetime

from .core.config import settings
from .core.database import engine, Base, get_db
from .core.cache import redis_client, get_redis
from .services.auth_service import AuthService
from .services.audit_service import AuditService
from .services.notification_service import NotificationService

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/app.log")
    ]
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    
    Handles:
    - Database connection initialization
    - Redis connection establishment
    - Service initialization
    - Graceful shutdown
    """
    # Startup
    logger.info("Starting AegisMedBot backend application...")
    
    # Initialize database
    try:
        async with engine.begin() as conn:
            # Create tables if they don't exist
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")
        # Continue anyway - tables might already exist
    
    # Initialize Redis
    try:
        await redis_client.ping()
        logger.info("Redis connection established")
    except Exception as e:
        logger.error(f"Redis connection failed: {str(e)}")
        # Continue with degraded functionality
    
    # Initialize services
    app.state.auth_service = AuthService(redis_client)
    app.state.audit_service = AuditService(redis_client)
    app.state.notification_service = NotificationService(redis_client)
    
    logger.info("Services initialized")
    logger.info(f"Application started in {settings.ENVIRONMENT} mode")
    
    yield
    
    # Shutdown
    logger.info("Shutting down AegisMedBot application...")
    
    # Close database connections
    await engine.dispose()
    logger.info("Database connections closed")
    
    # Close Redis connection
    await redis_client.close()
    logger.info("Redis connection closed")
    
    logger.info("Shutdown complete")


# Create FastAPI application
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="AegisMedBot - Agentic AI Hospital Intelligence Platform",
    lifespan=lifespan,
    docs_url="/api/docs" if settings.ENVIRONMENT != "production" else None,
    redoc_url="/api/redoc" if settings.ENVIRONMENT != "production" else None,
    openapi_url="/api/openapi.json" if settings.ENVIRONMENT != "production" else None
)


# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request, call_next):
    """
    Middleware to track request processing time.
    
    Adds X-Process-Time header to responses.
    """
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time * 1000)  # Convert to ms
    return response


# Request logging middleware
@app.middleware("http")
async def log_requests(request, call_next):
    """
    Middleware to log all incoming requests.
    """
    request_id = str(uuid4())
    
    # Log request
    logger.info(f"Request {request_id}: {request.method} {request.url.path}")
    
    # Process request
    try:
        response = await call_next(request)
        
        # Log response
        logger.info(
            f"Response {request_id}: {response.status_code} "
            f"({response.headers.get('X-Process-Time', '?')}ms)"
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Request {request_id} failed: {str(e)}")
        raise


# Root endpoint
@app.get("/")
async def root() -> Dict[str, Any]:
    """
    Root endpoint providing basic API information.
    
    Returns:
        API metadata and status
    """
    return {
        "name": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "status": "operational",
        "environment": settings.ENVIRONMENT,
        "timestamp": datetime.utcnow().isoformat(),
        "documentation": "/api/docs" if settings.ENVIRONMENT != "production" else None
    }


# Health check endpoint
@app.get("/health")
async def health_check(
    db = Depends(get_db),
    redis = Depends(get_redis)
) -> Dict[str, Any]:
    """
    Health check endpoint for monitoring systems.
    
    Checks connectivity to all dependent services.
    
    Returns:
        Health status of all services
    """
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {}
    }
    
    # Check database
    try:
        # Execute simple query to verify connection
        await db.execute("SELECT 1")
        health_status["services"]["database"] = {
            "status": "connected",
            "latency_ms": 0  # Would measure actual latency in production
        }
    except Exception as e:
        health_status["services"]["database"] = {
            "status": "disconnected",
            "error": str(e)
        }
        health_status["status"] = "degraded"
    
    # Check Redis
    try:
        await redis.ping()
        health_status["services"]["redis"] = {
            "status": "connected"
        }
    except Exception as e:
        health_status["services"]["redis"] = {
            "status": "disconnected",
            "error": str(e)
        }
        health_status["status"] = "degraded"
    
    return health_status


# Error handler for 500 errors
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """
    Custom handler for HTTP exceptions.
    
    Returns:
        JSON response with error details
    """
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": {
                "code": f"HTTP_{exc.status_code}",
                "message": exc.detail,
                "details": {
                    "path": request.url.path,
                    "method": request.method
                }
            },
            "metadata": {
                "timestamp": datetime.utcnow().isoformat(),
                "request_id": getattr(request.state, "request_id", None)
            }
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """
    Handler for unhandled exceptions.
    
    Returns:
        JSON response with error details
    """
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An internal server error occurred",
                "details": {
                    "path": request.url.path,
                    "method": request.method
                }
            },
            "metadata": {
                "timestamp": datetime.utcnow().isoformat(),
                "request_id": getattr(request.state, "request_id", None)
            }
        }
    )


# Import and include routers
from .api.routes import chat, agents, patients, admin

app.include_router(
    chat.router,
    prefix=f"{settings.API_V1_PREFIX}/chat",
    tags=["chat"]
)

app.include_router(
    agents.router,
    prefix=f"{settings.API_V1_PREFIX}/agents",
    tags=["agents"]
)

app.include_router(
    patients.router,
    prefix=f"{settings.API_V1_PREFIX}/patients",
    tags=["patients"]
)

app.include_router(
    admin.router,
    prefix=f"{settings.API_V1_PREFIX}/admin",
    tags=["admin"]
)


# Startup event (additional)
@app.on_event("startup")
async def startup_event():
    """
    Additional startup tasks.
    """
    logger.info("Running startup tasks...")
    
    # Log startup
    if hasattr(app.state, "audit_service"):
        await app.state.audit_service.log_event(
            action="SYSTEM_START",
            user_id="system",
            details={"version": settings.VERSION, "environment": settings.ENVIRONMENT}
        )


# Shutdown event (additional)
@app.on_event("shutdown")
async def shutdown_event():
    """
    Additional shutdown tasks.
    """
    logger.info("Running shutdown tasks...")


# Helper function for UUID generation
def uuid4():
    """Generate a random UUID."""
    import uuid
    return str(uuid.uuid4())


# Run with uvicorn if executed directly
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.ENVIRONMENT == "development",
        log_level="info"
    )