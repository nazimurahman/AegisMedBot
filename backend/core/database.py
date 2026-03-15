"""
Database configuration and session management for AegisMedBot.
Sets up async SQLAlchemy with PostgreSQL for production use.
Handles connection pooling, migrations, and session lifecycle.
"""

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    AsyncEngine,
    create_async_engine,
    async_sessionmaker
)
from sqlalchemy.orm import declarative_base, declared_attr
from sqlalchemy import MetaData
from typing import AsyncGenerator, Optional
from contextlib import asynccontextmanager
import logging
from .config import settings

logger = logging.getLogger(__name__)

# Create base class for models with automatic tablename generation
class CustomBase:
    """
    Custom base class for SQLAlchemy models.
    Provides automatic table name generation from class name.
    """
    @declared_attr
    def __tablename__(cls):
        """Generate table name from class name (convert CamelCase to snake_case)."""
        import re
        # Convert CamelCase to snake_case
        name = re.sub(r'(?<!^)(?=[A-Z])', '_', cls.__name__).lower()
        return name

# Create declarative base with custom base class
Base = declarative_base(cls=CustomBase)

# Create metadata instance
metadata = MetaData()

class DatabaseManager:
    """
    Manages database connections, sessions, and lifecycle.
    Implements connection pooling and async session management.
    """
    
    def __init__(self):
        """Initialize database manager with configuration from settings."""
        self.database_url = str(settings.DATABASE_URL)
        self.pool_size = settings.DATABASE_POOL_SIZE
        self.max_overflow = settings.DATABASE_MAX_OVERFLOW
        self.pool_timeout = settings.DATABASE_POOL_TIMEOUT
        self.echo = settings.DATABASE_ECHO
        
        self._engine: Optional[AsyncEngine] = None
        self._async_session_maker: Optional[async_sessionmaker[AsyncSession]] = None
    
    async def initialize(self):
        """
        Initialize database engine and session maker.
        Must be called before using the database.
        """
        logger.info("Initializing database connection...")
        
        # Create async engine with connection pooling
        self._engine = create_async_engine(
            self.database_url,
            echo=self.echo,
            pool_size=self.pool_size,
            max_overflow=self.max_overflow,
            pool_timeout=self.pool_timeout,
            pool_pre_ping=True,  # Verify connections before using
            pool_recycle=3600,   # Recycle connections after 1 hour
        )
        
        # Create async session maker
        self._async_session_maker = async_sessionmaker(
            self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False
        )
        
        logger.info("Database initialized successfully")
    
    async def close(self):
        """Close database connections and cleanup."""
        if self._engine:
            logger.info("Closing database connections...")
            await self._engine.dispose()
            logger.info("Database connections closed")
    
    def get_session_maker(self) -> async_sessionmaker[AsyncSession]:
        """Get the async session maker."""
        if not self._async_session_maker:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        return self._async_session_maker
    
    async def create_all(self):
        """
        Create all database tables.
        WARNING: Only use in development, not in production.
        """
        if settings.ENVIRONMENT == "production":
            logger.warning("Attempted to create tables in production - use migrations instead")
            return
        
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables created")
    
    async def drop_all(self):
        """
        Drop all database tables.
        WARNING: Only use in development, not in production.
        """
        if settings.ENVIRONMENT == "production":
            logger.warning("Attempted to drop tables in production - aborting")
            return
        
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        logger.warning("Database tables dropped")

# Create global database manager
db_manager = DatabaseManager()

# Session dependency for FastAPI
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that provides a database session.
    Handles session lifecycle and ensures proper cleanup.
    
    Yields:
        Async SQLAlchemy session
    
    Example:
        @app.get("/items")
        async def get_items(db: AsyncSession = Depends(get_db)):
            result = await db.execute(select(Item))
            return result.scalars().all()
    """
    session_maker = db_manager.get_session_maker()
    async with session_maker() as session:
        try:
            yield session
        finally:
            await session.close()

# Context manager for manual session handling
@asynccontextmanager
async def get_db_context() -> AsyncGenerator[AsyncSession, None]:
    """
    Context manager for manual database session handling.
    Useful for background tasks and scripts.
    
    Example:
        async with get_db_context() as db:
            result = await db.execute(select(User))
    """
    session_maker = db_manager.get_session_maker()
    async with session_maker() as session:
        try:
            yield session
        finally:
            await session.close()

# Database health check
async def check_database_health() -> bool:
    """
    Check if database is accessible and healthy.
    
    Returns:
        True if database is healthy, False otherwise
    """
    try:
        async with get_db_context() as db:
            # Execute simple query to check connection
            await db.execute("SELECT 1")
        return True
    except Exception as e:
        logger.error(f"Database health check failed: {str(e)}")
        return False

# Initialize database on startup
async def init_db():
    """Initialize database connection on application startup."""
    await db_manager.initialize()

# Close database on shutdown
async def close_db():
    """Close database connection on application shutdown."""
    await db_manager.close()

# Transaction helpers
class TransactionManager:
    """
    Context manager for database transactions.
    Ensures proper commit/rollback handling.
    """
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def __aenter__(self):
        """Enter transaction context."""
        return self.session
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit transaction context - commit on success, rollback on error."""
        if exc_type:
            # An exception occurred, rollback
            await self.session.rollback()
            logger.error(f"Transaction rolled back due to: {exc_type.__name__}: {exc_val}")
        else:
            # No exception, commit
            await self.session.commit()

async def transaction(session: AsyncSession) -> AsyncGenerator[AsyncSession, None]:
    """
    Context manager for database transactions.
    
    Args:
        session: The database session
    
    Yields:
        The session with transaction management
    
    Example:
        async with transaction(db) as session:
            session.add(new_user)
            await session.flush()
    """
    async with TransactionManager(session):
        yield session

# Retry logic for database operations
async def run_with_retry(
    operation,
    max_retries: int = 3,
    retry_delay: float = 1.0
):
    """
    Run a database operation with retry logic.
    
    Args:
        operation: Async function to run
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds
        
    Returns:
        Result of the operation
        
    Raises:
        Exception: If all retries fail
    """
    import asyncio
    
    last_error = None
    for attempt in range(max_retries):
        try:
            return await operation()
        except Exception as e:
            last_error = e
            logger.warning(f"Database operation failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay * (attempt + 1))  # Exponential backoff
    
    logger.error(f"Database operation failed after {max_retries} attempts")
    raise last_error