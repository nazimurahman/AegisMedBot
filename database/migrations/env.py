"""
Alembic Environment Configuration

This module sets up the Alembic migration environment for the AegisMedBot
database. It handles database connections, model metadata loading, and
environment-specific configurations.
"""

from logging.config import fileConfig
from sqlalchemy import engine_from_config
from sqlalchemy import pool
from alembic import context
import sys
import os

# Add the parent directory to Python path for imports
# This allows Alembic to find our model definitions
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our models so Alembic can detect them
# This is critical for autogenerate to work properly
from database.models.patient import Base
from database.config import get_database_url

# This is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Add your model's MetaData object here for 'autogenerate' support
# Target metadata is what Alembic compares against to detect changes
target_metadata = Base.metadata

def get_url():
    """
    Get database URL from environment or config
    
    This function prioritizes environment variables over the
    alembic.ini file for better security and flexibility.
    
    Returns:
        Database connection URL string
    """
    # Try to get URL from environment first
    database_url = os.getenv("DATABASE_URL")
    
    if database_url:
        return database_url
    
    # Fall back to alembic.ini configuration
    return config.get_main_option("sqlalchemy.url")

def run_migrations_offline() -> None:
    """
    Run migrations in 'offline' mode.
    
    This configures the context with just a URL and not an Engine,
    though an Engine is acceptable here as well. By skipping the Engine
    creation we don't even need a DBAPI to be available.
    
    Calls to context.execute() here emit the given string to the
    script output.
    
    Offline mode is useful for generating SQL scripts without
    connecting to a database.
    """
    url = get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        # Include schemas for PostgreSQL
        include_schemas=True,
        # Compare type for better change detection
        compare_type=True,
        # Compare server defaults to detect changes
        compare_server_default=True
    )

    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online() -> None:
    """
    Run migrations in 'online' mode.
    
    In this scenario we need to create an Engine and associate a
    connection with the context.
    
    Online mode directly applies migrations to the database.
    This is the normal mode for development and production.
    """
    # Get database configuration
    configuration = config.get_section(config.config_ini_section)
    configuration["sqlalchemy.url"] = get_url()
    
    # Create engine with connection pooling
    # Pool size and overflow control database connection limits
    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.QueuePool,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,  # Verify connections before using
        pool_recycle=3600    # Recycle connections after 1 hour
    )

    with connectable.connect() as connection:
        # Configure context with connection
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            # Include schemas for multi-schema databases
            include_schemas=True,
            # Detect column type changes
            compare_type=True,
            # Detect server default changes
            compare_server_default=True,
            # Include name for constraint changes
            include_name=True,
            # Transaction per migration for safety
            transactional_ddl=True
        )

        with context.begin_transaction():
            context.run_migrations()

# Determine which mode to run based on environment
# Offline mode is used for SQL generation
# Online mode is used for direct database migration
if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()