"""
Central configuration management for AegisMedBot.
Uses Pydantic settings management with environment variable support.
All configuration is validated at startup to ensure proper setup.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseSettings, PostgresDsn, validator, Field
from functools import lru_cache
import os
from dotenv import load_dotenv
import secrets

# Load environment variables from .env file
load_dotenv()

class Settings(BaseSettings):
    """
    Application settings with environment variable support.
    All settings are loaded from environment variables with defaults.
    Settings are validated and typed for safety.
    """
    
    # API Settings
    API_V1_PREFIX: str = "/api/v1"
    PROJECT_NAME: str = "AegisMedBot - Agentic Hospital Intelligence Platform"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    ENVIRONMENT: str = "production"
    
    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 4
    
    # Security Settings
    # Generate a secure random key if not provided (for development only)
    SECRET_KEY: str = Field(
        default_factory=lambda: secrets.token_urlsafe(32),
        env="SECRET_KEY"
    )
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30  # 30 minutes
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7  # 7 days
    TOKEN_URL: str = "/api/v1/auth/token"
    
    # CORS Settings
    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:7860",
        "https://app.aegismedbot.com"
    ]
    
    # Database Settings
    POSTGRES_SERVER: str = Field("localhost", env="POSTGRES_SERVER")
    POSTGRES_USER: str = Field("postgres", env="POSTGRES_USER")
    POSTGRES_PASSWORD: str = Field("postgres", env="POSTGRES_PASSWORD")
    POSTGRES_DB: str = Field("aegismedbot", env="POSTGRES_DB")
    POSTGRES_PORT: str = Field("5432", env="POSTGRES_PORT")
    DATABASE_URL: Optional[PostgresDsn] = None
    DATABASE_POOL_SIZE: int = 20
    DATABASE_MAX_OVERFLOW: int = 10
    DATABASE_POOL_TIMEOUT: int = 30
    DATABASE_ECHO: bool = False  # Set to True to log SQL queries
    
    @validator("DATABASE_URL", pre=True)
    def assemble_db_connection(cls, v: Optional[str], values: Dict[str, Any]) -> str:
        """
        Build database connection URL from individual components.
        This allows using either a full DATABASE_URL or individual parameters.
        """
        if isinstance(v, str):
            return v
        
        # Build PostgreSQL connection string
        return PostgresDsn.build(
            scheme="postgresql+asyncpg",  # Use asyncpg for async support
            user=values.get("POSTGRES_USER"),
            password=values.get("POSTGRES_PASSWORD"),
            host=values.get("POSTGRES_SERVER"),
            port=values.get("POSTGRES_PORT"),
            path=f"/{values.get('POSTGRES_DB') or ''}",
        )
    
    # Redis Settings
    REDIS_HOST: str = Field("localhost", env="REDIS_HOST")
    REDIS_PORT: int = Field(6379, env="REDIS_PORT")
    REDIS_PASSWORD: Optional[str] = Field(None, env="REDIS_PASSWORD")
    REDIS_DB: int = 0
    REDIS_URL: Optional[str] = None
    REDIS_MAX_CONNECTIONS: int = 10
    REDIS_TIMEOUT: int = 5
    
    @validator("REDIS_URL", pre=True)
    def assemble_redis_connection(cls, v: Optional[str], values: Dict[str, Any]) -> str:
        """
        Build Redis connection URL from individual components.
        """
        if isinstance(v, str):
            return v
        
        # Build Redis URL
        password = values.get("REDIS_PASSWORD")
        if password:
            return f"redis://:{password}@{values.get('REDIS_HOST')}:{values.get('REDIS_PORT')}/{values.get('REDIS_DB')}"
        else:
            return f"redis://{values.get('REDIS_HOST')}:{values.get('REDIS_PORT')}/{values.get('REDIS_DB')}"
    
    # Qdrant Vector Database Settings
    QDRANT_HOST: str = Field("localhost", env="QDRANT_HOST")
    QDRANT_PORT: int = Field(6333, env="QDRANT_PORT")
    QDRANT_COLLECTION: str = "medical_knowledge"
    QDRANT_API_KEY: Optional[str] = Field(None, env="QDRANT_API_KEY")
    QDRANT_PREFER_GRPC: bool = True
    QDRANT_TIMEOUT: int = 10
    EMBEDDING_DIM: int = 768  # Dimension for PubMedBERT embeddings
    
    # Model Settings
    TRANSFORMER_MODEL: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    LSTM_HIDDEN_SIZE: int = 256
    LSTM_NUM_LAYERS: int = 2
    MAX_SEQUENCE_LENGTH: int = 512
    BATCH_SIZE: int = 32
    MODEL_CACHE_DIR: str = "/app/models/cache"
    USE_GPU: bool = True
    QUANTIZE_MODELS: bool = True  # Use quantization for faster inference
    
    # Agent Settings
    MAX_AGENT_ITERATIONS: int = 5  # Maximum number of agent interactions per request
    AGENT_TIMEOUT_SECONDS: int = 30  # Timeout for agent processing
    HUMAN_IN_LOOP_THRESHOLD: float = 0.7  # Confidence threshold for human review
    ENABLE_AGENT_METRICS: bool = True
    
    # RAG Settings
    TOP_K_RETRIEVAL: int = 5  # Number of documents to retrieve
    CHUNK_SIZE: int = 1000  # Size of document chunks
    CHUNK_OVERLAP: int = 200  # Overlap between chunks
    SIMILARITY_THRESHOLD: float = 0.75  # Minimum similarity score
    USE_HYBRID_SEARCH: bool = True  # Combine vector and keyword search
    RERANK_RESULTS: bool = True  # Rerank retrieved documents
    
    # Rate Limiting
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_REQUESTS: int = 1000  # Default requests per window
    RATE_LIMIT_PERIOD: int = 60  # Window in seconds
    RATE_LIMIT_BY_USER: bool = True
    RATE_LIMIT_BY_IP: bool = True
    
    # Monitoring
    ENABLE_METRICS: bool = True
    SENTRY_DSN: Optional[str] = Field(None, env="SENTRY_DSN")
    ENABLE_TRACING: bool = False
    JAEGER_HOST: Optional[str] = Field(None, env="JAEGER_HOST")
    JAEGER_PORT: int = 6831
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"  # json or text
    ENABLE_AUDIT_LOG: bool = True
    AUDIT_LOG_RETENTION_DAYS: int = 90  # HIPAA requires 6 years, but we rotate
    
    # File Upload
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS: List[str] = [".pdf", ".txt", ".csv", ".json"]
    UPLOAD_DIR: str = "/app/uploads"
    
    # Email Settings
    SMTP_HOST: Optional[str] = Field(None, env="SMTP_HOST")
    SMTP_PORT: int = 587
    SMTP_USER: Optional[str] = Field(None, env="SMTP_USER")
    SMTP_PASSWORD: Optional[str] = Field(None, env="SMTP_PASSWORD")
    SMTP_TLS: bool = True
    FROM_EMAIL: Optional[str] = Field(None, env="FROM_EMAIL")
    
    # Security
    PASSWORD_MIN_LENGTH: int = 8
    PASSWORD_REQUIRE_UPPERCASE: bool = True
    PASSWORD_REQUIRE_LOWERCASE: bool = True
    PASSWORD_REQUIRE_NUMBERS: bool = True
    PASSWORD_REQUIRE_SPECIAL: bool = True
    MAX_LOGIN_ATTEMPTS: int = 5
    LOCKOUT_TIME_MINUTES: int = 30
    SESSION_TIMEOUT_MINUTES: int = 60
    PHI_DETECTION_ENABLED: bool = True  # Detect and redact PHI
    
    # Feature Flags
    FEATURE_CLINICAL_AGENT: bool = True
    FEATURE_RISK_PREDICTION: bool = True
    FEATURE_OPERATIONS_AGENT: bool = True
    FEATURE_DIRECTOR_AGENT: bool = True
    FEATURE_RESEARCH_AGENT: bool = True
    FEATURE_COMPLIANCE_AGENT: bool = True
    FEATURE_MULTIMODAL: bool = False
    FEATURE_REAL_TIME_UPDATES: bool = True
    
    class Config:
        """
        Pydantic configuration for settings.
        """
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        validate_assignment = True  # Validate when setting attributes
        
    @validator("BACKEND_CORS_ORIGINS", pre=True)
    def assemble_cors_origins(cls, v: str | List[str]) -> List[str]:
        """
        Parse CORS origins from environment variable.
        Can be comma-separated string or JSON list.
        """
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(f"Invalid CORS origins: {v}")
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.ENVIRONMENT.lower() == "production"
    
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.ENVIRONMENT.lower() == "development"
    
    def is_testing(self) -> bool:
        """Check if running in testing environment."""
        return self.ENVIRONMENT.lower() == "testing"

@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    Uses LRU cache to avoid reloading settings on every request.
    """
    return Settings()

# Create global settings instance
settings = get_settings()