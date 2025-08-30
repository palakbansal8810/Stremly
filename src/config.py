"""Configuration management for the compliance system."""
import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    # Model configurations
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    LLM_MODEL: str = "gemma2-9b-it"
    LLM_TEMPERATURE: float = 0.0
    
    # File paths
    EMBEDDINGS_INDEX_PATH: str = "embeddings/index_76780302ded643719a0fa7e65b559d66.faiss"
    EMBEDDINGS_DATA_PATH: str = "embeddings/data_76780302ded643719a0fa7e65b559d66.pkl"
    
    # Data directories
    POLICY_DIR: str = "data/policies"
    SPECS_DIR: str = "data/product_specs"
    SCREENSHOTS_DIR: str = "data/screenshots"
    CODE_DIR: str = "data/code"
    
    # Database
    MONGO_URL: str = "mongodb://localhost:27017/"
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    
    # Processing
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    MAX_RETRIEVAL_RESULTS: int = 10
    
    # Agent timeouts and retries
    AGENT_TIMEOUT: int = 60
    MAX_RETRIES: int = 3
    CIRCUIT_BREAKER_THRESHOLD: int = 3
    
    # HITL
    HITL_TIMEOUT: int = 300  # 5 minutes
    
    @classmethod
    def from_env(cls) -> 'Config':
        """Load configuration from environment variables."""
        return cls(
            MONGO_URL=os.getenv("MONGO_URL", cls.MONGO_URL),
            REDIS_HOST=os.getenv("REDIS_HOST", cls.REDIS_HOST),
            REDIS_PORT=int(os.getenv("REDIS_PORT", cls.REDIS_PORT)),
            # Add other env vars as needed
        )