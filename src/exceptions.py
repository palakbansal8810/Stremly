class ComplianceSystemError(Exception):
    """Base exception for compliance system."""
    pass

class IngestionError(ComplianceSystemError):
    """Raised when data ingestion fails."""
    pass

class EmbeddingError(ComplianceSystemError):
    """Raised when embedding operations fail."""
    pass

class RetrievalError(ComplianceSystemError):
    """Raised when retrieval fails."""
    pass

class AgentError(ComplianceSystemError):
    """Raised when agent execution fails."""
    pass

class HumanInteractionTimeoutError(ComplianceSystemError):
    """Raised when human interaction times out."""
    pass