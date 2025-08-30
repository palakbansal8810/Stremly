from pydantic import BaseModel, Field, validator
from typing import List, Literal, Optional, Dict, Any
from datetime import datetime
import hashlib
import json
import logging

logger = logging.getLogger(__name__)

class Citation(BaseModel):
    doc_id: str
    chunk_id: str
    snippet: str = Field(..., max_length=200)
    relevance_score: float = Field(ge=0.0, le=1.0)

class HumanInteraction(BaseModel):
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    type: Literal["clarification", "approval", "upload_request", "validation"]
    prompt: str
    response: Optional[str] = None
    status: Literal["pending", "approved", "denied", "provided", "timeout"] = "pending"
    timeout_seconds: int = 300

class AgentResult(BaseModel):
    agent_name: str
    execution_time: float
    status: Literal["success", "failure", "timeout"]
    output: Dict[str, Any]
    citations: List[Citation] = []
    error_message: Optional[str] = None

class OutputSchema(BaseModel):
    session_id: str
    decision: Literal["compliant", "non_compliant", "insufficient_evidence", "requires_human_review"]
    confidence: float = Field(ge=0.0, le=1.0)
    risk_score: float = Field(ge=0.0, le=1.0)
    rationale: str
    citations: List[Citation]
    open_questions: List[str]
    human_interactions: List[HumanInteraction] = []
    agent_results: List[AgentResult] = []
    processing_time: float
    
    @validator('confidence', 'risk_score')
    def validate_scores(cls, v):
        return round(v, 3)

def generate_session_id(query: str, user_id: Optional[str] = None) -> str:
    """Generate a deterministic session ID."""
    content = f"{query}_{user_id}_{datetime.now().strftime('%Y%m%d')}"
    return hashlib.md5(content.encode()).hexdigest()

def chunk_text_with_overlap(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Improved text chunking with overlap."""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        chunks.append(chunk)
        
        if end >= len(text):
            break
        start = end - overlap
    
    return chunks