import pytest
import asyncio
import logging
from unittest.mock import Mock

# Configure logging for tests
logging.basicConfig(level=logging.INFO)

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def mock_llm():
    """Mock LLM for testing."""
    llm = Mock()
    llm.invoke.return_value = Mock(content='{"test": "response"}')
    return llm

# tests/test_utils.py
"""Test utility functions."""
import pytest
from src.utils import chunk_text_with_overlap, generate_session_id, Citation, OutputSchema

def test_chunk_text_with_overlap():
    """Test text chunking utility."""
    text = "A" * 1000
    chunks = chunk_text_with_overlap(text, chunk_size=300, overlap=50)
    
    assert len(chunks) > 1
    assert all(len(chunk) <= 300 for chunk in chunks)
    
    # Test overlap
    if len(chunks) > 1:
        assert chunks[0][-50:] == chunks[1][:50]

def test_generate_session_id():
    """Test session ID generation."""
    query = "test query"
    session_id1 = generate_session_id(query, "user1")
    session_id2 = generate_session_id(query, "user1")
    session_id3 = generate_session_id(query, "user2")
    
    # Same inputs should generate same ID (on same day)
    assert session_id1 == session_id2
    # Different users should generate different IDs
    assert session_id1 != session_id3

def test_output_schema_validation():
    """Test OutputSchema validation."""
    valid_output = OutputSchema(
        session_id="test",
        decision="compliant",
        confidence=0.85,
        risk_score=0.25,
        rationale="Test rationale",
        citations=[],
        open_questions=[],
        processing_time=1.5
    )
    
    assert valid_output.confidence == 0.85
    assert valid_output.risk_score == 0.25
    
    # Test validation bounds
    with pytest.raises(ValueError):
        OutputSchema(
            session_id="test",
            decision="compliant",
            confidence=1.5,  # Invalid: > 1.0
            risk_score=0.25,
            rationale="Test",
            citations=[],
            open_questions=[],
            processing_time=1.0
        )

# Example usage and testing script
if __name__ == "__main__":
    # Run tests
    pytest.main([
        "tests/",
        "-v",
        "--tb=short",
        "--durations=10"
    ])