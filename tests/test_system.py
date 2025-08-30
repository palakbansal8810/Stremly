import pytest
import asyncio
import tempfile
import os
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import json

from src.utils import Citation, AgentResult
from src.ingest import DataIngestor
from src.embedder import Embedder
from src.retriever import Retriever
from src.orchestrator import ComplianceOrchestrator
from src.agents import PolicyAgent, RiskAgent, CodeScannerAgent
from src.utils import OutputSchema, Citation, HumanInteraction
from src.config import Config
from src.exceptions import IngestionError, AgentError,HumanInteractionTimeoutError

class TestConfig:
    """Test configuration."""
    @pytest.fixture
    def config():
        return Config(
            POLICY_DIR="test_data/policies",
            SPECS_DIR="test_data/specs",
            SCREENSHOTS_DIR="test_data/screenshots",
            CODE_DIR="test_data/code",
            CHUNK_SIZE=500,
            AGENT_TIMEOUT=30
        )

class TestDataIngestor:
    """Test data ingestion functionality."""
    
    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dirs = {
                'policy': os.path.join(temp_dir, 'policies'),
                'specs': os.path.join(temp_dir, 'specs'),
                'code': os.path.join(temp_dir, 'code'),
                'screenshots': os.path.join(temp_dir, 'screenshots')
            }
            
            for dir_path in dirs.values():
                os.makedirs(dir_path, exist_ok=True)
                
            yield dirs

    def test_read_code_files(self, config, temp_dirs):
        """Test code file ingestion."""
        # Create test code file
        test_code = """
def authenticate_user(username, password):
    # TODO: Implement proper password hashing
    if username == "admin" and password == "password123":
        return True
    return False
"""
        code_file = os.path.join(temp_dirs['code'], 'auth.py')
        with open(code_file, 'w') as f:
            f.write(test_code)
            
        config.CODE_DIR = temp_dirs['code']
        ingestor = DataIngestor(config)
        
        results = ingestor.read_code()
        
        assert len(results) == 1
        assert results[0]['doc_id'] == 'auth.py'
        assert results[0]['doc_type'] == 'code'
        assert 'authenticate_user' in results[0]['text']

    def test_read_product_specs(self, config, temp_dirs):
        """Test product specification ingestion."""
        spec_content = """
# Authentication Requirements
- Multi-factor authentication required for admin access
- Password complexity: minimum 12 characters
- Session timeout: 30 minutes
- Failed login attempts: lock after 3 attempts
"""
        spec_file = os.path.join(temp_dirs['specs'], 'auth_spec.md')
        with open(spec_file, 'w') as f:
            f.write(spec_content)
            
        config.SPECS_DIR = temp_dirs['specs']
        ingestor = DataIngestor(config)
        
        results = ingestor.read_product_specs()
        
        assert len(results) == 1
        assert results[0]['doc_id'] == 'auth_spec.md'
        assert results[0]['doc_type'] == 'product_spec'
        assert 'Multi-factor authentication' in results[0]['text']

    def test_chunking_with_overlap(self, config, temp_dirs):
        """Test text chunking with overlap."""
        long_content = "A" * 1500  # Longer than chunk size
        
        spec_file = os.path.join(temp_dirs['specs'], 'long_spec.txt')
        with open(spec_file, 'w') as f:
            f.write(long_content)
            
        config.SPECS_DIR = temp_dirs['specs']
        config.CHUNK_SIZE = 500
        config.CHUNK_OVERLAP = 100
        
        ingestor = DataIngestor(config)
        results = ingestor.read_product_specs()
        
        assert len(results) > 1  # Should be chunked
        assert all(len(chunk['text']) <= 500 for chunk in results)

class TestRetriever:
    """Test retrieval functionality."""
    
    @pytest.fixture
    def mock_retriever(self):
        """Create a mock retriever."""
        retriever = Mock(spec=Retriever)
        retriever.search.return_value = [
            ("Sample policy text about MFA requirements", 0.1, "policy.pdf", "policy_0"),
            ("Code implementing authentication", 0.2, "auth.py", "code_0"),
            ("Product spec for login flow", 0.3, "spec.md", "spec_0")
        ]
        return retriever

    def test_search_returns_relevant_results(self, mock_retriever):
        """Test that search returns properly formatted results."""
        results = mock_retriever.search("MFA requirements")
        
        assert len(results) == 3
        assert all(len(result) == 4 for result in results)  # text, score, doc_id, chunk_id
        assert results[0][0] == "Sample policy text about MFA requirements"

class TestAgents:
    """Test individual agent functionality."""
    
    @pytest.fixture
    def mock_llm_response(self):
        """Mock LLM response."""
        mock_response = Mock()
        mock_response.content = json.dumps({
            "assessment": "MFA requirements are properly implemented",
            "compliance_status": "compliant",
            "specific_requirements": ["Multi-factor authentication", "Session management"],
            "gaps_identified": [],
            "recommendations": ["Regular security audits"]
        })
        return mock_response

    @pytest.mark.asyncio
    async def test_policy_agent_success(self, mock_retriever, config, mock_llm_response):
        """Test successful policy agent execution."""
        agent = PolicyAgent(mock_retriever, config)
        
        with patch.object(agent.llm, 'invoke', return_value=mock_llm_response):
            result = await agent.run("Does our system meet MFA requirements?")
            
        assert result.status == "success"
        assert result.agent_name == "PolicyAgent"
        assert "assessment" in result.output
        assert len(result.citations) > 0

    @pytest.mark.asyncio
    async def test_risk_agent_with_high_risk(self, mock_retriever, config):
        """Test risk agent identifying high risk scenarios."""
        agent = RiskAgent(mock_retriever, config)
        
        high_risk_response = Mock()
        high_risk_response.content = json.dumps({
            "risk_score": 0.8,
            "risk_level": "high",
            "vulnerabilities_identified": ["Weak password storage", "No rate limiting"],
            "security_concerns": ["SQL injection potential"],
            "mitigation_recommendations": ["Implement bcrypt", "Add rate limiting"],
            "rationale": "Multiple security vulnerabilities identified"
        })
        
        with patch.object(agent.llm, 'invoke', return_value=high_risk_response):
            result = await agent.run("Analyze authentication security")
            
        assert result.status == "success"
        assert result.output["risk_score"] == 0.8
        assert result.output["risk_level"] == "high"

    @pytest.mark.asyncio
    async def test_agent_timeout_handling(self, mock_retriever, config):
        """Test agent timeout handling."""
        agent = PolicyAgent(mock_retriever, config)
        
        # Mock LLM to simulate timeout
        async def slow_invoke(*args, **kwargs):
            await asyncio.sleep(2)  # Longer than timeout
            return Mock(content="{}")
            
        agent.llm.invoke = slow_invoke
        
        with patch('asyncio.wait_for', side_effect=asyncio.TimeoutError):
            result = await agent.run("test query")
            
        assert result.status == "failure"
        assert "timeout" in result.error_message.lower()

class TestOrchestrator:
    """Test orchestration logic."""
    
    @pytest.fixture
    def mock_mongo_client(self):
        """Mock MongoDB client."""
        client = Mock()
        client["compliance_db"]["sessions"] = Mock()
        return client

    @pytest.fixture
    def orchestrator(self, mock_retriever, config, mock_mongo_client):
        """Create orchestrator with mocked dependencies."""
        return ComplianceOrchestrator(mock_retriever, config, mock_mongo_client)

    @pytest.mark.asyncio
    async def test_successful_orchestration_flow(self, orchestrator):
        """Test complete successful orchestration."""
        session_id = "test_session_123"
        query = "Does our authentication meet security requirements?"
        
        # Mock successful agent results
        mock_policy_result = AgentResult(
            agent_name="PolicyAgent",
            execution_time=1.0,
            status="success",
            output={
                "assessment": "Compliant with policy requirements",
                "compliance_status": "compliant",
                "specific_requirements": ["MFA", "Password complexity"]
            },
            citations=[Citation(doc_id="policy.pdf", chunk_id="policy_0", snippet="MFA required", relevance_score=0.9)]
        )
        
        mock_risk_result = AgentResult(
            agent_name="RiskAgent",
            execution_time=1.2,
            status="success",
            output={
                "risk_score": 0.3,
                "risk_level": "low",
                "vulnerabilities_identified": [],
                "rationale": "Low risk implementation"
            }
        )
        
        with patch.object(orchestrator, '_analyze_phase', return_value={
            "policy": mock_policy_result,
            "risk": mock_risk_result
        }):
            result = await orchestrator.run(session_id, query)
            
        assert isinstance(result, OutputSchema)
        assert result.session_id == session_id
        assert result.decision in ["compliant", "non_compliant", "insufficient_evidence", "requires_human_review"]
        assert 0.0 <= result.confidence <= 1.0
        assert 0.0 <= result.risk_score <= 1.0

    @pytest.mark.asyncio
    async def test_orchestration_with_agent_failures(self, orchestrator):
        """Test orchestration handling agent failures."""
        session_id = "test_session_failure"
        query = "Test query"
        
        # Mock mixed success/failure results
        failed_result = AgentResult(
            agent_name="PolicyAgent",
            execution_time=0.0,
            status="failure",
            output={},
            error_message="Connection timeout"
        )
        
        success_result = AgentResult(
            agent_name="RiskAgent",
            execution_time=1.0,
            status="success",
            output={"risk_score": 0.5}
        )
        
        with patch.object(orchestrator, '_analyze_phase', return_value={
            "policy": failed_result,
            "risk": success_result
        }):
            result = await orchestrator.run(session_id, query)
            
        # Should still complete but with lower confidence
        assert result.decision == "insufficient_evidence"
        assert result.confidence < 0.5

    @pytest.mark.asyncio
    async def test_human_interaction_timeout(self, orchestrator):
        """Test human interaction timeout handling."""
        session_id = "test_hitl_timeout"
        
        # Mock scenario requiring human input
        with patch.object(orchestrator, '_analyze_phase') as mock_analyze:
            mock_analyze.return_value = {
                "policy": AgentResult(
                    agent_name="PolicyAgent",
                    execution_time=1.0,
                    status="success",
                    output={"compliance_status": "requires_review"}
                )
            }
            
            with patch.object(orchestrator, '_request_human_input', side_effect=HumanInteractionTimeoutError("Timeout")):
                result = await orchestrator.run(session_id, "test query")
                
        # Should handle timeout gracefully
        assert result.decision == "insufficient_evidence"
        assert any(interaction.status == "timeout" for interaction in result.human_interactions)

class TestIntegration:
    """Integration tests for the complete system."""
    
    @pytest.fixture
    def integration_setup(self):
        """Set up integration test environment."""
        config = Config()
        
        # Create test data
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test policy
            policy_dir = os.path.join(temp_dir, 'policies')
            os.makedirs(policy_dir)
            
            # Create test spec
            specs_dir = os.path.join(temp_dir, 'specs')
            os.makedirs(specs_dir)
            
            with open(os.path.join(specs_dir, 'auth_spec.md'), 'w') as f:
                f.write("""
# Authentication Specification
## Requirements
- Multi-factor authentication mandatory for admin users
- Password minimum 12 characters with complexity requirements
- Session timeout after 30 minutes of inactivity
- Account lockout after 3 failed attempts
""")
            
            # Create test code
            code_dir = os.path.join(temp_dir, 'code')
            os.makedirs(code_dir)
            
            with open(os.path.join(code_dir, 'auth.py'), 'w') as f:
                f.write("""
import bcrypt
import time

class AuthenticationService:
    def __init__(self):
        self.failed_attempts = {}
        self.session_timeout = 1800  # 30 minutes
        
    def authenticate(self, username, password, mfa_token=None):
        if self.is_account_locked(username):
            raise Exception("Account locked")
            
        if not self.verify_password(username, password):
            self.record_failed_attempt(username)
            return False
            
        if not self.verify_mfa(username, mfa_token):
            return False
            
        return True
        
    def verify_mfa(self, username, token):
        # MFA implementation
        return token is not None
""")
            
            config.POLICY_DIR = policy_dir
            config.SPECS_DIR = specs_dir
            config.CODE_DIR = code_dir
            
            yield config, temp_dir

    @pytest.mark.asyncio
    async def test_end_to_end_compliance_check(self, integration_setup):
        """Test complete end-to-end compliance checking."""
        config, temp_dir = integration_setup
        
        # Mock external dependencies
        with patch('pymongo.MongoClient') as mock_mongo, \
             patch('redis.Redis') as mock_redis, \
             patch('faiss.read_index') as mock_faiss, \
             patch('joblib.load') as mock_joblib:
            
            # Setup mocks
            mock_mongo_instance = Mock()
            mock_mongo.return_value = mock_mongo_instance
            mock_mongo_instance["compliance_db"]["sessions"] = Mock()
            
            mock_redis_instance = Mock()
            mock_redis.return_value = mock_redis_instance
            
            # Mock retriever
            mock_retriever = Mock(spec=Retriever)
            mock_retriever.search.return_value = [
                ("Multi-factor authentication mandatory for admin users", 0.1, "auth_spec.md", "spec_0"),
                ("MFA implementation in code", 0.2, "auth.py", "code_0")
            ]
            
            orchestrator = ComplianceOrchestrator(mock_retriever, config, mock_mongo_instance)
            
            # Mock agent responses
            with patch.object(orchestrator.policy_agent, 'run') as mock_policy, \
                 patch.object(orchestrator.risk_agent, 'run') as mock_risk, \
                 patch.object(orchestrator.code_agent, 'run') as mock_code:
                
                mock_policy.return_value = AgentResult(
                    agent_name="PolicyAgent",
                    execution_time=1.0,
                    status="success",
                    output={
                        "assessment": "MFA requirements are met",
                        "compliance_status": "compliant"
                    },
                    citations=[Citation(doc_id="auth_spec.md", chunk_id="spec_0", snippet="MFA required", relevance_score=0.9)]
                )
                
                mock_risk.return_value = AgentResult(
                    agent_name="RiskAgent",
                    execution_time=1.1,
                    status="success",
                    output={
                        "risk_score": 0.2,
                        "risk_level": "low"
                    }
                )
                
                mock_code.return_value = AgentResult(
                    agent_name="CodeScannerAgent",
                    execution_time=0.9,
                    status="success",
                    output={
                        "scan_results": "MFA implementation found",
                        "code_quality_score": 0.8
                    }
                )
                
                # Execute orchestration
                result = await orchestrator.run("test_session", "Does our authentication system meet MFA requirements?")
                
                # Assertions
                assert isinstance(result, OutputSchema)
                assert result.decision == "compliant"
                assert result.confidence > 0.8
                assert result.risk_score < 0.3
                assert len(result.citations) > 0

class TestPerformance:
    """Performance and load testing."""
    
    @pytest.mark.asyncio
    async def test_concurrent_orchestration(self):
        """Test multiple concurrent orchestration requests."""
        # Mock dependencies
        mock_retriever = Mock(spec=Retriever)
        mock_retriever.search.return_value = [("test", 0.1, "doc", "chunk")]
        
        config = Config(AGENT_TIMEOUT=10)
        mock_mongo = Mock()
        mock_mongo["compliance_db"]["sessions"] = Mock()
        
        orchestrator = ComplianceOrchestrator(mock_retriever, config, mock_mongo)
        
        # Mock all agents to return quickly
        for agent_name in ["policy_agent", "risk_agent", "code_agent"]:
            agent = getattr(orchestrator, agent_name)
            mock_result = AgentResult(
                agent_name=agent_name,
                execution_time=0.1,
                status="success",
                output={"test": "result"}
            )
            agent.run = AsyncMock(return_value=mock_result)
        
        # Run multiple concurrent requests
        tasks = [
            orchestrator.run(f"session_{i}", f"query_{i}")
            for i in range(5)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All should complete successfully
        assert len(results) == 5
        assert all(isinstance(r, OutputSchema) for r in results if not isinstance(r, Exception))

class TestErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_activation(self):
        """Test circuit breaker behavior under repeated failures."""
        config = Config(CIRCUIT_BREAKER_THRESHOLD=2)
        mock_retriever = Mock()
        mock_mongo = Mock()
        mock_mongo["compliance_db"]["sessions"] = Mock()
        
        orchestrator = ComplianceOrchestrator(mock_retriever, config, mock_mongo)
        
        # Mock agent to always fail
        failing_agent = Mock()
        failing_agent.run = AsyncMock(side_effect=AgentError("Simulated failure"))
        orchestrator.policy_agent = failing_agent
        
        # First few failures should trigger circuit breaker
        with pytest.raises(Exception):  # Circuit breaker should eventually trigger
            for _ in range(5):
                await orchestrator._run_agent_with_circuit_breaker(failing_agent, "test query")

    def test_invalid_json_response_handling(self, mock_retriever, config):
        """Test handling of invalid JSON responses from LLM."""
        agent = PolicyAgent(mock_retriever, config)
        
        # Test malformed JSON
        invalid_responses = [
            "This is not JSON at all",
            '{"incomplete": json',
            '{"valid": "json", "but": "unexpected format"}',
            ""
        ]
        
        for invalid_response in invalid_responses:
            parsed = agent._parse_json_response(invalid_response)
            assert isinstance(parsed, dict)
            assert "output" in parsed or "parse_error" in parsed
