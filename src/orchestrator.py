import asyncio
import time
from typing import Dict, Any, List, Optional, Literal
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential
from pybreaker import CircuitBreaker
import pymongo
import logging
from contextlib import asynccontextmanager

from src.retriever import Retriever
from src.agents import PolicyAgent, RiskAgent, CodeScannerAgent
from src.utils import OutputSchema, HumanInteraction, AgentResult, generate_session_id
from src.exceptions import ComplianceSystemError, HumanInteractionTimeoutError, AgentError
from src.config import Config

logger = logging.getLogger(__name__)

class HitlRequest(BaseModel):
    session_id: str
    type: Literal["clarification", "approval", "upload_request", "validation"]
    prompt: str
    required_artifact: Optional[str] = None
    timeout_seconds: int = 300

class ComplianceOrchestrator:
    def __init__(self, retriever: Retriever, config: Config, mongo_client: pymongo.MongoClient):
        self.retriever = retriever
        self.config = config
        self.mongo = mongo_client["compliance_db"]["sessions"]
        
        # Initialize agents
        self.policy_agent = PolicyAgent(retriever, config)
        self.risk_agent = RiskAgent(retriever, config)
        self.code_agent = CodeScannerAgent(retriever, config)
        
        # Circuit breaker for agent failures
        self.breaker = CircuitBreaker(fail_max=config.CIRCUIT_BREAKER_THRESHOLD)
        
        # Human-in-the-loop management
        self.hitl_events = {}  # session_id -> asyncio.Event
        self.hitl_responses = {}  # session_id -> response
        self.active_sessions = set()

    @asynccontextmanager
    async def session_context(self, session_id: str):
        """Context manager for session lifecycle."""
        self.active_sessions.add(session_id)
        try:
            yield
        finally:
            self.active_sessions.discard(session_id)
            # Cleanup HITL resources
            self.hitl_events.pop(session_id, None)
            self.hitl_responses.pop(session_id, None)

    async def _persist_session_state(self, session_id: str, state: dict):
        """Persist session state to MongoDB."""
        try:
            await asyncio.to_thread(
                self.mongo.update_one,
                {"_id": session_id},
                {"$set": {**state, "last_updated": time.time()}},
                upsert=True
            )
            print(f"Persisted state for session {session_id}: {state}")
        except Exception as e:
            logger.error(f"Failed to persist session {session_id}: {e}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _run_agent_with_circuit_breaker(self, agent, query: str, timeout: int = None, **kwargs) -> AgentResult:
        """Run agent with circuit breaker and timeout."""
        timeout = timeout or self.config.AGENT_TIMEOUT
        
        @self.breaker
        async def wrapped_agent():
            try:
                return await asyncio.wait_for(agent.run(query, **kwargs), timeout=timeout)
            except asyncio.TimeoutError:
                raise AgentError(f"Agent {agent.agent_name} timed out after {timeout}s")
                
        return await wrapped_agent()

    async def _request_human_input(self, session_id: str, request: HitlRequest) -> str:
        """Request human input with timeout handling."""
        if session_id not in self.active_sessions:
            print(f"Session {session_id} not active for HITL request")
            raise HumanInteractionTimeoutError("Session no longer active")
            
        # Create event for this request
        event = asyncio.Event()
        self.hitl_events[session_id] = event
        
        # Broadcast request (this would integrate with your WebSocket system)
        await self._broadcast_hitl_request(session_id, request)
        
        try:
            print(f"Waiting for human input for session {session_id} with timeout {request.timeout_seconds}s")
            # Wait for response with timeout
            await asyncio.wait_for(event.wait(), timeout=request.timeout_seconds)
            response = self.hitl_responses.get(session_id, "")
            
            # Log interaction
            interaction = HumanInteraction(
                type=request.type,
                prompt=request.prompt,
                response=response,
                status="provided"
            )
            
            return response, interaction
            
        except asyncio.TimeoutError:
            interaction = HumanInteraction(
                type=request.type,
                prompt=request.prompt,
                status="timeout"
            )
            raise HumanInteractionTimeoutError(f"Human input timed out after {request.timeout_seconds}s")

    async def _broadcast_hitl_request(self, session_id: str, request: HitlRequest):
    # Actually send the HITL request to the WebSocket clients
        if hasattr(self, "send_hitl_request") and callable(self.send_hitl_request):
            await self.send_hitl_request(session_id, request)
        else:
            print(f"[WARNING] send_hitl_request not set on orchestrator. HITL not delivered.")
            logger.warning(f"send_hitl_request not set on orchestrator. HITL not delivered.")

    async def handle_hitl_response(self, session_id: str, response: str):
        """Handle human response to HITL request."""
        self.hitl_responses[session_id] = response
        if session_id in self.hitl_events:
            self.hitl_events[session_id].set()

    async def _analyze_phase(self, session_id: str, query: str, image_data: Optional[bytes] = None) -> Dict[str, AgentResult]:
        """Phase 1: Parallel analysis by specialized agents."""
        await self._persist_session_state(session_id, {"status": "analyzing", "phase": "data_collection"})
        
        # Run agents in parallel with different priorities
        high_priority_tasks = {
            "policy": self._run_agent_with_circuit_breaker(self.policy_agent, query),
            "risk": self._run_agent_with_circuit_breaker(self.risk_agent, query),
        }
        
        medium_priority_tasks = {
            "code": self._run_agent_with_circuit_breaker(self.code_agent, query),
        }
        
        # Execute high priority first
        high_priority_results = await asyncio.gather(
            *high_priority_tasks.values(), 
            return_exceptions=True
        )
        
        # Execute medium priority
        medium_priority_results = await asyncio.gather(
            *medium_priority_tasks.values(),
            return_exceptions=True
        )
        
        # Combine results
        all_tasks = {**high_priority_tasks, **medium_priority_tasks}
        all_results = high_priority_results + medium_priority_results
        
        agent_results = {}
        for (task_name, _), result in zip(all_tasks.items(), all_results):
            print(f"Agent {task_name} completed with result: {result}")
            if isinstance(result, Exception):
                logger.error(f"Agent {task_name} failed: {result}")
                agent_results[task_name] = AgentResult(
                    agent_name=task_name,
                    execution_time=0.0,
                    status="failure",
                    output={},
                    error_message=str(result)
                )
            else:
                agent_results[task_name] = result
                
        return agent_results

    async def _synthesis_phase(self, session_id: str, agent_results: Dict[str, AgentResult]) -> Dict[str, Any]:
        """Phase 2: Synthesize agent results and compute overall risk."""
        await self._persist_session_state(session_id, {"status": "synthesizing", "phase": "analysis"})
        
        # Extract successful results
        successful_results = {
            name: result.output for name, result in agent_results.items() 
            if result.status == "success"
        }
        
        if len(successful_results) < 1:  # Minimum threshold
            return {
                "decision": "insufficient_evidence",
                "confidence": 0.0,
                "risk_score": 0.0,
                "rationale": "Insufficient agent responses for reliable analysis"
            }
        
        # Compute aggregated risk score
        risk_scores = []
        for name, result in successful_results.items():
            print(f"Synthesis processing result from {name}: {result}")
            if "output" in result and isinstance(result["output"], str):
                # Handle JSON parsing failure in agent
                logger.warning(f"Agent {name} returned plain text: {result['output']}")
                risk_scores.append(0.5)  # Default risk score for malformed output
            elif name == "risk" and "risk_score" in result:
                risk_scores.append(float(result["risk_score"]))
            elif name == "code" and "code_quality_score" in result:
                risk_scores.append(1.0 - float(result["code_quality_score"]))
        
        avg_risk_score = sum(risk_scores) / len(risk_scores) if risk_scores else 0.5
        
        # Determine overall compliance decision
        policy_status = successful_results.get("policy", {}).get("compliance_status", "insufficient_evidence")
        if "output" in successful_results.get("policy", {}) and isinstance(successful_results["policy"]["output"], str):
            policy_status = "insufficient_evidence"  # Fallback for malformed PolicyAgent output
        
        if policy_status == "non_compliant" or avg_risk_score > 0.7:
            decision = "non_compliant"
            confidence = 0.8
        elif policy_status == "compliant" and avg_risk_score < 0.3:
            decision = "compliant"
            confidence = 0.9
        else:
            decision = "requires_human_review"
            confidence = 0.6
            
        return {
            "decision": decision,
            "confidence": confidence,
            "risk_score": avg_risk_score,
            "synthesis_results": successful_results
        }

    async def _validation_phase(self, session_id: str, synthesis_results: Dict[str, Any]) -> List[str]:
        """Phase 3: Validate results and identify open questions."""
        await self._persist_session_state(session_id, {"status": "validating", "phase": "validation"})
        
        open_questions = []
        
        # Check for low confidence decisions
        if synthesis_results["confidence"] < 0.7:
            open_questions.append("Low confidence in analysis - recommend manual review")
            
        # Check for missing critical data
        if synthesis_results["decision"] == "insufficient_evidence":
            open_questions.append("Insufficient policy documentation found")
            
        # Check for high risk with compliant decision (potential false positive)
        if synthesis_results["decision"] == "compliant" and synthesis_results["risk_score"] > 0.6:
            open_questions.append("High risk score despite compliant decision - requires validation")
            
        return open_questions

    async def _human_review_phase(self, session_id: str, synthesis_results: Dict[str, Any], open_questions: List[str]) -> List[HumanInteraction]:
        """Phase 4: Handle human interactions if needed."""
        interactions = []
        
        # Request human review for edge cases
        if synthesis_results["decision"] == "requires_human_review":
            try:
                print(f"Requesting human approval for session {session_id} due to mixed signals")
                request = HitlRequest(
                    session_id=session_id,
                    type="approval",
                    prompt=f"Analysis shows mixed compliance signals. Risk score: {synthesis_results['risk_score']:.2f}. Approve final decision?",
                    timeout_seconds=self.config.HITL_TIMEOUT
                )
                
                response, interaction = await self._request_human_input(session_id, request)
                interactions.append(interaction)
                
                if response.lower() in ["approved", "approve", "yes"]:
                    synthesis_results["decision"] = "compliant" if synthesis_results["risk_score"] < 0.5 else "non_compliant"
                    synthesis_results["confidence"] = min(synthesis_results["confidence"] + 0.2, 1.0)
                    
            except HumanInteractionTimeoutError as e:
                interactions.append(e.args[1] if len(e.args) > 1 else HumanInteraction(
                    type="approval", prompt="Human review timeout", status="timeout"
                ))
        
        # Handle open questions
        for question in open_questions:
            try:
                request = HitlRequest(
                    session_id=session_id,
                    type="clarification",
                    prompt=question,
                    timeout_seconds=self.config.HITL_TIMEOUT
                )
                
                response, interaction = await self._request_human_input(session_id, request)
                interactions.append(interaction)
                
            except HumanInteractionTimeoutError as e:
                interactions.append(e.args[1] if len(e.args) > 1 else HumanInteraction(
                    type="clarification", prompt=question, status="timeout"
                ))
                
        return interactions

    async def run(self, session_id: str, query: str, image_data: Optional[bytes] = None) -> OutputSchema:
        """Enhanced orchestration with proper phase management."""
        start_time = time.time()
        print(f"Starting compliance session {session_id} for query: {query}")
        
        async with self.session_context(session_id):
            try:
                await self._persist_session_state(session_id, {
                    "query": query,
                    "status": "started",
                    "start_time": start_time
                })
                
                # Phase 1: Data Collection and Analysis
                logger.info(f"Starting analysis phase for session {session_id}")
                agent_results = await self._analyze_phase(session_id, query, image_data)
                
                # Phase 2: Synthesis
                logger.info(f"Starting synthesis phase for session {session_id}")
                synthesis_results = await self._synthesis_phase(session_id, agent_results)
                
                # Phase 3: Validation
                logger.info(f"Starting validation phase for session {session_id}")
                open_questions = await self._validation_phase(session_id, synthesis_results)
                
                # Phase 4: Human Review (if needed)
                human_interactions = []
                if synthesis_results["decision"] == "requires_human_review" or open_questions:
                    logger.info(f"Starting human review phase for session {session_id}")
                    human_interactions = await self._human_review_phase(session_id, synthesis_results, open_questions)
                
                # Compile final result
                all_citations = []
                for result in agent_results.values():
                    all_citations.extend(result.citations)
                
                processing_time = time.time() - start_time
                
                final_result = OutputSchema(
                    session_id=session_id,
                    decision=synthesis_results["decision"],
                    confidence=synthesis_results["confidence"],
                    risk_score=synthesis_results["risk_score"],
                    rationale=synthesis_results.get("synthesis_results", {}).get("policy", {}).get("assessment", "Analysis completed"),
                    citations=all_citations,
                    open_questions=open_questions,
                    human_interactions=human_interactions,
                    agent_results=list(agent_results.values()),
                    processing_time=processing_time
                )
                
                # Final persistence
                await self._persist_session_state(session_id, {
                    "result": final_result.dict(),
                    "status": "completed",
                    "processing_time": processing_time
                })
                
                logger.info(f"Completed session {session_id} in {processing_time:.2f}s")
                return final_result
                
            except Exception as e:
                processing_time = time.time() - start_time
                print(f"Session {session_id} encountered an error: {e}")
                error_msg = f"Orchestration failed: {str(e)}"
                logger.error(f"Session {session_id} failed: {error_msg}")
                
                await self._persist_session_state(session_id, {
                    "status": "error",
                    "error": error_msg,
                    "processing_time": processing_time
                })
                
                # Return error result
                return OutputSchema(
                    session_id=session_id,
                    decision="insufficient_evidence",
                    confidence=0.0,
                    risk_score=0.0,
                    rationale=error_msg,
                    citations=[],
                    open_questions=[f"System error: {error_msg}"],
                    human_interactions=[],
                    agent_results=[],
                    processing_time=processing_time
                )
