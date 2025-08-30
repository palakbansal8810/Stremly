import json
import time
import asyncio
from abc import ABC, abstractmethod
from langchain_groq import ChatGroq
from src.retriever import Retriever
from src.utils import Citation, AgentResult
from src.exceptions import AgentError
from src.config import Config
import logging
from typing import List
from dotenv import load_dotenv
load_dotenv()
import os

os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    def __init__(self, retriever: Retriever, config: Config, agent_name: str):
        self.retriever = retriever
        self.config = config
        self.agent_name = agent_name
        self.llm = ChatGroq(
            model=config.LLM_MODEL,
            temperature=config.LLM_TEMPERATURE,
        )

    def _extract_citations(self, hits: List[tuple]) -> List[Citation]:
        """Extract citations from retrieval hits."""
        citations = []
        for text, score, doc_id, chunk_id in hits:
            rel_score = max(0.0, 1.0 - score)  # Clamp to >= 0.0
            citations.append(Citation(
                doc_id=doc_id,
                chunk_id=chunk_id,
                snippet=text[:200],
                relevance_score=rel_score
            ))
            print(f"Extracted citation: doc_id={doc_id}, chunk_id={chunk_id}, score={rel_score}")
        return citations

    def _parse_json_response(self, response_text: str) -> dict:
        """Safely parse JSON response from LLM."""
        try:
            # Try to find JSON in the response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            print(f"Parsing JSON from response: {response_text}")
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                return json.loads(json_str)
            else:
                # Fallback: assume entire response is the answer
                return {"output": response_text.strip()}
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON from {self.agent_name}: {e}")
            return {"output": response_text.strip(), "parse_error": str(e)}

    @abstractmethod
    async def _execute_core_logic(self, query: str, **kwargs) -> dict:
        """Core agent logic to be implemented by subclasses."""
        pass

    async def run(self, query: str, **kwargs) -> AgentResult:
        """Execute agent with error handling and timing."""
        start_time = time.time()
        
        try:
            output = await self._execute_core_logic(query, **kwargs)
            execution_time = time.time() - start_time
            if output is None:
                output = {}
            
            return AgentResult(
                agent_name=self.agent_name,
                execution_time=execution_time,
                status="success",
                output=output,
                citations=output.get("citations", [])
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Agent {self.agent_name} failed: {e}")
            
            return AgentResult(
                agent_name=self.agent_name,
                execution_time=execution_time,
                status="failure",
                output={},
                error_message=str(e)
            )

class PolicyAgent(BaseAgent):
    def __init__(self, retriever: Retriever, config: Config):
        super().__init__(retriever, config, "PolicyAgent")

    async def _execute_core_logic(self, query: str, **kwargs) -> dict:
        # Enhanced retrieval with filtering
        hits = self.retriever.search(f"policy compliance {query}", top_k=self.config.MAX_RETRIEVAL_RESULTS)
        policy_hits = [
            hit for hit in hits
            if isinstance(hit, (list, tuple)) and len(hit) > 3 and (
                str(hit[2]).lower().endswith('.pdf') or 'policy' in str(hit[3]).lower()
            )
        ]
        if not policy_hits:
            return {
                "assessment": "No relevant policies found",
                "compliance_status": "insufficient_evidence",
                "citations": []
            }

        context = "\n\n".join([
                f"[{doc[2]}]: {doc[0]}"
                for doc in policy_hits[:5]
                if isinstance(doc, (list, tuple)) and len(doc) > 2
            ])
        citations = self._extract_citations(policy_hits[:5])
        
        prompt = f"""You are a compliance expert analyzing policy requirements.

CONTEXT (Relevant Policy Excerpts):
{context}

QUERY: {query}

INSTRUCTIONS:
1. Analyze the policy excerpts for compliance requirements related to the query
2. Determine if the current implementation meets, violates, or has insufficient evidence for compliance
3. Provide specific policy references and requirements
4. Return response in JSON format

Required JSON format:
{{
    "assessment": "detailed analysis of compliance status",
    "compliance_status": "compliant|non_compliant|insufficient_evidence",
    "specific_requirements": ["list of specific policy requirements"],
    "gaps_identified": ["list of compliance gaps if any"],
    "recommendations": ["list of recommendations"]
}}"""

        response = await asyncio.to_thread(self.llm.invoke, prompt)
        parsed_response = self._parse_json_response(response.content)
        parsed_response["citations"] = citations
        print(f"PolicyAgent parsed response: {parsed_response}")
        return parsed_response

class RiskAgent(BaseAgent):
    def __init__(self, retriever: Retriever, config: Config):
        super().__init__(retriever, config, "RiskAgent")

    async def _execute_core_logic(self, query: str, **kwargs) -> dict:
        hits = self.retriever.search(f"risk security vulnerability {query}", top_k=self.config.MAX_RETRIEVAL_RESULTS)
        # Defensive: only use valid tuples/lists
        valid_hits = [
            hit for hit in hits
            if isinstance(hit, (list, tuple)) and len(hit) > 3
        ]
        context = "\n\n".join([
            f"[{doc[2]}]: {doc[0]}"
            for doc in valid_hits[:5]
        ])
        citations = self._extract_citations(valid_hits[:5])
        
        prompt = f"""You are a cybersecurity risk analyst.

CONTEXT (Code/Specs for Risk Analysis):
{context}

QUERY: {query}

INSTRUCTIONS:
Analyze for security risks, vulnerabilities, and compliance risks.
Return response in JSON format.

Required JSON format:
{{
    "risk_score": 0.0-1.0,
    "risk_level": "low|medium|high|critical",
    "vulnerabilities_identified": ["list of specific vulnerabilities"],
    "security_concerns": ["list of security concerns"],
    "mitigation_recommendations": ["list of recommendations"],
    "rationale": "detailed explanation of risk assessment"
}}"""

        response = await asyncio.to_thread(self.llm.invoke, prompt)
        parsed_response = self._parse_json_response(response.content)
        parsed_response["citations"] = citations
        print(f"RiskAgent parsed response: {parsed_response}")
        return parsed_response

class CodeScannerAgent(BaseAgent):
    def __init__(self, retriever: Retriever, config: Config):
        super().__init__(retriever, config, "CodeScannerAgent")

    async def _execute_core_logic(self, query: str, **kwargs) -> dict:
        hits = self.retriever.search(f"code implementation {query}", top_k=self.config.MAX_RETRIEVAL_RESULTS)
        # Defensive: only use valid tuples/lists and check for 'code' in chunk_id
        code_hits = [
            hit for hit in hits
            if isinstance(hit, (list, tuple)) and len(hit) > 3 and 'code' in str(hit[3]).lower()
        ]
        context = "\n\n".join([
            f"[{doc[2]}]: {doc[0]}"
            for doc in code_hits[:5]
        ])
        citations = self._extract_citations(code_hits[:5])
        
        prompt = f"""You are a code security scanner analyzing implementation for compliance.

CODE CONTEXT:
{context}

QUERY: {query}

INSTRUCTIONS:
Scan code for compliance-relevant patterns, security issues, and implementation gaps.

Required JSON format:
{{
    "scan_results": "summary of findings",
    "security_issues": ["list of security issues found"],
    "compliance_patterns": ["list of compliance-relevant code patterns"],
    "missing_implementations": ["list of missing security/compliance features"],
    "code_quality_score": 0.0-1.0
}}"""

        response = await asyncio.to_thread(self.llm.invoke, prompt)
        parsed_response = self._parse_json_response(response.content)
        
        parsed_response["citations"] = citations
        print(f"CodeScannerAgent parsed response: {parsed_response}")
        return parsed_response