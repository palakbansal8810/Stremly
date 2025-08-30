# tests/test_agent.py
import sys
import os
from dotenv import load_dotenv
load_dotenv()
os.environ['GROQ_API_KEY'] = os.environ.get('GROQ_API_KEY')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.orchestrator import Orchestrator  # Import the orchestrator
from src.retriever import Retriever
from src.agents import PolicyAgent, RiskAgent, SpecAgent

def test_agents():
    retriever = Retriever()
    agents = [
        PolicyAgent(retriever),
        RiskAgent(retriever),
        SpecAgent(retriever)
    ]

    for agent in agents:
        result = agent.run("Check MFA compliance in the login flow")
        print(f"--- {agent.__class__.__name__} ---")
        print(result["response"])
        print()

def test_orchestrator():
    orchestrator = Orchestrator()
    query = "Does our new login flow meet MFA requirements under Policy X?"
    # Simulate multimodal input (text only for now)
    result = orchestrator.run(
        session_id="test_session_1",
        question=query,
        attachments=None  # Add image/file mocks if needed
    )
    print("--- Orchestrator Output ---")
    print(result)
    print()

if __name__ == "__main__":
    print("=== Testing Individual Agents ===")
    test_agents()
    print("=== Testing Orchestrator Workflow ===")
    test_orchestrator()