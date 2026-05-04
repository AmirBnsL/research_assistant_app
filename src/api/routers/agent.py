import os

from openai import OpenAI

from src.agent.orchestrator import ResearchOrchestrator

client = OpenAI(
    base_url=os.environ.get("OPENAI_BASE_URL"), api_key=os.environ.get("OPENAI_API_KEY")
)
Agent = ResearchOrchestrator().get_agent()
