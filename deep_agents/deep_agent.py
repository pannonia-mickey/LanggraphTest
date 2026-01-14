import os

from deepagents import create_deep_agent, SubAgent
from deepagents.backends import FilesystemBackend
from dotenv import load_dotenv
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.store.memory import InMemoryStore

load_dotenv()

model = ChatOllama(
    base_url=os.getenv("BASE_URL"),
    model=os.getenv("MODEL_NAME"),
    temperature=0.0,
)


# Define custom tools
@tool()
def search_tool(query: str):
    """
    Performs a web search using Serper API and returns results.

    Args:
        query (str): The search query string.
    """
    search = GoogleSerperAPIWrapper()
    results = search.results(query)

    return results

# Custom system prompt for the agent
custom_prompt = """
You are a job application assistant. Your tasks:
1. Search for job openings based on user input.
2. Generate tailored cover letters.
Use the filesystem to store job details and drafts.
Delegate to sub-agents for research if needed.
"""

# Create sub-agent for research
research_subagent = SubAgent(
    name="researcher",
    description="Research job market",
    system_prompt="Research job markets and company info.",
    tools=[search_tool],
    model=model
)

# Build the deep agent
agent = create_deep_agent(
    model=model,
    system_prompt=custom_prompt,
    subagents=[research_subagent],
    store=InMemoryStore(),
    backend=FilesystemBackend(),
)

# Run the agent with a user query
for chunk in agent.stream({
        "messages": [HumanMessage(content="Find software engineer jobs in Hungary and draft a cover letter in Hungarian.")]
    }):
    print(chunk)
