import os

from deepagents import create_deep_agent
from dotenv import load_dotenv
from langchain_ollama import ChatOllama

from deep_agents_from_scratch.full_agent import sub_agent_tools, INSTRUCTIONS
from deep_agents_from_scratch.prompt import RESEARCHER_INSTRUCTIONS
from deep_agents_from_scratch.research_tools import get_today_str
from deep_agents_from_scratch.task_tool import SubAgent
from deep_agents_from_scratch.utils import format_messages

load_dotenv()

# Create agent using create_react_agent directly
model = ChatOllama(
    base_url=os.getenv("BASE_URL"),
    model=os.getenv("MODEL_NAME"),
    temperature=0.0,
)

# Create research sub-agent
research_sub_agent = SubAgent(
    name="research-agent",
    description="Delegate research to the sub-agent researcher. Only give this researcher one topic at a time.",
    system_prompt=RESEARCHER_INSTRUCTIONS.format(date=get_today_str()),
    tools=["web_search", "think_tool"],
)

agent = create_deep_agent(  # updated
    tools=sub_agent_tools,
    system_prompt=INSTRUCTIONS,
    subagents=[research_sub_agent],
    model=model,
)

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Give me an overview of Model Context Protocol (MCP).",
            }
        ],
    }
)

format_messages(result["messages"])