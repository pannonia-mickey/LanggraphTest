import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from datetime import datetime

from deep_agents_from_scratch.file_tools import ls, read_file, write_file
from deep_agents_from_scratch.prompt import RESEARCHER_INSTRUCTIONS, SUBAGENT_USAGE_INSTRUCTIONS, TODO_USAGE_INSTRUCTIONS, \
    FILE_USAGE_INSTRUCTIONS
from deep_agents_from_scratch.research_tools import think_tool, get_today_str, web_search
from deep_agents_from_scratch.state import DeepAgentState
from deep_agents_from_scratch.task_tool import _create_task_tool, SubAgent
from deep_agents_from_scratch.todo_tools import write_todos, read_todos
from deep_agents_from_scratch.utils import format_messages

load_dotenv()

# Create agent using create_react_agent directly
model = ChatOllama(
    base_url=os.getenv("BASE_URL"),
    model=os.getenv("MODEL_NAME"),
    temperature=0.0,
)

# Limits
max_concurrent_research_units = 3
max_researcher_iterations = 3

# Tools
sub_agent_tools = [web_search, think_tool]
built_in_tools = [ls, read_file, write_file, write_todos, read_todos, think_tool]

# Create research sub-agent
research_sub_agent = SubAgent(
    name="research-agent",
    description="Delegate research to the sub-agent researcher. Only give this researcher one topic at a time.",
    prompt=RESEARCHER_INSTRUCTIONS.format(date=get_today_str()),
    tools=["web_search", "think_tool"],
)

# Create task tool to delegate tasks to sub-agents
task_tool = _create_task_tool(
    sub_agent_tools, [research_sub_agent], model, DeepAgentState
)

delegation_tools = [task_tool]
all_tools = sub_agent_tools + built_in_tools + delegation_tools  # search available to main agent for trivial cases

# Build prompt
SUBAGENT_INSTRUCTIONS = SUBAGENT_USAGE_INSTRUCTIONS.format(
    max_concurrent_research_units=max_concurrent_research_units,
    max_researcher_iterations=max_researcher_iterations,
    date=datetime.now().strftime("%a %b %d, %Y"),
)

INSTRUCTIONS = (
    "# TODO MANAGEMENT\n"
    + TODO_USAGE_INSTRUCTIONS
    + "\n\n"
    + "=" * 80
    + "\n\n"
    + "# FILE SYSTEM USAGE\n"
    + FILE_USAGE_INSTRUCTIONS
    + "\n\n"
    + "=" * 80
    + "\n\n"
    + "# SUB-AGENT DELEGATION\n"
    + SUBAGENT_INSTRUCTIONS
)

# Create agent
agent = create_agent(  #updated 1.0
    model, all_tools, system_prompt=INSTRUCTIONS, state_schema=DeepAgentState
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