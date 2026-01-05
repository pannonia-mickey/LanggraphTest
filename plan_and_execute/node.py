import ast
import os
import re
from typing import Literal, List, Union

from dotenv import load_dotenv
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import Tool
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from pydantic import Field, BaseModel

from state import PlanExecute


load_dotenv()

llm = ChatOllama(
    base_url=os.getenv("BASE_URL"),
    model=os.getenv("MODEL_NAME"),
    temperature=0.0,
)

class Plan(BaseModel):
    """Plan to follow in future"""

    steps: List[str] = Field(...,
        description="Different steps to follow, should be in sorted order"
    )

class Response(BaseModel):
    """Response to user."""

    response: str = Field(...,
        description="The final answer to the user."
    )


class Act(BaseModel):
    """Action to perform."""

    action: Union[Response, Plan] = Field(
        description="Action to perform. If you want to respond to user, use Response. "
        "If you need to further use tools to get the answer, use Plan."
    )

def extract_json(message: AIMessage) -> List[dict]:
    """Extracts JSON content from a string where JSON is embedded between ```json and ``` tags.

    Parameters:
        text (str): The text containing the JSON content.

    Returns:
        list: A list of extracted JSON strings.
    """
    text = message.content
    # Define the regular expression pattern to match JSON blocks
    pattern = r"\`\`\`json(.*?)\`\`\`"

    # Find all non-overlapping matches of the pattern in the string
    matches = re.findall(pattern, text, re.DOTALL)

    # Return the list of matched JSON strings, stripping any leading or trailing whitespace
    if not matches: #empty
        match = text
    else:
        match = matches[-1].strip()

    if not (match[0] == "{"):
        match = "{" + match + "}"
    python_obj = ast.literal_eval(match)
    return [python_obj]


async def execute_step(state: PlanExecute):
    search = GoogleSerperAPIWrapper()
    tools = [
        Tool(
            name="Google Search",
            func=search.run,
            description="useful for when you need to search on Google",
        )
    ]

    agent_executor = create_agent(llm, tools, system_prompt="You are a helpful agent.")

    plan = state["plan"]
    plan_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))
    task = plan[0]
    task_formatted = f"""For the following plan:
{plan_str}\n\nYou are tasked with executing step {1}, {task}."""
    agent_response = await agent_executor.ainvoke(
        {"messages": [("user", task_formatted)]}
    )
    return {
        "past_steps": [(task, agent_response["messages"][-1].content)],
    }


async def plan_step(state: PlanExecute):
    planner_prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
                """For the given objective, come up with a simple step by step plan. \
    This plan should involve individual tasks without any user interaction, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
    The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.
        
    
    Output should be the a step-by-step Python list in JSON format. For example,
    ```json
        "steps": ["Perform the first task", 
                  "Search the answer", 
                  "Give the final answer"]
    ```
    Escape any character that avoids parsing the JSON object. 
    """,
            ),
            ("placeholder", "{messages}"),
        ]
    )

    planner = planner_prompt | llm | extract_json

    output = await planner.ainvoke({"messages": [("user", state["input"])]})

    return {"plan": output[0]["steps"]}


async def replan_step(state: PlanExecute):
    replanner_prompt = ChatPromptTemplate.from_template(
        """For the given objective, come up with a simple step by step plan. \
    This plan should involve individual tasks without any user interaction, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
    The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

    Your objective was this:
    {input}

    Your original plan was this:
    {plan}

    You have currently done the follow steps:
    {past_steps}

    If no more steps are needed and you can return to the user, then respond with that. \
    Otherwise, update your plan accordingly. Only add steps to the plan that still NEED to be done. \
    Do not return previously done steps as part of the plan.
    
    If you want to respond to user, output should be in JSON format. For example,
    ```json
        'response': 'Final answer to the user.'
    ```
    
    If you need to further use tools to get the answer, output should be a step-by-step Python list in JSON format. For example,
    ```json
        'steps': ['Load data', 
                  'Process data', 
                  'Save data']
    ```
    Ensure escaping any character that avoids parsing in the answer to JSON object. 
    """
    )

    replanner = replanner_prompt | llm | extract_json

    output = await replanner.ainvoke(state)

    if "steps" in output[0]:
        return {"plan": output[0]["steps"]}
    else:
        return {"response": output[0]["response"]}


def should_end(state: PlanExecute) -> Literal["agent", "__end__"]:
    if "response" in state and state["response"]:
        return "__end__"
    else:
        return "agent"