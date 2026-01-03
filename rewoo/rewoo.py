import re
from typing import List, TypedDict

from dotenv import load_dotenv
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama
from langgraph.constants import END, START
from langgraph.graph import StateGraph


load_dotenv()

model = ChatOllama(
    base_url="http://localhost:11434",
    model="mistral",
    temperature=0.0,
)

search = GoogleSerperAPIWrapper()

# Define graph state

class ReWOO(TypedDict):
    task: str
    step: str
    results: List[str]
    result: str

# Planner

prompt = """For the following task, make the first step of a plan that can solve the problem. For each step, indicate \
which external tool together with tool input to retrieve evidence. You can store the evidence into a \
variable #E that can be called by later tools. (Step, #E = Tool[tool input])

Tools can be one of the following:
(1) WebSearch[input]: Worker that searches results from web. Useful when you need to find short
and succinct answers about a specific topic. The input should be a search query.
(2) LLM[input]: A pretrained LLM like yourself. Useful when you need to act with general
world knowledge and common sense. Prioritize it when you are confident in solving the problem
yourself. Input can be any instruction.

For example,
Task: Thomas, Toby, and Rebecca worked a total of 157 hours in one week. Thomas worked x
hours. Toby worked 10 hours less than twice what Thomas worked, and Rebecca worked 8 hours
less than Toby. How many hours did Rebecca work?
Step: Given Thomas worked x hours, translate the problem into algebraic expressions and solve
with Wolfram Alpha. #E = WolframAlpha[Solve x + (2x − 10) + ((2x − 10) − 8) = 157]

Begin! 
Describe your step with rich details. Each Step should be followed by only one #E.

Task: {task}"""

regex_pattern = r"Step( \d*)*:\s*(.+)\s*(#E\d*)\s*=\s*(\w+)\s*\[([^\]]+)\]"
prompt_template = ChatPromptTemplate.from_messages([("user", prompt)])
planner = prompt_template | model

def get_plan(state: ReWOO):
    task = state["task"]
    result = planner.invoke({"task": task})
    # Find all matches in the sample text
    matches = re.findall(regex_pattern, result.content)
    return {"step": matches[0] if len(matches) > 0 else None}

# Reviewer

review_prompt = """For the following task and the last result, make the next step of a plan that can solve the problem. \
For each step, indicate which external tool together with tool input to retrieve evidence. You can store the evidence into a \
variable #E that can be called by later tools. (Step, #E = Tool[tool input])

Tools can be one of the following:
(1) WebSearch[input]: Worker that searches results from web. Useful when you need to find short
and succinct answers about a specific topic. The input should be a search query.
(2) LLM[input]: A pretrained LLM like yourself. Useful when you need to act with general
world knowledge and common sense. Prioritize it when you are confident in solving the problem
yourself. Input can be any instruction.

For example,
Task: Thomas, Toby, and Rebecca worked a total of 157 hours in one week. Thomas worked x
hours. Toby worked 10 hours less than twice what Thomas worked, and Rebecca worked 8 hours
less than Toby. How many hours did Rebecca work?
Step: Given Thomas worked x hours, translate the problem into algebraic expressions and solve
with Wolfram Alpha. #E = WolframAlpha[Solve x + (2x − 10) + ((2x − 10) − 8) = 157]

Begin! 
Describe your step with rich details. Each Step should be followed by only one #E.

Task: {task}

Last step: {step}

Results: 
{results}
"""

review_prompt_template = ChatPromptTemplate.from_messages([("user", review_prompt)])
reviewer = review_prompt_template | model

def review_plan(state: ReWOO):
    task = state["task"]
    step = state["step"]
    results = state["results"]

    result = reviewer.invoke({
        "task": task,
        "step": step,
        "results": "\n- ".join(results),
    })
    # Find all matches in the sample text
    matches = re.findall(regex_pattern, result.content)
    return {"step": matches[0] if len(matches) > 0 else None}

# Executor

def tool_execution(state: ReWOO):
    """Worker node that executes the tools of a given plan."""
    results = (state["results"] or []) if "results" in state else []
    _index, _plan, step_name, tool, tool_input = state["step"]
    if tool == "WebSearch":
        result = search.run(tool_input)
    elif tool == "LLM":
        result = model.invoke(tool_input).content
    else:
        raise ValueError
    results.append(str(result))
    return {"results": results}


# Solver

solve_prompt = """Solve the following task or problem. To solve the problem, we have made step-by-step plan and \
retrieved corresponding results to each step. Use them with caution since long result might \
contain irrelevant information.

Now solve the question or task according to provided results. Respond with the answer directly with no extra words.

Task: {task}

Results: {results}

Response:"""


def solve(state: ReWOO):
    formatted_prompt = solve_prompt.format(
        task=state["task"],
        results="\n- ".join(state["results"])
    )
    result = model.invoke(formatted_prompt)
    return {"result": result.content}

# Define Graph

def _route(state):
    _step = state["step"]
    if _step is None:
        # We have executed all tasks
        return "solve"
    else:
        # We are still executing tasks, loop back to the "tool" node
        return "tool"

graph = StateGraph(ReWOO)
graph.add_node("plan", get_plan)
graph.add_node("tool", tool_execution)
graph.add_node("review", review_plan)
graph.add_node("solve", solve)

graph.add_edge(START, "plan")
graph.add_edge("plan", "tool")
graph.add_edge("tool", "review")
graph.add_conditional_edges("review", _route)
graph.add_edge("solve", END)

app = graph.compile()

response = app.invoke({"task": "What is the most valueable stock share?"})
print(response)
