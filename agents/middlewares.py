import os
from typing import TypedDict

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse, wrap_tool_call, dynamic_prompt, \
    TodoListMiddleware
from langchain.tools import tool
from langchain.messages import ToolMessage
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

load_dotenv()

basic_model = ChatOllama(
    base_url=os.getenv("BASE_URL"),
    model="llama3.1",
    temperature=0.0,
)

advanced_model = ChatOllama(
    base_url=os.getenv("BASE_URL"),
    model="qwen3",
    temperature=0.0,
)

class Context(TypedDict):
    user_role: str

# Tools give agents the ability to take actions.
@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"

@tool
def get_weather(location: str) -> str:
    """Get weather information for a location."""
    return f"Weather in {location}: Sunny, 72°F"

# Dynamic models are selected at runtime based on the current state and context.
# This enables sophisticated routing logic and cost optimization.
@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    """Choose model based on conversation complexity."""
    # message_count = len(request.state["messages"])
    user_role = request.runtime.context.get("user_role", "user")

    # if message_count > 10:
    if user_role == "expert":
        # Use an advanced model for longer conversations
        model = advanced_model
    else:
        model = basic_model

    return handler(request.override(model=model))

# Tool error handling
@wrap_tool_call
def handle_tool_errors(request: ModelRequest, handler) -> ToolMessage:
    """Handle tool execution errors with custom messages."""
    try:
        return handler(request)
    except Exception as e:
        # Return a custom error message to the model
        return ToolMessage(
            content=f"Tool error: Please check your input and try again. ({str(e)})",
            tool_call_id=request.tool_call["id"]
        )

# Dynamic system prompt
@dynamic_prompt
def user_role_prompt(request: ModelRequest) -> str:
    """Generate system prompt based on user role."""
    user_role = request.runtime.context.get("user_role", "user")
    base_prompt = "You are a helpful assistant."

    if user_role == "expert":
        return f"{base_prompt} Provide detailed technical responses."
    elif user_role == "beginner":
        return f"{base_prompt} Explain concepts simply and avoid jargon."

    return base_prompt


agent = create_agent(
    model=basic_model,
    tools=[search, get_weather],
    # system_prompt="You are a helpful assistant",
    middleware=[dynamic_model_selection, handle_tool_errors, user_role_prompt, TodoListMiddleware()],
    context_schema=Context
)

if __name__ == "__main__":
    result = agent.invoke({
           "messages": [HumanMessage(content="What's the weather in San Francisco?")]
        },
        context={"user_role": "expert"}
    )

    # Print the conversation
    for message in result["messages"]:
        if hasattr(message, 'pretty_print'):
            message.pretty_print()
        else:
            print(f"{message.type}: {message.content}")