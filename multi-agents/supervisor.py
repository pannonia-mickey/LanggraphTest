import os
from dotenv import load_dotenv

from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langgraph_supervisor import create_supervisor

load_dotenv()

def book_hotel(hotel_name: str):
    """Book a hotel"""
    return f"Successfully booked a stay at {hotel_name}."

def book_flight(from_airport: str, to_airport: str):
    """Book a flight"""
    return f"Successfully booked a flight from {from_airport} to {to_airport}."

flight_assistant = create_agent(
    model=f"ollama:{os.getenv('MODEL_NAME')}",
    tools=[book_flight],
    system_prompt="You are a flight booking assistant",
    name="flight_assistant"
)

hotel_assistant = create_agent(
    model=f"ollama:{os.getenv('MODEL_NAME')}",
    tools=[book_hotel],
    system_prompt="You are a hotel booking assistant",
    name="hotel_assistant"
)

supervisor = create_supervisor(
    agents=[flight_assistant, hotel_assistant],
    model=ChatOllama(model=os.getenv('MODEL_NAME')),
    prompt=(
        "You manage a hotel booking assistant and a"
        "flight booking assistant. Assign work to them."
    )
).compile()

for chunk in supervisor.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": "book a flight from BOS to JFK and a stay at McKittrick Hotel"
            }
        ]
    }
):
    print(chunk)