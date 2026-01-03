from langchain.agents import create_agent
# highlight-next-line
from langgraph_swarm import create_swarm, create_handoff_tool


def book_hotel(hotel_name: str):
    """Book a hotel"""
    return f"Successfully booked a stay at {hotel_name}."

def book_flight(from_airport: str, to_airport: str):
    """Book a flight"""
    return f"Successfully booked a flight from {from_airport} to {to_airport}."

transfer_to_hotel_assistant = create_handoff_tool(
    agent_name="hotel_assistant",
    description="Transfer user to the hotel-booking assistant.",
)
transfer_to_flight_assistant = create_handoff_tool(
    agent_name="flight_assistant",
    description="Transfer user to the flight-booking assistant.",
)

flight_assistant = create_agent(
    model="ollama:mistral",
    # highlight-next-line
    tools=[book_flight, transfer_to_hotel_assistant],
    system_prompt="You are a flight booking assistant",
    # highlight-next-line
    name="flight_assistant"
)
hotel_assistant = create_agent(
    model="ollama:mistral",
    # highlight-next-line
    tools=[book_hotel, transfer_to_flight_assistant],
    system_prompt="You are a hotel booking assistant",
    # highlight-next-line
    name="hotel_assistant"
)

# highlight-next-line
swarm = create_swarm(
    agents=[flight_assistant, hotel_assistant],
    default_active_agent="flight_assistant"
).compile()

for chunk in swarm.stream(
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