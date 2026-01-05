import asyncio
import os
from datetime import datetime

from dotenv import load_dotenv
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from pydantic import BaseModel

load_dotenv()

model = ChatOllama(
    base_url=os.getenv("BASE_URL"),
    model=os.getenv("MODEL_NAME"),
    temperature=0.0,
)

@tool()
def search_web(query: str) -> list[Document]:
    """
    Performs a web search using Serper API and returns results.

    Args:
        query (str): The search query string.

    Returns:
        list[Document]: A list of Document objects containing page content and metadata.
    """
    search = GoogleSerperAPIWrapper()
    results = search.results(query)

    documents = [Document(
        page_content=result.get("snippet", "No snippet available"),
        metadata= {"title": result.get("title", "No title"), "url": result.get("link", "")})
        for result in results.get("organic", [])
    ]

    return documents

current_date = datetime.now().isoformat()

system_prompt = f"""
You're a friendly assistant and your goal is to answer general questions based on results provided by search.
You don't add anything yourself and provide only information baked by other sources. 
For your reference, the current date is {current_date}."""

class StockNewsSchema(BaseModel):
    headline: str
    source: str
    summary: str

agent = create_agent(
    model=model,
    tools=[search_web],
    response_format=StockNewsSchema
)

async def main():
    input_messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content="Az NVidia, Apple vagy a Microsoft részvények most jó befektetések? Milyen a befektetői hangulat?")
    ]
    async for step in agent.astream({"messages": input_messages}, stream_mode="values"):
        message = step["messages"][-1]
        message.pretty_print()

if __name__ == '__main__':
    asyncio.run(main())