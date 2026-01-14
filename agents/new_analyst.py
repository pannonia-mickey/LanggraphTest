import os
from datetime import datetime
from typing import Annotated

from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain.tools import tool


load_dotenv()

model = ChatOllama(
    base_url=os.getenv("BASE_URL"),
    model=os.getenv("MODEL_NAME"),
    temperature=0.0,
)


@tool
def get_global_news(
    query: Annotated[str, "Query to search with"],
    curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
    look_back_days: Annotated[int, "Number of days to look back"] = 365
) -> str:
    """
    Retrieve global news data using GoogleSerperAPIWrapper.

    Args:
        query (str): Query to search with
        curr_date (str): Current date in yyyy-mm-dd format
        look_back_days (int): Number of days to look back (default 7)

    Returns:
        str: A formatted string containing global news data
    """
    # Calculate the date range
    curr_date_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    before = curr_date_dt - relativedelta(days=look_back_days)
    before_str = before.strftime("%m/%d/%Y")
    curr_date_str = curr_date_dt.strftime("%m/%d/%Y")

    try:
        # Initialize the GoogleSerperAPIWrapper
        search = GoogleSerperAPIWrapper(
            type="news",
            tbs=f"cdr:1,cd_min:{before_str},cd_max:{curr_date}"
        )

        # Search for global market and economic news
        query = query.replace(" ", "+")
        results = search.results(query)

        # Format the news articles
        if "news" not in results or not results["news"]:
            return f"No news articles found for the period {before_str} to {curr_date_str}."

        formatted_news = []
        formatted_news.append(f"Global News Summary ({before_str} to {curr_date_str}):\n")
        formatted_news.append("=" * 80 + "\n")

        for idx, article in enumerate(results["news"], 1):
            title = article.get("title", "No title")
            snippet = article.get("snippet", "No description available")
            source = article.get("source", "Unknown source")
            date = article.get("date", "Unknown date")
            link = article.get("link", "")

            formatted_news.append(f"\n{idx}. {title}")
            formatted_news.append(f"   Source: {source} | Date: {date}")
            formatted_news.append(f"   Summary: {snippet}")
            if link:
                formatted_news.append(f"   URL: {link}")
            formatted_news.append("")

        return "\n".join(formatted_news)

    except Exception as e:
        return f"Error fetching news from SERPERAPI: {str(e)}"

ticker = "AMD"

agent = create_agent(
    model,
    tools=[get_global_news],
    system_prompt=f"""
You are a news researcher tasked with analyzing recent news and trends over the past week. Please write a comprehensive
report of the current state of the world that is relevant for trading and macroeconomics.
Make sure to append a Markdown table at the end of the report to organize key points in the report, organized and easy to read.
For your reference, the current date is {datetime.now()}. We are looking at the company {ticker}"""
)

for chunk in agent.stream({
    "messages": [HumanMessage(content="Summarize the news for the company")]
}):
    print(chunk)
