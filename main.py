from dotenv import load_dotenv
import os
load_dotenv()
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langchain.tools import tool

tavily = TavilySearchResults()

@tool
def search_web(query: str) -> str:
    """Search the web for real-time information. Use this for current events, news, or anything requiring up-to-date information."""
    results = tavily.invoke(query)
    return str(results)

llm = ChatOpenAI(model="gpt-3.5-turbo")
tools = [search_web]
agent = create_agent(llm, tools)
def main():
    user_input = input("You: ")
    result = agent.invoke({"messages": [HumanMessage(content=user_input)]})
    print(result["messages"][-1].content)




if __name__ == "__main__":
    main()
