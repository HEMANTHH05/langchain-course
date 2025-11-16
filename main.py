import json

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from tavily import TavilyClient

from schemas import AgentResponse

load_dotenv()

tavily = TavilyClient()

@tool
def search(query: str) -> str:
    """Searches the web using Tavily and returns the top results as JSON (title, url)."""
    res = tavily.search(query=query, max_results=5)
    items = [{"title": r.get("title"), "url": r.get("url")} for r in res.get("results", [])[:5]]
    return json.dumps(items)

# 1) Plain LLM (no with_structured_output here)
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# 2) Pydantic parser + format instructions
parser = PydanticOutputParser(pydantic_object=AgentResponse)
format_instructions = parser.get_format_instructions()

SYSTEM_PROMPT = (
    "You are a helpful job-search assistant. Use the `search` tool for live results.\n"
    "Return ONLY in the following JSON schema:\n"
    f"{format_instructions}\n"
)

# 3) Build agent the way create_agent expects (chat model + tools + plain string prompt)
agent = create_agent(
    model=llm,
    tools=[search],
    system_prompt=SYSTEM_PROMPT,
)

def _final_text(result: dict) -> str:
    """Extract the final text from the agent result across versions."""
    if isinstance(result, dict):
        if result.get("output"):
            return result["output"]
        msgs = result.get("messages") or []
        if msgs:
            last = msgs[-1]
            # LangChain message objects typically have .content
            content = getattr(last, "content", None)
            if content:
                return content
    return str(result)

def main():
    result = agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": "Search for 3 job postings for an AI engineer using LangChain in the Bay Area on LinkedIn and list their details + links.",
            }
        ]
    })

    text = _final_text(result)
    try:
        parsed = parser.parse(text)
        print(parsed.model_dump())   # ✅ structured dict
    except Exception as e:
        print("⚠️ Parse failed. Raw output:\n", text)
        print("Error:", e)

if __name__ == "__main__":
    main()
