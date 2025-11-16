import json
from dotenv import load_dotenv
from tavily import TavilyClient

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import JsonOutputParser  # not Pydantic

load_dotenv()

tavily = TavilyClient()
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# 1) A pre-step that pulls web results (no agent/tool-calls; just Python)
def fetch_web(input_dict):
    q = input_dict["query"]
    res = tavily.search(query=q, max_results=5)
    items = [{"title": r.get("title"), "url": r.get("url")} for r in res.get("results", [])[:5]]
    return {"query": q, "web_results": items}

fetcher = RunnableLambda(fetch_web)

# 2) Prompt asks the model to produce strict JSON (schema described in text)
prompt = PromptTemplate.from_template(
    """You are a helpful job-search assistant.
You are given web results (title, url). Use only this data.

User query:
{query}

Web results (JSON):
{web_results}

Return ONLY valid JSON with this shape:
{{
  "answer": "one-paragraph summary of the 3 jobs",
  "jobs": [
    {{"title": "...", "url": "..."}},
    {{"title": "...", "url": "..."}},
    {{"title": "...", "url": "..."}}
  ]
}}"""
)

# 3) Parse JSON (no Pydantic)
json_parser = JsonOutputParser()

# 4) LCEL chain
chain = fetcher | prompt | llm | json_parser

# Run
out = chain.invoke({"query": "AI Engineer LangChain Bay Area site:linkedin.com/jobs"})
print(out)          # -> dict with "answer" and "jobs"
