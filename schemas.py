from typing import List

from pydantic import BaseModel, Field


class Source(BaseModel):
    """Schema for source used by the agent"""
    url: str = Field(description= "The url of the source")

class AgentResponse(BaseModel):
    """Schema for the agent response with answer and sources"""

    answer: str= Field(description= "The agent answers to the query")
    sources: List[Source] = Field(
        default_factory= list, description= "List of the sources used to generate the answers"
    )