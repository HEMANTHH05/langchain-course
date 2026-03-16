import re
from typing import Union
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import render_text_description
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.output_parsers import BaseOutputParser
from langchain_openai import ChatOpenAI
from langchain.tools import tool

load_dotenv()


class ReActSingleInputOutputParser(BaseOutputParser[Union[AgentAction, AgentFinish]]):
    """Parse ReAct-style LLM output into AgentAction or AgentFinish."""
    
    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        """Parse the LLM output."""
        # Check for Final Answer
        final_answer_match = re.search(r"Final Answer:\s*(.+)", text, re.DOTALL)
        if final_answer_match:
            return AgentFinish(
                return_values={"output": final_answer_match.group(1).strip()},
                log=text
            )
        
        # Check for Action
        action_match = re.search(r"Action:\s*(.+)", text)
        action_input_match = re.search(r"Action Input:\s*(.+)", text, re.DOTALL)
        
        if action_match and action_input_match:
            tool = action_match.group(1).strip()
            tool_input = action_input_match.group(1).strip()
            
            # Try to parse tool_input as JSON, otherwise use as string
            try:
                import json
                tool_input = json.loads(tool_input)
            except:
                # Remove quotes if present
                if (tool_input.startswith('"') and tool_input.endswith('"')) or \
                   (tool_input.startswith("'") and tool_input.endswith("'")):
                    tool_input = tool_input[1:-1]
            
            return AgentAction(
                tool=tool,
                tool_input=tool_input,
                log=text
            )
        
        # If we can't parse, assume it's a finish with the whole text
        return AgentFinish(
            return_values={"output": text.strip()},
            log=text
        )

def find_tool_by_name(tools: list, tool_name: str):
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"tool with name {tool_name} not found")
@tool
def get_text_length(text: str) -> int:
    """Return the length of the text by characters."""
    text = text.strip()                # safer trim
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        text = text[1:-1]
    return len(text)


template = """ 
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:
"""

tools = [get_text_length]
prompt = PromptTemplate.from_template(template).partial(
    tools=render_text_description(tools),
    tool_names=", ".join(t.name for t in tools),
)
llm = ChatOpenAI(temperature=0, stop=["\nObservation"])
output_parser = ReActSingleInputOutputParser()

# ✅ Correct LCEL wiring: Prompt → LLM → Parser
chain = prompt | llm | output_parser

agentstep: Union[AgentAction, AgentFinish] = chain.invoke({"input": "what is the length of 'DOG' in characters?"})
print(agentstep)

if isinstance(agentstep, AgentAction):
    tool_name = agentstep.tool
    tool_to_use = find_tool_by_name(tools, tool_name)
    tool_input = agentstep.tool_input
    
    # Handle tool_input - it might be a string or dict
    if isinstance(tool_input, str):
        observation = tool_to_use.invoke({"text": tool_input})
    else:
        observation = tool_to_use.invoke(tool_input)
    print(f"{observation = }")
elif isinstance(agentstep, AgentFinish):
    print(f"Final answer: {agentstep.return_values['output']}")

