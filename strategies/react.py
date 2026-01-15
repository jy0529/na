from typing import Any
from patterns.react import react_loop
from llm import get_tool_registry, WebFetchTool, WebSearchTool
import asyncio

async def react_strategy(user_query: str) -> Any:

    tools_to_register = [
        WebFetchTool, 
        WebSearchTool,
    ]
    tool_registry = get_tool_registry()
    for tool in tools_to_register:
        tool_registry.register_tool(tool)

    return await react_loop(user_query)

    
if __name__ == "__main__":
    user_query = "最近有什么电视剧好看？"
    result = asyncio.run(react_strategy(user_query))
    print(result)