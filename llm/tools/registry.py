from typing import List, Dict, Any
from .base import Tool

class ToolRegistry:
    def __init__(self):
        self.tools = []
        self.tool_instances = {}
        
        
    def register_tool(self, tool: Tool):
        self.tools.append(tool)
        
    def get_tool(self, name: str) -> Tool:
        if name not in self.tool_instances:
            for tool in self.tools:
                if tool.get_metadata().name == name:
                    self.tool_instances[name] = tool()
                    break
            else:
                raise ValueError(f"Tool {name} not found")
        return self.tool_instances[name]
        
    def get_all_tools(self) -> List[Tool]:
        return self.tools
    
    def get_all_tool_instances(self) -> List[Tool]:
        """获取所有工具的实例"""
        instances = []
        for tool_class in self.tools:
            tool_name = tool_class().get_metadata().name
            if tool_name not in self.tool_instances:
                self.tool_instances[tool_name] = tool_class()
            instances.append(self.tool_instances[tool_name])
        return instances
    
    def to_openai_tools(self) -> List[Dict[str, Any]]:
        """
        将所有注册的工具转换为 OpenAI 函数调用格式
        
        Returns:
            OpenAI 工具列表，格式如下：
            [
                {
                    "type": "function",
                    "function": {...}
                },
                ...
            ]
        """
        tools = []
        for tool_instance in self.get_all_tool_instances():
            tools.append(tool_instance.to_openai_function())
        return tools

        
_tool_registry = None

def get_tool_registry() -> ToolRegistry:
    global _tool_registry
    if _tool_registry is None:
        _tool_registry = ToolRegistry()
    return _tool_registry