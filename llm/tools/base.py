### Base Tool Class ###

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Any, Union, Dict
from enum import Enum
import json

@dataclass
class ToolMetadata:
    name: str
    description: str
    version: Optional[str] = None
    author: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "author": self.author,
        }
    
    def _to_json(self) -> str:
        return json.dumps(self.to_dict())


class ToolParameterType(Enum):
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"
    FILE = "file"  # For file paths or file content
    ENUM = "enum"  # Enum type (should use STRING with enum list)
    
    
@dataclass
class ToolParameter:
    name: str
    description: str
    required: Optional[bool] = False
    type: Optional[ToolParameterType] = None
    # if type is enum, enum is the list of allowed values
    enum: Optional[List[str]] = None
    # minimum value for number
    minimum: Optional[Union[int, float]] = None
    # maximum value for number
    maximum: Optional[Union[int, float]] = None
    # default value
    default: Optional[Any] = None
    # pattern for validation
    pattern: Optional[str] = None
    # for array type, items schema
    items: Optional[Dict[str, Any]] = None
    
    def to_openai_schema(self) -> Dict[str, Any]:
        """
        转换为 OpenAI 函数调用的参数 schema
        
        Returns:
            OpenAI 函数参数 schema 字典
        """
        # 如果类型是 ENUM，转换为 string 类型
        param_type = self.type
        if param_type == ToolParameterType.ENUM:
            param_type = ToolParameterType.STRING
        
        schema: Dict[str, Any] = {
            "type": param_type.value if param_type else "string",
            "description": self.description,
        }
        
        # 处理 enum（ENUM 类型或带 enum 列表的 STRING 类型）
        if self.enum:
            schema["enum"] = self.enum
        
        # 处理数字类型的约束
        if param_type in [ToolParameterType.INTEGER, ToolParameterType.FLOAT]:
            if self.minimum is not None:
                schema["minimum"] = self.minimum
            if self.maximum is not None:
                schema["maximum"] = self.maximum
        
        # 处理 pattern（字符串验证）
        if self.pattern:
            schema["pattern"] = self.pattern
        
        # 处理 array 类型的 items
        if param_type == ToolParameterType.ARRAY:
            if self.items:
                schema["items"] = self.items
            else:
                # 默认 items 为 string
                schema["items"] = {"type": "string"}
        
        return schema


@dataclass
class ToolResult:
    success: bool
    error: Optional[str] = None
    output: Optional[Any] = None

    
class Tool(ABC):
    def __init__(self):
        self.metadata = self.get_metadata()
        self.parameters = self.get_parameters()

    @abstractmethod
    def get_metadata(self) -> ToolMetadata:
        pass

    @abstractmethod
    def get_parameters(self) -> List[ToolParameter]:
        pass

    @abstractmethod
    async def execute_impl(self, **kwargs) -> ToolResult:
        pass

    async def execute(self, **kwargs) -> ToolResult:
        try:
            kwargs = self._cohere_parameters(**kwargs)
            return await self.execute_impl(**kwargs)
        except Exception as e:
            return ToolResult(success=False, error=str(e), output=None)
    
    def to_openai_function(self) -> Dict[str, Any]:
        """
        转换为 OpenAI 函数调用格式
        
        Returns:
            OpenAI 函数调用格式的字典，格式如下：
            {
                "type": "function",
                "function": {
                    "name": "tool_name",
                    "description": "tool description",
                    "parameters": {
                        "type": "object",
                        "properties": {...},
                        "required": [...]
                    }
                }
            }
        """
        # 构建 properties
        properties: Dict[str, Any] = {}
        required: List[str] = []
        
        for param in self.parameters:
            properties[param.name] = param.to_openai_schema()
            if param.required:
                required.append(param.name)
        
        # 构建完整的函数定义
        function_def = {
            "name": self.metadata.name,
            "description": self.metadata.description,
            "parameters": {
                "type": "object",
                "properties": properties,
            }
        }
        
        # 只有当有必需参数时才添加 required
        if required:
            function_def["parameters"]["required"] = required
        
        return {
            "type": "function",
            "function": function_def
        }

    def _cohere_parameters(self, **kwargs) -> Dict[str, Any]:
        out: dict[str, Any] = dict[str, Any](kwargs)
        spec = {p.name: p for p in self.parameters}

        for name, param in spec.items():
            if name not in out:
                continue
            val = out[name]
            
            try:
                if param.type == ToolParameterType.INTEGER:
                    if isinstance(val, float) and val.is_integer():
                        out[name] = int(val)
                    elif isinstance(val, str):
                        out[name] = int(val)
                    
                    if isinstance(out[name], int):
                        if param.minimum is not None and out[name] < param.minimum:
                            out[name] = param.minimum
                        if param.maximum is not None and out[name] > param.maximum:
                            out[name] = param.maximum
                    
                elif param.type == ToolParameterType.FLOAT:
                    if isinstance(val, str):
                        out[name] = float(val)
                    elif isinstance(val, int):
                        out[name] = float(val)
                    if isinstance(out[name], float):
                        if param.minimum is not None and out[name] < param.minimum:
                            out[name] = float(param.minimum)
                        if param.maximum is not None and out[name] > param.maximum:
                            out[name] = float(param.maximum)
                    
                elif param.type == ToolParameterType.BOOLEAN:
                    if isinstance(val, str):
                        s = val.strip().lower()
                        if s in ["true", "1", "yes", "y"]:
                            out[name] = True
                        elif s in ["false", "0", "no", "n"]:
                            out[name] = False
                            
            except Exception:
                pass
            
        return out