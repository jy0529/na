# Built-in Web Search Tool

# 使用 exa 的 api

from ..base import Tool, ToolMetadata, ToolParameter, ToolResult, ToolParameterType
from typing import List, Optional
from enum import Enum
from exa_py import Exa
from ...settings import LLMSettings
from dataclasses import dataclass

class ExaWebSearchType(Enum):
    FAST = "fast"
    AUTO = "auto"

@dataclass
class SearchResult:
    id: str
    title: str
    url: str
    publishedDate: Optional[str] = None
    author: Optional[str] = None
    image: Optional[str] = None
    favicon: Optional[str] = None


class WebSearchTool(Tool):
    def __init__(self):
        super().__init__()
        self.settings = LLMSettings()
    
    def get_metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="web_search",
            description="""
            Use Exa's web search API to search the web for information. The search type can be fast, auto.
            Discovers URLS with AI-powered relevance scoring.

            USE WHEN:
            - You need to search the web for information
            - You need to search the web for information with a specific search type (fast, auto)
            NOT FOR:
                - reading full pages of page content (use web_fetch after getting URLs)
            PARAMS:
                - query: The query to search for
                - search_type: The search type to use (fast, auto)
            RETURNS:
                - results: A list of search results
                    - id: The id of the search result
                    - title: The title of the search result
                    - url: The URL of the search result
                    - publishedDate: The published date of the search result (YYYY-MM-DD HH:MM:SS) UTC (if available)
                    - author: The author of the search result (if available)
                    - image: The image of the search result (if available)
                    - favicon: The favicon of the search result (if available)
            """,
            version="0.0.1",
            author="Nick",
        )
    
    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="query",
                description="The query to search for\n Example: 'What is the capital of France?'",
                type=ToolParameterType.STRING,
                required=True,
            ),
            ToolParameter(
                name="search_type",
                description="The search type to use (fast, auto)\n Example: 'fast'",
                type=ToolParameterType.ENUM, 
                enum=[search_type.value for search_type in ExaWebSearchType], 
                required=True,
                pattern=r"^(fast|auto)$",
            ),
        ]
        
    def exa_web_search(self, query: str, search_type: ExaWebSearchType) -> List[SearchResult]:
        api_key = self.settings.exa_api_key
        if not api_key:
            raise ValueError("EXA_API_KEY is not set")
        
        exa = Exa(api_key=api_key)

        response = exa.search(
            query=query,
            type=search_type.value,
        )
        
        results = []
        for result in response.results:
            # 安全地获取属性，如果不存在则使用默认值
            results.append(SearchResult(
                id=getattr(result, 'id', ''),
                title=getattr(result, 'title', ''),
                url=getattr(result, 'url', ''),
                publishedDate=getattr(result, 'published_date', getattr(result, 'publishedDate', '')),
                author=getattr(result, 'author', ''),
                image=getattr(result, 'image', ''),
                favicon=getattr(result, 'favicon', ''),
            ))
        return results
    
    async def execute_impl(self, **kwargs) -> ToolResult:
        try:
            query = kwargs.get("query")
            search_type = ExaWebSearchType(kwargs.get("search_type"))
            results = self.exa_web_search(query, search_type)
            return ToolResult(success=True, output=results)
        except Exception as e:
            return ToolResult(success=False, error=str(e), output=None)