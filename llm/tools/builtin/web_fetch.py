""" Built-in Web Fetch Tool """

from ..base import Tool, ToolMetadata, ToolParameter, ToolResult, ToolParameterType
from typing import List, Optional
from dataclasses import dataclass
from ...settings import LLMSettings
from exa_py import Exa

@dataclass
class WebFetchResult:
    id: str
    title: str
    url: str
    text: str
    author: Optional[str] = None
    summary: Optional[str] = None
    image: Optional[str] = None


class WebFetchTool(Tool):
    def __init__(self):
        super().__init__()
        self.settings = LLMSettings()
        self.exa = Exa(api_key=self.settings.exa_api_key)
    
    def get_metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="web_fetch",
            description="""
            Use Exa's web fetch API to fetch the content of a web page or multiple web pages.
            
            USE WHEN:
            - You need to fetch the content of a single web page
            - You need to fetch the content of multiple web pages
           
            PARAMS:
                - urls: The URLs of the web pages to fetch, allow batch or single URL, always list of strings
            RETURNS:
                - results: A list of web fetch results
                    - id: The id of the web fetch result
                    - title: The title of the web fetch result
                    - url: The URL of the web fetch result
                    - author: The author of the web fetch result (if available)
                    - text: The text of the web fetch result
                    - summary: The summary of the web fetch result (if available)
                    - image: The image of the web fetch result (if available)
            """,
            version="0.0.1",
            author="Nick",
        )
        
    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="urls",
                description="The URLs of the web pages to fetch\n Batch Example: ['https://www.google.com', 'https://www.yahoo.com']\n Single Example: 'https://www.google.com'",
                type=ToolParameterType.ARRAY,
                required=True,
            ),
        ]
        
        
    def fetch_by_exa(self, **kwargs) -> List[WebFetchResult]:
        try:
            urls = kwargs.get("urls")
            response = self.exa.get_contents(urls, livecrawl="fallback", text=True)
            results = []
            for result in response.results:
                # 安全地获取属性，如果不存在则使用默认值
                results.append(WebFetchResult(
                    id=getattr(result, 'id', ''),
                    title=getattr(result, 'title', ''),
                    url=getattr(result, 'url', ''),
                    text=getattr(result, 'text', ''),
                    author=getattr(result, 'author', None),
                    summary=getattr(result, 'summary', None),
                    image=getattr(result, 'image', None),
                ))
            return results
        except Exception as e:
            raise Exception(f"Error fetching web pages: {e}")

    async def execute_impl(self, **kwargs) -> ToolResult:
        try:
            urls = kwargs.get("urls")
            if isinstance(urls, str):
                urls = [urls]
            results = self.fetch_by_exa(urls=urls)
            return ToolResult(success=True, output=results)
        except Exception as e:
            return ToolResult(success=False, error=str(e), output=None)