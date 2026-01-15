from openai import OpenAI, AsyncOpenAI
from llm.llm import LLMClient
from llm.settings import LLMSettings

from typing import List, Dict, Optional, Any, AsyncIterator

class OpenAIClient(LLMClient):
    def __init__(self,
        llm_settings: LLMSettings,
        **kwargs: Any
    ):
        self.client = OpenAI(
            api_key=llm_settings.api_key,
            base_url=llm_settings.api_base_url,
            **kwargs
        )

        self.async_client = AsyncOpenAI(
            api_key=llm_settings.api_key,
            base_url=llm_settings.api_base_url,
            **kwargs
        )


    def chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: Optional[float] = None,
        **kwargs: Any
    ) -> str:
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            **kwargs
        )

        if not response.choices or len(response.choices) == 0:
            raise ValueError("No choices in response")
        
        message = response.choices[0].message
        if not message:
            raise ValueError("No message in response")
        
        return message

        
    async def stream_chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: Optional[float] = None,
        **kwargs: Any
    ) -> AsyncIterator[Any]:
        response = await self.async_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            stream=True,  # 确保启用流式输出
            **kwargs
        )

        async for chunk in response:
            yield chunk
    
