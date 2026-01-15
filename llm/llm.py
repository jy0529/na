from typing import AsyncIterator, Protocol, List, Dict, Optional, Any

class LLMClient(Protocol):
    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: Optional[float] = None,
        **kwargs: Any
    ) -> str:
        """
        Args:
            messages: The messages to chat with.
            model: The model to use.
            temperature: The temperature to use.
            **kwargs: Additional keyword arguments.

        Returns:
            The response from the LLM.
        """
        ...
        
    async def stream_chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: Optional[float] = None,
        **kwargs: Any
    ) -> AsyncIterator[Any]: 
        """
        Args:
            messages: The messages to chat with.
            model: The model to use.
            temperature: The temperature to use.
            **kwargs: Additional keyword arguments.

        Returns:
            The stream of responses from the LLM as a generator.
        """
        ...