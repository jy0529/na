from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

# 获取项目根目录（llm/settings.py -> llm/ -> 根目录）
ROOT_DIR = Path(__file__).parent.parent

class LLMSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",  # 指向根目录的 .env
        env_file_encoding="utf-8",
        env_prefix="NA_",
        case_sensitive=False,
    )

    api_key: str = Field(description="The API key for the LLM")
    api_base_url: str = Field(description="The base URL for the LLM API")
    exa_api_key: str = Field(description="The API key for the Exa API")
