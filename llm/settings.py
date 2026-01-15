from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class LLMSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file = ".env",
        env_file_encoding = "utf-8",
        env_prefix="NA_LLM_",
        case_sensitive = False,
    )

    api_key: str = Field(description="The API key for the LLM")
    api_base_url: str = Field(description="The base URL for the LLM API")
