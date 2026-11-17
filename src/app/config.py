from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    azure_openai_endpoint: str
    azure_openai_api_key: SecretStr
    azure_openai_deployment: str
    azure_openai_api_version: str = "2024-08-01-preview"

    azure_search_endpoint: str | None = None
    azure_search_api_key: SecretStr | None = None
    azure_search_index: str = "azureblob-index"
    azure_search_top_k: int = 3

    azure_storage_connection_string: SecretStr | None = None
    azure_storage_container: str = "product"

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: SecretStr | None = None
    qdrant_collection: str = "docs_collection"
    qdrant_top_k: int = 3
    docs_dir: str = "docs"

    llm_timeout_s: float = 30.0
    llm_max_retries: int = 3
    llm_base_delay_s: float = 1.0
    llm_max_delay_s: float = 8.0

    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
