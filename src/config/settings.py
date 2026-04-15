"""Pydantic Settings for all project configuration."""

from typing import Optional

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    """PostgreSQL + pgvector connection configuration."""

    model_config = SettingsConfigDict(env_prefix="POSTGRES_", env_file=".env", extra="ignore")

    host: str = "localhost"
    port: int = 5432
    user: str = "raguser"
    password: SecretStr = SecretStr("ragpass")
    db: str = "ragdb"

    @property
    def connection_url(self) -> str:
        """SQLAlchemy-compatible connection string."""
        pwd = self.password.get_secret_value()
        return f"postgresql://{self.user}:{pwd}@{self.host}:{self.port}/{self.db}"

    @property
    def async_connection_url(self) -> str:
        """Async connection string for asyncpg."""
        pwd = self.password.get_secret_value()
        return f"postgresql+asyncpg://{self.user}:{pwd}" f"@{self.host}:{self.port}/{self.db}"


class ConfluenceSettings(BaseSettings):
    """Confluence API configuration."""

    model_config = SettingsConfigDict(env_prefix="CONFLUENCE_", env_file=".env", extra="ignore")

    base_url: str = "https://your-company.atlassian.net"
    api_token: SecretStr = SecretStr("changeme")
    username: str = "user@company.com"
    space_key: str = "PRODOCS"
    root_page_id: Optional[str] = Field(
        default=None,
        description="If set, only this page and its descendants are downloaded. Downloads the whole space if unset.",
    )


class EmbeddingSettings(BaseSettings):
    """Embedding model hosting configuration."""

    model_config = SettingsConfigDict(env_prefix="EMBEDDING_", env_file=".env", extra="ignore")

    device: str = "cuda"
    batch_size: int = 4
    harrier_model_path: str = "microsoft/Harrier-OSS-v1-27B"
    qwen_model_path: str = "Qwen/Qwen3-Embedding-8B"
    bge_m3_model_path: str = "BAAI/bge-m3"
    quantize_harrier: bool = Field(
        default=False,
        description="Load Harrier-OSS-v1 in 4-bit quantization for 24GB GPUs",
    )


class LlmSettings(BaseSettings):
    """LLM configuration for entity extraction and RAGAS."""

    model_config = SettingsConfigDict(env_prefix="LLM_", env_file=".env", extra="ignore")

    model_path: str = "Qwen/Qwen2.5-7B-Instruct"
    device: str = "cuda"


class RerankerSettings(BaseSettings):
    """Reranker model configuration for ensemble retrieval."""

    model_config = SettingsConfigDict(env_prefix="RERANKER_", env_file=".env", extra="ignore")

    bge_model_path: str = "BAAI/bge-reranker-v2-m3"
    qwen3_model_path: str = "Qwen/Qwen3-Reranker-4B"
    device: str = "mps"
    initial_top_k: int = 20
    final_top_k: int = 10


class OpenRouterSettings(BaseSettings):
    """OpenRouter API configuration for dataset generation."""

    model_config = SettingsConfigDict(env_prefix="OPENROUTER_", env_file=".env", extra="ignore")

    api_key: SecretStr = SecretStr("changeme")
    model: str = "google/gemini-2.5-flash"
    eval_model: str = "google/gemini-3-flash-preview"
    base_url: str = "https://openrouter.ai/api/v1"


class LangfuseSettings(BaseSettings):
    """Langfuse observability configuration."""

    model_config = SettingsConfigDict(env_prefix="LANGFUSE_", env_file=".env", extra="ignore")

    public_key: SecretStr = SecretStr("pk-lf-changeme")
    secret_key: SecretStr = SecretStr("sk-lf-changeme")
    host: str = "http://localhost:3000"


class AppSettings(BaseSettings):
    """Root settings aggregator. Composes all sub-settings."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    confluence: ConfluenceSettings = Field(default_factory=ConfluenceSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    llm: LlmSettings = Field(default_factory=LlmSettings)
    langfuse: LangfuseSettings = Field(default_factory=LangfuseSettings)
    openrouter: OpenRouterSettings = Field(default_factory=OpenRouterSettings)
    reranker: RerankerSettings = Field(default_factory=RerankerSettings)
    data_dir: str = "data"
