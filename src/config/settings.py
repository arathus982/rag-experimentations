"""Pydantic Settings for all project configuration."""

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    """PostgreSQL + pgvector connection configuration."""

    model_config = SettingsConfigDict(env_prefix="POSTGRES_")

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

    model_config = SettingsConfigDict(env_prefix="CONFLUENCE_")

    base_url: str = "https://your-company.atlassian.net"
    api_token: SecretStr = SecretStr("changeme")
    username: str = "user@company.com"
    space_key: str = "PRODOCS"


class EmbeddingSettings(BaseSettings):
    """Embedding model hosting configuration."""

    model_config = SettingsConfigDict(env_prefix="EMBEDDING_")

    device: str = "cuda"
    batch_size: int = 32
    harrier_model_path: str = "microsoft/Harrier-OSS-v1-27B"
    qwen_model_path: str = "Qwen/Qwen3-Embedding-8B"
    bge_m3_model_path: str = "BAAI/bge-m3"
    quantize_harrier: bool = Field(
        default=False,
        description="Load Harrier-OSS-v1 in 4-bit quantization for 24GB GPUs",
    )


class LlmSettings(BaseSettings):
    """LLM configuration for entity extraction and RAGAS."""

    model_config = SettingsConfigDict(env_prefix="LLM_")

    model_path: str = "Qwen/Qwen2.5-7B-Instruct"
    device: str = "cuda"


class LangfuseSettings(BaseSettings):
    """Langfuse observability configuration."""

    model_config = SettingsConfigDict(env_prefix="LANGFUSE_")

    public_key: SecretStr = SecretStr("pk-lf-changeme")
    secret_key: SecretStr = SecretStr("sk-lf-changeme")
    host: str = "http://localhost:3000"


class AppSettings(BaseSettings):
    """Root settings aggregator. Composes all sub-settings."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    confluence: ConfluenceSettings = Field(default_factory=ConfluenceSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    llm: LlmSettings = Field(default_factory=LlmSettings)
    langfuse: LangfuseSettings = Field(default_factory=LangfuseSettings)
    data_dir: str = "data"
