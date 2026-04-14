"""Langfuse observability integration for LlamaIndex."""

from langfuse import Langfuse
from llama_index.core import Settings
from llama_index.core.callbacks import CallbackManager

from src.config.settings import LangfuseSettings


class LangfuseObservability:
    """Initializes Langfuse tracing for all LlamaIndex operations."""

    def __init__(self, settings: LangfuseSettings) -> None:
        self._settings = settings
        self._langfuse: Langfuse | None = None

    def setup(self) -> None:
        """Register Langfuse callback handler with LlamaIndex global Settings."""
        from llama_index.callbacks.langfuse import (  # type: ignore[attr-defined]
            LangfuseCallbackHandler,
        )

        handler = LangfuseCallbackHandler(
            public_key=self._settings.public_key.get_secret_value(),
            secret_key=self._settings.secret_key.get_secret_value(),
            host=self._settings.host,
        )

        # Register with LlamaIndex's global callback manager
        Settings.callback_manager = CallbackManager([handler])

        self._langfuse = Langfuse(
            public_key=self._settings.public_key.get_secret_value(),
            secret_key=self._settings.secret_key.get_secret_value(),
            host=self._settings.host,
        )

    def flush(self) -> None:
        """Ensure all traces are sent before shutdown."""
        if self._langfuse:
            self._langfuse.flush()
