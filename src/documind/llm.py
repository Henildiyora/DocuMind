"""Thin wrapper around Ollama for local Gemma / any chat model."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from .config import Config


class LLMError(RuntimeError):
    """Raised when the local LLM isn't reachable or the model is missing."""


@dataclass
class OllamaClient:
    """Thin synchronous wrapper around `ollama` with streaming support."""

    cfg: Config

    def _client(self):
        try:
            from ollama import Client
        except ImportError as exc:  # pragma: no cover
            raise LLMError(
                "The `ollama` python package is required. Install with: pip install ollama"
            ) from exc
        return Client(host=self.cfg.ollama_base_url)

    def ping(self) -> bool:
        """Return True if the Ollama daemon responds."""
        try:
            self._client().list()
            return True
        except Exception:
            return False

    def model_available(self, model: str | None = None) -> bool:
        """Return True if the named model is pulled locally."""
        target = (model or self.cfg.model).strip()
        try:
            data = self._client().list()
        except Exception:
            return False
        # `ollama-python` returns either a pydantic object or a plain dict
        models = getattr(data, "models", None) or data.get("models", [])  # type: ignore[union-attr]
        for m in models:
            name = getattr(m, "model", None) or (m.get("model") if isinstance(m, dict) else None)
            if not name:
                name = getattr(m, "name", None) or (m.get("name") if isinstance(m, dict) else None)
            if not name:
                continue
            if name == target or name.split(":")[0] == target.split(":")[0]:
                return True
        return False

    def pull(self, model: str | None = None) -> None:
        """Pull a model (blocking). Useful from `documind doctor --pull`."""
        target = model or self.cfg.model
        try:
            self._client().pull(target)
        except Exception as exc:
            raise LLMError(f"Failed to pull model {target!r}: {exc}") from exc

    def chat_stream(
        self,
        messages: list[dict],
        model: str | None = None,
    ) -> Iterator[str]:
        """Stream response tokens from the chat endpoint."""
        target = model or self.cfg.model
        options = {
            "temperature": self.cfg.llm_temperature,
            "num_ctx": self.cfg.llm_num_ctx,
        }
        try:
            client = self._client()
            stream = client.chat(
                model=target,
                messages=messages,
                stream=True,
                options=options,
            )
        except Exception as exc:
            raise LLMError(
                f"Could not reach Ollama at {self.cfg.ollama_base_url}. "
                f"Is `ollama serve` running? ({exc})"
            ) from exc

        try:
            for part in stream:
                msg = getattr(part, "message", None) or (
                    part.get("message") if isinstance(part, dict) else None
                )
                if not msg:
                    continue
                content = getattr(msg, "content", None) or (
                    msg.get("content") if isinstance(msg, dict) else ""
                )
                if content:
                    yield content
        except Exception as exc:
            raise LLMError(f"LLM streaming failed: {exc}") from exc

    def chat(self, messages: list[dict], model: str | None = None) -> str:
        """Non-streaming convenience wrapper."""
        return "".join(self.chat_stream(messages, model=model))
