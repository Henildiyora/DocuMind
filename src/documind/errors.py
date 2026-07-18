"""User-facing DocuMind errors with plain-English fix hints.

These replace raw stack traces for common failure modes. Print via
``format_user_error`` / ``print_user_error``; only re-raise with a
traceback when ``--debug`` / ``-v`` is set.
"""

from __future__ import annotations

from .ollama_daemon import install_hint


class DocuMindError(Exception):
    """Base class for friendly, actionable CLI errors."""

    exit_code: int = 1

    def user_message(self) -> str:
        """One to three lines: what went wrong + the exact next command."""
        return str(self)


class OllamaNotInstalled(DocuMindError):
    """The ``ollama`` binary is not on PATH."""

    exit_code = 2

    def user_message(self) -> str:
        return (
            "Ollama isn't installed on this machine.\n"
            f"Install it with: {install_hint()}\n"
            "Then retry. (Tip: `documind search` works without any model.)"
        )


class OllamaNotRunning(DocuMindError):
    """Ollama is installed but the daemon is not reachable."""

    exit_code = 2

    def user_message(self) -> str:
        return (
            "Couldn't start or reach the Ollama daemon.\n"
            "Try: ollama serve\n"
            "Or run: documind doctor"
        )


class ModelNotPulled(DocuMindError):
    """The configured model tag is not present locally."""

    exit_code = 2

    def __init__(self, model: str) -> None:
        self.model = model
        super().__init__(model)

    def user_message(self) -> str:
        return (
            f"Model {self.model!r} isn't pulled yet.\n"
            f"Run: documind setup --pull\n"
            f"Or: ollama pull {self.model}"
        )


class ModelPullFailed(DocuMindError):
    """``ollama pull`` failed (network, disk, etc.)."""

    exit_code = 2

    def __init__(self, model: str, detail: str = "") -> None:
        self.model = model
        self.detail = detail
        super().__init__(model)

    def user_message(self) -> str:
        extra = f" ({self.detail})" if self.detail else ""
        return (
            f"Failed to pull model {self.model!r}{extra}.\n"
            f"Retry: ollama pull {self.model}\n"
            "Check disk space and network, then try again."
        )


class IndexMissing(DocuMindError):
    """No project index under ``.documind/``."""

    exit_code = 1

    def user_message(self) -> str:
        return (
            "No index found for this project.\n"
            "Run: documind index"
        )


def format_user_error(exc: DocuMindError) -> str:
    """Return the plain-English message for ``exc``."""
    return exc.user_message()
