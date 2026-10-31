class HarnessError(Exception):
    """Base class for harness errors."""


class SandboxUnavailableError(HarnessError):
    """Raised when the sandbox runtime or image is not ready."""


class ModelUnavailableError(HarnessError):
    """Raised when the configured LLM endpoint or model cannot be reached."""
