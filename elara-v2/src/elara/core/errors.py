"""Exception hierarchy. Every error ELARA raises deliberately derives from ElaraError."""


class ElaraError(Exception):
    """Base class for expected, reportable failures."""


class ConfigError(ElaraError):
    pass


class DatabaseError(ElaraError):
    pass


class ValidationError(ElaraError):
    pass


class ProviderError(ElaraError):
    """An LLM provider failed. ``retryable`` tells the retry layer what to do."""

    def __init__(self, message: str, *, retryable: bool = False, status: int | None = None):
        super().__init__(message)
        self.retryable = retryable
        self.status = status


class ProviderUnavailable(ProviderError):
    """No usable provider is configured."""


class ToolError(ElaraError):
    """A tool failed while running. The message is shown to the user / LLM."""


class PermissionDenied(ElaraError):
    pass


class PathNotAllowed(PermissionDenied):
    pass


class SourceError(ElaraError):
    """An external data source (research API, web) failed.

    ``kind``: unreachable | timeout | rate_limited | blocked | http_error | invalid_response.
    """

    def __init__(self, message: str, kind: str = "http_error"):
        super().__init__(message)
        self.kind = kind
