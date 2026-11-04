"""Provider-neutral model request and streaming interface."""

from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class ModelRequest:
    """The minimal input required to start a model response."""

    input: str
    instructions: str
    query: str = ""


class ProviderOutputLimit(Exception):
    """The provider exhausted its output budget after continuation attempts."""


class ModelProvider(Protocol):
    """Stream text deltas from a model and release provider resources."""

    def stream(self, request: ModelRequest) -> AsyncGenerator[str, None]:
        """Yield text deltas for one request."""

        ...

    async def close(self) -> None:
        """Release persistent network resources."""

        ...
