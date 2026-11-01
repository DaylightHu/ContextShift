"""Provider-neutral model request and streaming interface."""

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class ModelRequest:
    """The minimal input required to start a model response."""

    input: str
    instructions: str


class ModelProvider(Protocol):
    """Stream text deltas from a model and release provider resources."""

    def stream(self, request: ModelRequest) -> AsyncIterator[str]:
        """Yield text deltas for one request."""

        ...

    async def close(self) -> None:
        """Release persistent network resources."""

        ...
