"""DeepSeek implementation of the model provider interface."""

from collections.abc import AsyncGenerator

from openai import AsyncOpenAI

from contextshift.config import Settings
from contextshift.llm.base import ModelRequest


class DeepSeekProvider:
    """Stream DeepSeek responses through one reusable API client."""

    def __init__(
        self,
        settings: Settings,
        *,
        client: AsyncOpenAI | None = None,
    ) -> None:
        self._settings = settings
        self._client = client or AsyncOpenAI(
            api_key=settings.deepseek_api_key.get_secret_value(),
            base_url=settings.deepseek_base_url,
        )
        self._closed = False

    async def stream(self, request: ModelRequest) -> AsyncGenerator[str, None]:
        """Yield only text deltas and always close the response stream."""

        if self._closed:
            raise RuntimeError("DeepSeekProvider is closed")

        stream = await self._client.responses.create(
            model=self._settings.model,
            instructions=request.instructions,
            input=request.input,
            reasoning={"effort": self._settings.reasoning_effort},
            max_output_tokens=self._settings.max_output_tokens,
            stream=True,
        )

        try:
            async for event in stream:
                if event.type == "response.output_text.delta":
                    yield event.delta
        finally:
            await stream.close()

    async def close(self) -> None:
        """Close the reusable API client once."""

        if self._closed:
            return

        self._closed = True
        await self._client.close()
