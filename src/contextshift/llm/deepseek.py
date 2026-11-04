"""DeepSeek implementation of the model provider interface."""

from collections.abc import AsyncGenerator

from openai import AsyncOpenAI

from contextshift.code_explorer import CodeExplorer
from contextshift.config import Settings
from contextshift.llm.base import ModelRequest, ProviderOutputLimit
from contextshift.repository import RepositoryIndex, WorkspaceIndex


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

    async def prepare(self) -> None:
        """Check authentication/model availability and warm the HTTP connection."""

        if self._closed:
            raise RuntimeError("DeepSeekProvider is closed")
        models = await self._client.models.list(timeout=10)
        if self._settings.model not in {model.id for model in models.data}:
            raise ValueError(f"云端未提供所选模型：{self._settings.model}")

    async def stream(self, request: ModelRequest) -> AsyncGenerator[str, None]:
        """Stream the answer, continuing when the provider reaches its token cap."""

        if self._closed:
            raise RuntimeError("DeepSeekProvider is closed")

        answer = ""
        input_data: str | list[dict[str, str]] = request.input
        for attempt in range(3):
            stream = await self._client.responses.create(
                model=self._settings.model,
                instructions=request.instructions,
                input=input_data,
                reasoning={"effort": self._settings.reasoning_effort},
                max_output_tokens=max(
                    self._settings.max_output_tokens,
                    2048 if attempt else 0,
                ),
                stream=True,
            )
            incomplete = False
            terminal = False
            try:
                async for event in stream:
                    if event.type == "response.output_text.delta":
                        answer += event.delta
                        yield event.delta
                    elif event.type == "response.incomplete":
                        terminal = True
                        details = getattr(event.response, "incomplete_details", None)
                        if getattr(details, "reason", None) != "max_output_tokens":
                            raise RuntimeError("云端未能完成回答")
                        incomplete = True
                    elif event.type == "response.completed":
                        terminal = True
                    elif event.type == "response.failed":
                        raise RuntimeError("云端生成失败")
            finally:
                await stream.close()

            if not terminal:
                raise RuntimeError("云端回复意外中断")
            if not incomplete:
                return
            if attempt == 2:
                raise ProviderOutputLimit("云端输出上限")
            input_data = [
                {"role": "user", "content": request.input},
                {"role": "assistant", "content": answer},
                {
                    "role": "user",
                    "content": "请直接续写尚未完成的回答，不要重复已经输出的内容。",
                },
            ]

    async def explore_code(
        self,
        index: RepositoryIndex | WorkspaceIndex,
        query: str,
        dialogue: tuple[str, ...] = (),
    ):
        """Let the model search and read only selected workspace snapshots."""
        if self._closed:
            raise RuntimeError("DeepSeekProvider is closed")
        return await CodeExplorer(index).explore(
            self._client, model=self._settings.model, query=query, dialogue=dialogue
        )

    async def close(self) -> None:
        """Close the reusable API client once."""

        if self._closed:
            return

        self._closed = True
        await self._client.close()
