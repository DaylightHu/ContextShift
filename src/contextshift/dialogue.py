"""In-memory history of text the user actually entered."""

from collections import deque


class DialogueHistory:
    """Build bounded model context from actual user input only."""

    def __init__(
        self,
        *,
        max_turns: int = 8,
        max_characters: int = 2000,
    ) -> None:
        if max_turns < 1:
            raise ValueError("max_turns must be at least 1")
        if max_characters < 1:
            raise ValueError("max_characters must be at least 1")

        self._entries: deque[str] = deque(maxlen=max_turns)
        self._max_characters = max_characters

    @property
    def entries(self) -> tuple[str, ...]:
        """Return the retained actual inputs in chronological order."""

        return tuple(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def append(self, text: str) -> None:
        """Record one actual user input in memory."""

        normalized = text.strip()
        if normalized:
            self._entries.append(normalized)

    def clear(self) -> None:
        """Forget all retained user input."""

        self._entries.clear()

    def build_model_input(self, question: str) -> str:
        """Combine recent actual inputs with the current question."""

        if not self._entries:
            return question

        selected: list[str] = []
        remaining = self._max_characters

        for entry in reversed(self._entries):
            if len(entry) <= remaining:
                selected.append(entry)
                remaining -= len(entry)
                continue

            if not selected and remaining > 0:
                selected.append(entry[-remaining:])
            break

        selected.reverse()
        dialogue = "\n".join(
            f"{index}. {entry}" for index, entry in enumerate(selected, start=1)
        )

        return (
            "以下内容仅包含用户在当前会话中真实输入过的文字。"
            "模型之前生成的建议不属于对话历史。\n"
            "<actual_dialogue>\n"
            f"{dialogue}\n"
            "</actual_dialogue>\n\n"
            f"当前问题：{question}"
        )
