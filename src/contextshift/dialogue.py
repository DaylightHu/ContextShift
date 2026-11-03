"""In-memory history of text the user actually entered."""

from collections import deque
from dataclasses import dataclass
from typing import Literal

DialogueRole = Literal["user", "candidate", "interviewer"]
_ROLE_LABELS = {"user": "用户", "candidate": "候选人", "interviewer": "面试官"}
DEFAULT_HISTORY_MAX_TURNS = 64
DEFAULT_HISTORY_MAX_CHARACTERS = 12_000


@dataclass(frozen=True, slots=True)
class DialogueEntry:
    """One actual input with its speaker; never a generated suggestion."""

    text: str
    role: DialogueRole


class DialogueHistory:
    """Build bounded model context from actual user input only."""

    def __init__(
        self,
        *,
        max_turns: int = DEFAULT_HISTORY_MAX_TURNS,
        max_characters: int = DEFAULT_HISTORY_MAX_CHARACTERS,
    ) -> None:
        if max_turns < 1:
            raise ValueError("max_turns must be at least 1")
        if max_characters < 1:
            raise ValueError("max_characters must be at least 1")

        self._entries: deque[DialogueEntry] = deque(maxlen=max_turns)
        self._max_characters = max_characters

    @property
    def entries(self) -> tuple[str, ...]:
        """Return the retained actual inputs in chronological order."""

        return tuple(entry.text for entry in self._entries)

    @property
    def turns(self) -> tuple[DialogueEntry, ...]:
        """Return actual inputs with speaker identities for live consumers."""

        return tuple(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def append(self, text: str, *, role: DialogueRole = "user") -> None:
        """Record one actual user input in memory."""

        if role not in _ROLE_LABELS:
            raise ValueError(f"Unknown dialogue role: {role!r}")
        normalized = text.strip()
        if normalized:
            self._entries.append(DialogueEntry(normalized, role))

    def clear(self) -> None:
        """Forget all retained user input."""

        self._entries.clear()

    def build_model_input(self, question: str, *, role: DialogueRole = "user") -> str:
        """Combine recent actual inputs with the current question."""

        if not self._entries and role == "user":
            return question

        selected: list[DialogueEntry] = []
        remaining = self._max_characters

        for entry in reversed(self._entries):
            if len(entry.text) <= remaining:
                selected.append(entry)
                remaining -= len(entry.text)
                continue

            if remaining > 0:
                selected.append(DialogueEntry(entry.text[-remaining:], entry.role))
            break

        selected.reverse()
        dialogue = "\n".join(
            f"{index}. {_ROLE_LABELS[entry.role]}：{entry.text}"
            for index, entry in enumerate(selected, start=1)
        )

        current = (
            f"面试官问题：{question}\n\n"
            "任务：结合真实对话，直接写出候选人此刻可以说的回答要点。"
            "不要写回答策略、表达建议或后续追问提示。只输出 1–3 行；"
            "不要虚构候选人的经历，也不要把模型内容当作已经说过的话。"
            if role == "interviewer"
            else f"当前问题：{question}"
        )

        return (
            "以下内容仅包含当前会话中真实输入或说过的文字，已标注说话人。"
            "模型之前生成的建议不属于对话历史。\n"
            "<actual_dialogue>\n"
            f"{dialogue}\n"
            "</actual_dialogue>\n\n"
            f"{current}"
        )
