"""Role-aware real history and bounded model context."""

import pytest

from contextshift.dialogue import DialogueHistory


def test_interviewer_prompt_is_role_aware_even_without_history() -> None:
    history = DialogueHistory()
    assert history.build_model_input("plain question") == "plain question"
    prompt = history.build_model_input("介绍项目", role="interviewer")
    assert "面试官问题：介绍项目" in prompt
    assert "候选人" in prompt
    assert "1–3 行" in prompt
    assert history.entries == ()


def test_roles_survive_bounded_history_and_truncation() -> None:
    history = DialogueHistory(max_turns=2, max_characters=5)
    history.append("evicted", role="interviewer")
    history.append("older", role="interviewer")
    history.append("abcdef", role="candidate")
    prompt = history.build_model_input("next", role="interviewer")
    assert history.entries == ("older", "abcdef")
    assert "候选人：bcdef" in prompt
    assert "older" not in prompt
    assert "evicted" not in prompt
    history.clear()
    assert history.turns == ()


@pytest.mark.parametrize("limits", [{"max_turns": 0}, {"max_characters": 0}])
def test_history_rejects_empty_limits(limits: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        DialogueHistory(**limits)


def test_default_history_keeps_earlier_context_through_short_utterances() -> None:
    history = DialogueHistory()
    earlier = "我负责服务拆分，缓存更新失败时保留旧缓存。" * 120
    history.append(earlier, role="candidate")
    for _ in range(20):
        history.append("嗯。", role="candidate")
    prompt = history.build_model_input("你刚才说的失败策略是什么？", role="interviewer")
    assert f"候选人：{earlier}" in prompt
    assert len(history) == 21


def test_context_uses_remaining_budget_for_part_of_older_entry() -> None:
    history = DialogueHistory(max_turns=3, max_characters=7)
    history.append("excluded", role="candidate")
    history.append("abcdef", role="candidate")
    history.append("WXYZ", role="interviewer")
    prompt = history.build_model_input("next", role="interviewer")
    assert "候选人：def\n2. 面试官：WXYZ" in prompt
    assert "excluded" not in prompt
    assert "abcdef" not in prompt
    assert history.entries == ("excluded", "abcdef", "WXYZ")


@pytest.mark.parametrize("budget", range(1, 13))
def test_context_budget_matches_suffix_of_actual_dialogue(budget: int) -> None:
    texts = ("abc", "defg", "hijkl")
    history = DialogueHistory(max_turns=3, max_characters=budget)
    for text in texts:
        history.append(text, role="candidate")
    prompt = history.build_model_input("question", role="interviewer")
    dialogue = prompt.split("<actual_dialogue>\n", 1)[1].split(
        "\n</actual_dialogue>", 1
    )[0]
    selected = "".join(line.split("候选人：", 1)[1] for line in dialogue.splitlines())
    assert selected == "".join(texts)[-budget:]


def test_enlarged_defaults_still_bound_retained_turns() -> None:
    history = DialogueHistory()
    for index in range(70):
        history.append(f"actual-{index}")
    assert len(history) == 64
    assert history.entries[0] == "actual-6"
    assert history.entries[-1] == "actual-69"
