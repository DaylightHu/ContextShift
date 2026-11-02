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
