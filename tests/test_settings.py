import pytest

from segmentation_web.settings import Settings


@pytest.mark.parametrize("value", ["false", "0", "no", "off", " FALSE "])
def test_llm_enabled_false_values(monkeypatch, value):
    monkeypatch.setenv("LLM_ENABLED", value)
    monkeypatch.setenv("LLM_API_KEY", "configured-key")

    settings = Settings.from_env()

    assert settings.llm_enabled is False
    assert settings.llm_configured is False


@pytest.mark.parametrize("value", ["true", "1", "yes", "on", " TRUE "])
def test_llm_enabled_true_values(monkeypatch, value):
    monkeypatch.setenv("LLM_ENABLED", value)
    monkeypatch.setenv("LLM_API_KEY", "configured-key")

    settings = Settings.from_env()

    assert settings.llm_enabled is True
    assert settings.llm_configured is True


def test_llm_enabled_rejects_ambiguous_value(monkeypatch):
    monkeypatch.setenv("LLM_ENABLED", "sometimes")

    with pytest.raises(ValueError, match="LLM_ENABLED"):
        Settings.from_env()
