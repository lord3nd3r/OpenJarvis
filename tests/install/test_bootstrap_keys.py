"""Tests for openjarvis.cli._bootstrap.detect_cloud_keys."""

from __future__ import annotations

import pytest

from openjarvis.cli import _bootstrap

ALL_KEYS = (
    "OPENROUTER_API_KEY",
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "XAI_API_KEY",
    "DEEPSEEK_API_KEY",
    "MINIMAX_API_KEY",
)


@pytest.fixture(autouse=True)
def _clear_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    for k in ALL_KEYS:
        monkeypatch.delenv(k, raising=False)


def test_no_keys_returns_none() -> None:
    assert _bootstrap.detect_cloud_keys() is None


def test_openrouter_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    p = _bootstrap.detect_cloud_keys()
    assert p is not None
    assert p.provider == "openrouter"
    assert p.api_key == "sk-or-test"


def test_anthropic_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    p = _bootstrap.detect_cloud_keys()
    assert p is not None
    assert p.provider == "anthropic"


def test_openai_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-oa-test")
    p = _bootstrap.detect_cloud_keys()
    assert p is not None
    assert p.provider == "openai"


def test_google_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "google-test")
    p = _bootstrap.detect_cloud_keys()
    assert p is not None
    assert p.provider == "google"


def test_gemini_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    """GEMINI_API_KEY is a recognized alias for GOOGLE_API_KEY."""
    monkeypatch.setenv("GEMINI_API_KEY", "gem-test")
    p = _bootstrap.detect_cloud_keys()
    assert p is not None
    assert p.provider == "google"


def test_xai_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XAI_API_KEY", "xai-test")
    p = _bootstrap.detect_cloud_keys()
    assert p is not None
    assert p.provider == "xai"
    assert p.env_var == "XAI_API_KEY"


def test_deepseek_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-ds-test")
    p = _bootstrap.detect_cloud_keys()
    assert p is not None
    assert p.provider == "deepseek"


def test_minimax_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MINIMAX_API_KEY", "mm-test")
    p = _bootstrap.detect_cloud_keys()
    assert p is not None
    assert p.provider == "minimax"


def test_multi_vendor_keys_do_not_outrank_the_big_four(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Single-vendor providers sit at the tail of the precedence order."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("XAI_API_KEY", "xai-test")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-ds-test")
    monkeypatch.setenv("MINIMAX_API_KEY", "mm-test")
    p = _bootstrap.detect_cloud_keys()
    assert p is not None
    assert p.provider == "anthropic"


def test_xai_has_a_default_cloud_model() -> None:
    """Every detectable provider needs a default model.

    ``bootstrap_cmd`` forces ``engine="cloud"`` once a key is detected but
    falls back to the *locally* recommended model on a map miss, which would
    pair the cloud engine with an Ollama model id.
    """
    for _, provider in _bootstrap._KEY_TO_PROVIDER:
        assert provider in _bootstrap._CLOUD_PROVIDER_DEFAULT_MODELS


def test_precedence_openrouter_over_others(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "or")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ant")
    monkeypatch.setenv("OPENAI_API_KEY", "oa")
    p = _bootstrap.detect_cloud_keys()
    assert p is not None
    assert p.provider == "openrouter"


def test_precedence_anthropic_over_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ant")
    monkeypatch.setenv("OPENAI_API_KEY", "oa")
    p = _bootstrap.detect_cloud_keys()
    assert p is not None
    assert p.provider == "anthropic"


def test_empty_string_treated_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ant")
    p = _bootstrap.detect_cloud_keys()
    assert p is not None
    assert p.provider == "anthropic"


def test_cloud_provider_repr_redacts_api_key() -> None:
    """Default repr must NOT expose the api_key (logging safety)."""
    p = _bootstrap.CloudProvider(
        provider="openrouter",
        env_var="OPENROUTER_API_KEY",
        api_key="sk-or-secret-do-not-leak",
    )
    r = repr(p)
    assert "sk-or-secret-do-not-leak" not in r
    assert "openrouter" in r
    assert "OPENROUTER_API_KEY" in r
    assert "redacted" in r.lower() or "***" in r
