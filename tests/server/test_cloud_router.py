"""Regression tests for OpenRouter model ID normalization."""

from __future__ import annotations

from pathlib import Path

import pytest

from openjarvis.core.types import Message
from openjarvis.server import cloud_router


def test_get_provider_detects_bare_openrouter_id():
    assert cloud_router.get_provider("anthropic/claude-haiku-4.5") == "openrouter"


def test_get_provider_detects_litellm_prefixed_openrouter_id():
    model = "openrouter/anthropic/claude-haiku-4.5"
    assert cloud_router.get_provider(model) == "openrouter"


@pytest.mark.parametrize(
    "requested_model,expected_forwarded_model",
    [
        ("anthropic/claude-haiku-4.5", "anthropic/claude-haiku-4.5"),
        ("openrouter/anthropic/claude-haiku-4.5", "anthropic/claude-haiku-4.5"),
        ("openrouter/auto", "openrouter/auto"),
    ],
)
@pytest.mark.asyncio
async def test_stream_cloud_normalizes_openrouter_model_before_forwarding(
    monkeypatch, requested_model, expected_forwarded_model
):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    captured: dict[str, str] = {}

    async def fake_stream_openai(model, messages, temperature, max_tokens, **kwargs):
        captured["model"] = model
        yield "ok"

    monkeypatch.setattr(cloud_router, "_stream_openai", fake_stream_openai)

    tokens = [
        token
        async for token in cloud_router.stream_cloud(
            requested_model, [Message(role="user", content="hi")]
        )
    ]

    assert tokens == ["ok"]
    assert captured["model"] == expected_forwarded_model


def test_get_provider_detects_grok_models():
    assert cloud_router.get_provider("grok-4.6") == "xai"
    assert cloud_router.get_provider("grok-4.20-0309-reasoning") == "xai"
    assert cloud_router.get_provider("grok-build-0.1") == "xai"
    assert cloud_router.is_cloud_model("grok-4.6") is True


def test_get_provider_keeps_openrouter_grok_on_openrouter():
    """A slash-qualified grok ID is an OpenRouter route, not direct xAI."""
    assert cloud_router.get_provider("x-ai/grok-4") == "openrouter"
    assert cloud_router.get_provider("openrouter/x-ai/grok-4") == "openrouter"


@pytest.mark.asyncio
async def test_stream_cloud_routes_grok_to_xai_base_url(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "xai-test-key")
    captured: dict[str, str] = {}

    async def fake_stream_openai(
        model, messages, temperature, max_tokens, base_url=None, api_key_name=None
    ):
        captured["model"] = model
        captured["base_url"] = base_url
        captured["api_key_name"] = api_key_name
        yield "ok"

    monkeypatch.setattr(cloud_router, "_stream_openai", fake_stream_openai)

    tokens = [
        token
        async for token in cloud_router.stream_cloud(
            "grok-4.6", [Message(role="user", content="hi")]
        )
    ]

    assert tokens == ["ok"]
    assert captured["model"] == "grok-4.6"
    assert captured["base_url"] == "https://api.x.ai/v1"
    assert captured["api_key_name"] == "XAI_API_KEY"


@pytest.mark.asyncio
async def test_stream_cloud_grok_without_key_raises(monkeypatch):
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.setattr(cloud_router, "_CLOUD_ENV_FILE", Path("/nonexistent"))

    with pytest.raises(ValueError, match="XAI_API_KEY"):
        async for _ in cloud_router.stream_cloud(
            "grok-4.6", [Message(role="user", content="hi")]
        ):
            pass


def test_get_provider_detects_deepseek_cloud_models():
    assert cloud_router.get_provider("deepseek-v4-pro") == "deepseek"
    assert cloud_router.get_provider("deepseek-v4-flash") == "deepseek"
    assert cloud_router.is_cloud_model("deepseek-v4-pro") is True


@pytest.mark.parametrize(
    "model",
    ["deepseek-r1:7b", "deepseek-r1:14b", "deepseek-coder-v2:16b", "qwen3.5:4b"],
)
def test_ollama_tagged_models_never_route_to_cloud(model):
    """An Ollama ``name:tag`` is local even when the vendor sells a cloud API.

    ``deepseek-r1:7b`` ships in this repo's own local model catalog; matching
    it on the ``deepseek-`` prefix would hand a local model to DeepSeek's
    cloud API the moment a DEEPSEEK_API_KEY exists.
    """
    assert cloud_router.get_provider(model) is None
    assert cloud_router.is_cloud_model(model) is False


@pytest.mark.asyncio
async def test_stream_cloud_routes_deepseek_to_its_base_url(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-ds-test")
    captured: dict[str, str] = {}

    async def fake_stream_openai(
        model, messages, temperature, max_tokens, base_url=None, api_key_name=None
    ):
        captured["model"] = model
        captured["base_url"] = base_url
        captured["api_key_name"] = api_key_name
        yield "ok"

    monkeypatch.setattr(cloud_router, "_stream_openai", fake_stream_openai)

    tokens = [
        token
        async for token in cloud_router.stream_cloud(
            "deepseek-v4-pro", [Message(role="user", content="hi")]
        )
    ]

    assert tokens == ["ok"]
    assert captured["model"] == "deepseek-v4-pro"
    assert captured["base_url"] == "https://api.deepseek.com/v1"
    assert captured["api_key_name"] == "DEEPSEEK_API_KEY"


@pytest.mark.asyncio
async def test_stream_cloud_deepseek_without_key_raises(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setattr(cloud_router, "_CLOUD_ENV_FILE", Path("/nonexistent"))

    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        async for _ in cloud_router.stream_cloud(
            "deepseek-v4-pro", [Message(role="user", content="hi")]
        ):
            pass


def test_router_and_engine_agree_on_cloud_vs_local_classification():
    """``server/cloud_router`` and ``engine/cloud`` must not disagree.

    The two classify independently (the router stays SDK-free on purpose), so
    a model either engine would serve must also route as cloud here.
    """
    from openjarvis.engine.cloud import _is_deepseek_model, _is_grok_model

    for model in ("deepseek-v4-pro", "deepseek-v4-flash", "grok-4.6"):
        assert _is_deepseek_model(model) or _is_grok_model(model)
        assert cloud_router.is_cloud_model(model) is True

    for model in ("deepseek-r1:7b", "deepseek-coder-v2:16b"):
        assert not _is_deepseek_model(model)
        assert cloud_router.is_cloud_model(model) is False
