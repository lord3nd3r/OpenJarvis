"""xAI (Grok) as a hybrid-harness cloud endpoint."""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from types import SimpleNamespace
from unittest import mock

import pytest

from openjarvis.agents.hybrid._base import LocalCloudAgent
from openjarvis.agents.hybrid._prices import PRICES, cost
from openjarvis.agents.hybrid.conductor import _CONDUCTOR_VALID_ENDPOINTS


@contextmanager
def _fake_openai(captured: dict):
    """Swap in an ``openai`` stand-in, touching only that one sys.modules key.

    Deliberately NOT ``mock.patch.dict("sys.modules", ...)``: that restores by
    clearing the dict and re-filling it from a pre-block snapshot, which
    *evicts* any module first imported inside the block. A later re-import then
    rebuilds the module's classes as new objects, so unrelated tests holding the
    original classes start failing ``isinstance`` checks — which is exactly how
    this file broke ``tests/agents/test_executor_error_detail.py`` under xdist.
    """
    module = _fake_openai_module(captured)
    sentinel = object()
    previous = sys.modules.get("openai", sentinel)
    sys.modules["openai"] = module
    try:
        yield module
    finally:
        if previous is sentinel:
            sys.modules.pop("openai", None)
        else:
            sys.modules["openai"] = previous


def _fake_openai_module(captured: dict) -> mock.MagicMock:
    """An ``openai`` module stand-in that records client + call kwargs."""
    resp = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="grok says hi", tool_calls=None),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=13, completion_tokens=5),
    )
    client = mock.MagicMock()
    client.chat.completions.create.return_value = resp

    def _ctor(**kwargs):
        captured["client_kwargs"] = kwargs
        return client

    module = mock.MagicMock()
    module.OpenAI = _ctor
    captured["client"] = client
    return module


class TestGrokPricing:
    def test_grok_models_are_priced(self) -> None:
        for model in ("grok-4.6", "grok-4.5", "grok-4.3", "grok-build-0.1"):
            assert model in PRICES

    def test_grok_cost_uses_standard_tier(self) -> None:
        assert cost("grok-4.6", 1_000_000, 1_000_000) == pytest.approx(8.00)
        assert cost("grok-build-0.1", 1_000_000, 1_000_000) == pytest.approx(3.00)

    def test_existing_research_prices_are_untouched(self) -> None:
        """``_prices.py`` is the authoritative cost reference for published
        n=500 numbers; adding Grok must not perturb any existing entry."""
        assert PRICES["gpt-5"] == (1.25, 10.0)
        assert PRICES["claude-sonnet-4-6"] == (3.00, 15.0)
        assert PRICES["gemini-2.5-pro"] == (1.25, 10.0)


class TestConductorEndpoint:
    def test_xai_is_a_valid_worker_endpoint(self) -> None:
        assert "xai" in _CONDUCTOR_VALID_ENDPOINTS

    def test_priced_grok_models_satisfy_the_worker_pool_price_check(self) -> None:
        """conductor rejects a non-openrouter worker whose model lacks a price."""
        for model in ("grok-4.6", "grok-4.3", "grok-build-0.1"):
            assert model in PRICES


class TestCallGrok:
    def test_targets_the_xai_base_url_with_the_xai_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("XAI_API_KEY", "xai-test")
        captured: dict = {}
        with _fake_openai(captured):
            text, p, c = LocalCloudAgent._call_grok(
                "grok-4.6", user="hi", system="be brief", max_tokens=256
            )

        assert text == "grok says hi"
        assert (p, c) == (13, 5)
        assert captured["client_kwargs"]["base_url"] == "https://api.x.ai/v1"
        assert captured["client_kwargs"]["api_key"] == "xai-test"

        call = captured["client"].chat.completions.create.call_args.kwargs
        assert call["model"] == "grok-4.6"
        assert call["max_tokens"] == 256
        assert call["messages"] == [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "hi"},
        ]

    def test_passes_tools_and_response_format_through(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("XAI_API_KEY", "xai-test")
        captured: dict = {}
        tools = [{"type": "function", "function": {"name": "f"}}]
        with _fake_openai(captured):
            LocalCloudAgent._call_grok(
                "grok-4.6",
                user="hi",
                tools=tools,
                tool_choice="auto",
                response_format={"type": "json_object"},
            )

        call = captured["client"].chat.completions.create.call_args.kwargs
        assert call["tools"] == tools
        assert call["tool_choice"] == "auto"
        assert call["response_format"] == {"type": "json_object"}

    def test_missing_key_raises_before_any_request(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="XAI_API_KEY"):
            LocalCloudAgent._call_grok("grok-4.6", user="hi")

    def test_no_routing_prefix_is_stripped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Unlike OpenRouter, xAI model IDs are bare and must pass verbatim."""
        monkeypatch.setenv("XAI_API_KEY", "xai-test")
        captured: dict = {}
        with _fake_openai(captured):
            LocalCloudAgent._call_grok("grok-4.20-0309-reasoning", user="hi")
        call = captured["client"].chat.completions.create.call_args.kwargs
        assert call["model"] == "grok-4.20-0309-reasoning"


class _StubAgent(LocalCloudAgent):
    """Concrete subclass so ``_call_cloud`` can be exercised in isolation."""

    def _run_paradigm(self, *args, **kwargs):  # pragma: no cover - unused
        raise NotImplementedError


class TestCallCloudDispatch:
    def test_xai_endpoint_routes_to_call_grok(self) -> None:
        agent = _StubAgent.__new__(_StubAgent)
        agent._cloud_endpoint = "xai"
        agent._cloud_model = "grok-4.6"

        with mock.patch.object(
            LocalCloudAgent, "_call_grok", return_value=("ok", 1, 2)
        ) as called:
            result = agent._call_cloud(user="hi", system=None)

        assert result == ("ok", 1, 2)
        assert called.call_args.args[0] == "grok-4.6"

    def test_unknown_endpoint_still_raises(self) -> None:
        agent = _StubAgent.__new__(_StubAgent)
        agent._cloud_endpoint = "nope"
        agent._cloud_model = "m"
        with pytest.raises(ValueError, match="unsupported cloud endpoint"):
            agent._call_cloud(user="hi")


def test_grok_is_not_a_hosted_search_endpoint() -> None:
    """Grok is deliberately excluded from the search-capable endpoint lists.

    Those paradigms drive provider-*hosted* web search (OpenAI Responses
    ``tools=[{"type": "web_search"}]``, Anthropic server tools). xAI exposes
    live search through a different request shape, so an xai orchestrator
    must keep failing loudly rather than silently searching nothing.
    """
    from openjarvis.agents.hybrid.advisors import _SEARCH_CAPABLE_ENDPOINTS
    from openjarvis.agents.hybrid.conductor import _SEARCH_CAPABLE_WORKER_ENDPOINTS

    assert "xai" not in _SEARCH_CAPABLE_ENDPOINTS
    assert "xai" not in _SEARCH_CAPABLE_WORKER_ENDPOINTS


def test_call_grok_does_not_use_the_openrouter_limiter() -> None:
    """The limiter guards a shared OpenRouter account's caps, not xAI's."""
    source = os.path.abspath(
        LocalCloudAgent._call_grok.__code__.co_filename  # type: ignore[union-attr]
    )
    assert source.endswith("_base.py")
    assert "_openrouter_limiter" not in LocalCloudAgent._call_grok.__code__.co_names
