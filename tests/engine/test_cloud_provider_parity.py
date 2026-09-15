"""Every cloud provider must be wired through the whole stack, not just the engine.

Adding a provider to ``engine/cloud.py`` alone leaves it unreachable: the
desktop app can't store its key, the chat UI can't route to it, and
install-time detection ignores it. DeepSeek shipped in that half-wired state —
engine support with no key storage, no router branch, no catalog entry and no
UI — until this table was made explicit.

Each row is one provider and the env var that unlocks it. The assertions walk
the integration points a provider has to appear in. They intentionally read the
Rust and TypeScript sources as text: those are separate toolchains, so a
string check is the only cheap way to keep one Python table honest about them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from openjarvis.cli._bootstrap import (
    _CLOUD_PROVIDER_DEFAULT_MODELS,
    _KEY_TO_PROVIDER,
)
from openjarvis.engine.cloud import CloudEngine
from openjarvis.intelligence.model_catalog import BUILTIN_MODELS
from openjarvis.security.data_boundary_audit import API_KEY_ENV_VARS
from openjarvis.server import cloud_router

ROOT = Path(__file__).resolve().parents[2]

_TAURI_LIB = ROOT / "frontend" / "src-tauri" / "src" / "lib.rs"
_SETTINGS_PAGE = ROOT / "frontend" / "src" / "pages" / "SettingsPage.tsx"
_COMMAND_PALETTE = ROOT / "frontend" / "src" / "components" / "CommandPalette.tsx"
_SERVE_CMD = ROOT / "src" / "openjarvis" / "cli" / "serve.py"

# provider slug -> (env var, CloudEngine client attribute, catalog provider)
#
# Codex is deliberately absent: it authenticates with a ChatGPT subscription
# token against the Responses API rather than a plain per-vendor API key, so it
# does not belong in the key-management surfaces this table guards.
CLOUD_PROVIDERS: dict[str, tuple[str, str, str]] = {
    "openai": ("OPENAI_API_KEY", "_openai_client", "openai"),
    "anthropic": ("ANTHROPIC_API_KEY", "_anthropic_client", "anthropic"),
    "google": ("GEMINI_API_KEY", "_google_client", "google"),
    "openrouter": ("OPENROUTER_API_KEY", "_openrouter_client", ""),
    "minimax": ("MINIMAX_API_KEY", "_minimax_client", "minimax"),
    "deepseek": ("DEEPSEEK_API_KEY", "_deepseek_client", "deepseek"),
    "xai": ("XAI_API_KEY", "_xai_client", "xai"),
}

_IDS = sorted(CLOUD_PROVIDERS)


@pytest.fixture(scope="module")
def sources() -> dict[Path, str]:
    return {
        path: path.read_text(encoding="utf-8")
        for path in (_TAURI_LIB, _SETTINGS_PAGE, _COMMAND_PALETTE, _SERVE_CMD)
    }


@pytest.mark.parametrize("provider", _IDS)
def test_engine_exposes_a_client_attribute(provider: str) -> None:
    _, attr, _ = CLOUD_PROVIDERS[provider]
    engine = CloudEngine.__new__(CloudEngine)
    engine.__init__()  # noqa: PLC2801 - exercise real client wiring
    assert hasattr(engine, attr), f"CloudEngine is missing {attr}"


@pytest.mark.parametrize("provider", _IDS)
def test_router_loads_the_key(provider: str, monkeypatch) -> None:
    """The SDK-free server router must pick the key up from the environment."""
    env_var, _, _ = CLOUD_PROVIDERS[provider]
    monkeypatch.setattr(cloud_router, "_CLOUD_ENV_FILE", Path("/nonexistent"))
    monkeypatch.setenv(env_var, "sentinel-value")
    assert cloud_router._load_keys().get(env_var) == "sentinel-value"


@pytest.mark.parametrize("provider", _IDS)
def test_serve_reports_cloud_enabled_for_the_key(
    provider: str, sources: dict[Path, str]
) -> None:
    env_var, _, _ = CLOUD_PROVIDERS[provider]
    assert env_var in sources[_SERVE_CMD], (
        f"{env_var} missing from serve.py's cloud-key detection; "
        "`jarvis serve` would not report cloud as enabled"
    )


@pytest.mark.parametrize("provider", _IDS)
def test_install_time_detection_knows_the_key(provider: str) -> None:
    env_var, _, _ = CLOUD_PROVIDERS[provider]
    detected = dict(_KEY_TO_PROVIDER)
    assert env_var in detected, f"{env_var} missing from _KEY_TO_PROVIDER"
    assert detected[env_var] in _CLOUD_PROVIDER_DEFAULT_MODELS, (
        f"{detected[env_var]!r} has no default model; bootstrap would force "
        "engine=cloud while keeping a locally recommended model id"
    )


@pytest.mark.parametrize("provider", _IDS)
def test_privacy_audit_tracks_the_key(provider: str) -> None:
    env_var, _, _ = CLOUD_PROVIDERS[provider]
    assert env_var in API_KEY_ENV_VARS, (
        f"{env_var} missing from the data-boundary audit; a configured cloud "
        "key would not show up in a privacy scan"
    )


@pytest.mark.parametrize("provider", _IDS)
def test_catalog_lists_cloud_models(provider: str) -> None:
    _, _, catalog_provider = CLOUD_PROVIDERS[provider]
    if not catalog_provider:
        pytest.skip("OpenRouter models are enumerated per-slug, not catalogued")
    specs = [
        m
        for m in BUILTIN_MODELS
        if m.provider == catalog_provider and "cloud" in m.supported_engines
    ]
    assert specs, f"no cloud ModelSpec entries for provider {catalog_provider!r}"
    assert all(m.requires_api_key for m in specs)


@pytest.mark.parametrize("provider", _IDS)
def test_desktop_app_can_store_the_key(provider: str, sources: dict[Path, str]) -> None:
    env_var, _, _ = CLOUD_PROVIDERS[provider]
    assert env_var in sources[_TAURI_LIB], (
        f"{env_var} missing from MANAGED_CLOUD_KEY_NAMES; the desktop app "
        "would save the key to the keyring but never read it back"
    )


@pytest.mark.parametrize("provider", _IDS)
def test_settings_page_offers_the_key(provider: str, sources: dict[Path, str]) -> None:
    env_var, _, _ = CLOUD_PROVIDERS[provider]
    assert env_var in sources[_SETTINGS_PAGE], (
        f"{env_var} has no Settings input; the key is unreachable from the UI"
    )


@pytest.mark.parametrize("provider", _IDS)
def test_command_palette_offers_models(provider: str, sources: dict[Path, str]) -> None:
    env_var, _, _ = CLOUD_PROVIDERS[provider]
    assert env_var in sources[_COMMAND_PALETTE], (
        f"{env_var} has no command-palette entry; its models cannot be picked"
    )


def test_router_and_engine_cover_the_same_providers(monkeypatch) -> None:
    """Every table key must make the engine healthy and the router cloud-aware."""
    monkeypatch.setattr(cloud_router, "_CLOUD_ENV_FILE", Path("/nonexistent"))
    for env_var, _, _ in CLOUD_PROVIDERS.values():
        monkeypatch.setenv(env_var, "sentinel-value")
    loaded = cloud_router._load_keys()
    missing = [
        env_var for env_var, _, _ in CLOUD_PROVIDERS.values() if env_var not in loaded
    ]
    assert not missing, f"router ignores these keys: {missing}"
