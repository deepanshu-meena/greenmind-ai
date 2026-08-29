"""
Tests for config.py — the key-lookup functions that check Streamlit
secrets first and fall back to environment variables.
"""
from unittest.mock import patch

import pytest

import config


class _RaisingSecrets:
    """Stands in for st.secrets when no secrets.toml exists — real
    Streamlit raises on access/subscript in that case."""
    def __getitem__(self, key):
        raise Exception("No secrets found")


def test_get_gemini_key_reads_from_streamlit_secrets():
    with patch.object(config.st, "secrets", {"GEMINI_API_KEY": "secret-key"}):
        result = config.get_gemini_key()
    assert result == "secret-key"


def test_get_gemini_key_falls_back_to_env_var(monkeypatch):
    # st.secrets raises (no secrets.toml present) -> should fall back to os.getenv
    with patch.object(config.st, "secrets", _RaisingSecrets()):
        monkeypatch.setenv("GEMINI_API_KEY", "env-key")
        result = config.get_gemini_key()
    assert result == "env-key"


def test_get_gemini_key_returns_empty_string_when_nothing_set(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with patch.object(config.st, "secrets", _RaisingSecrets()):
        result = config.get_gemini_key()
    assert result == ""


def test_get_hf_token_falls_back_to_env_var(monkeypatch):
    with patch.object(config.st, "secrets", _RaisingSecrets()):
        monkeypatch.setenv("HF_TOKEN", "env-hf-token")
        result = config.get_hf_token()
    assert result == "env-hf-token"


def test_gemini_flash_candidates_is_nonempty_and_flash_matches_first():
    assert len(config.GEMINI_FLASH_CANDIDATES) > 0
    assert config.GEMINI_FLASH == config.GEMINI_FLASH_CANDIDATES[0]


def test_sdg_topics_covers_all_17_goals():
    assert len(config.SDG_TOPICS) == 17
    ids = [t[0] for t in config.SDG_TOPICS]
    assert len(set(ids)) == 17  # no duplicate SDG ids
