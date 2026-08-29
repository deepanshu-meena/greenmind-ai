"""
Tests for agents.py — LLM wrapper + fallback logic and the 4 agents.

This is where the most valuable branching logic lives: Granite-first
with silent fallback to Gemini, and Gemini's own multi-model candidate
chain. These tests exercise that logic without ever making a real
network call.
"""
from unittest.mock import MagicMock, patch

import pytest

import agents
from config import GEMINI_FLASH_CANDIDATES


# ── llm_call: Granite-first, Gemini-fallback ────────────────────

def test_llm_call_uses_granite_when_hf_token_present():
    with patch("agents._granite", return_value="granite response") as mock_granite, \
         patch("agents._gemini") as mock_gemini:
        result = agents.llm_call("prompt", "system", hf_token="hf-token", gemini_key="g-key")

    assert result == "granite response"
    mock_granite.assert_called_once()
    mock_gemini.assert_not_called()


def test_llm_call_falls_back_to_gemini_when_granite_fails():
    with patch("agents._granite", side_effect=RuntimeError("granite down")), \
         patch("agents._gemini", return_value="gemini response") as mock_gemini:
        result = agents.llm_call("prompt", "system", hf_token="hf-token", gemini_key="g-key")

    assert result == "gemini response"
    mock_gemini.assert_called_once()


def test_llm_call_skips_granite_when_no_hf_token():
    with patch("agents._granite") as mock_granite, \
         patch("agents._gemini", return_value="gemini response"):
        result = agents.llm_call("prompt", "system", hf_token="", gemini_key="g-key")

    assert result == "gemini response"
    mock_granite.assert_not_called()


# ── _gemini: multi-model candidate fallback chain ───────────────

def test_gemini_succeeds_on_first_candidate():
    fake_response = MagicMock(text="answer")
    fake_client = MagicMock()
    fake_client.models.generate_content.return_value = fake_response

    with patch("agents.genai.Client", return_value=fake_client):
        result = agents._gemini("prompt", "system", "fake-key")

    assert result == "answer"
    # Only the first (newest) candidate model should have been tried
    called_model = fake_client.models.generate_content.call_args.kwargs["model"]
    assert called_model == GEMINI_FLASH_CANDIDATES[0]


def test_gemini_moves_to_next_candidate_on_failure():
    fake_client = MagicMock()
    call_log = []

    def side_effect(model, **kwargs):
        call_log.append(model)
        if model == GEMINI_FLASH_CANDIDATES[0]:
            raise RuntimeError("model retired (404)")
        return MagicMock(text="answer from second model")

    fake_client.models.generate_content.side_effect = side_effect

    with patch("agents.genai.Client", return_value=fake_client), \
         patch("agents.time.sleep"):  # skip the real 1s backoff in tests
        result = agents._gemini("prompt", "system", "fake-key")

    assert result == "answer from second model"
    # First candidate should have been retried once before moving on
    assert call_log.count(GEMINI_FLASH_CANDIDATES[0]) == 2
    assert GEMINI_FLASH_CANDIDATES[1] in call_log


def test_gemini_raises_only_after_every_candidate_exhausted():
    fake_client = MagicMock()
    fake_client.models.generate_content.side_effect = RuntimeError("always fails")

    with patch("agents.genai.Client", return_value=fake_client), \
         patch("agents.time.sleep"):
        with pytest.raises(RuntimeError) as exc_info:
            agents._gemini("prompt", "system", "fake-key")

    # Every candidate should have been attempted (2 tries each) before giving up
    assert fake_client.models.generate_content.call_count == len(GEMINI_FLASH_CANDIDATES) * 2
    assert "Gemini API key" in str(exc_info.value)


def test_gemini_treats_empty_response_as_failure_and_moves_on():
    fake_client = MagicMock()
    call_log = []

    def side_effect(model, **kwargs):
        call_log.append(model)
        if model == GEMINI_FLASH_CANDIDATES[0]:
            return MagicMock(text="")  # empty response, not an exception
        return MagicMock(text="real answer")

    fake_client.models.generate_content.side_effect = side_effect

    with patch("agents.genai.Client", return_value=fake_client), \
         patch("agents.time.sleep"):
        result = agents._gemini("prompt", "system", "fake-key")

    assert result == "real answer"
    # Empty response shouldn't be retried on the SAME model (it won't fix itself)
    assert call_log.count(GEMINI_FLASH_CANDIDATES[0]) == 1


# ── _granite: the HuggingFace Inference Providers call itself ──────

def test_granite_sends_system_and_user_messages_and_returns_content():
    fake_message = MagicMock(content="granite says hello")
    fake_choice = MagicMock(message=fake_message)
    fake_response = MagicMock(choices=[fake_choice])

    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = fake_response

    with patch("agents.InferenceClient", return_value=fake_client):
        result = agents._granite("user prompt", "system prompt", "hf-token")

    assert result == "granite says hello"
    _, kwargs = fake_client.chat.completions.create.call_args
    assert kwargs["messages"][0] == {"role": "system", "content": "system prompt"}
    assert kwargs["messages"][1] == {"role": "user", "content": "user prompt"}


# ── The 4 agent functions: each should call llm_call with a built prompt ──

@pytest.mark.parametrize("agent_fn,extra_args", [
    (agents.search_agent, ("web results text",)),
    (agents.rag_agent, ("rag context text",)),
])
def test_single_input_agents_call_llm_with_query_in_prompt(agent_fn, extra_args):
    with patch("agents.llm_call", return_value="agent output") as mock_llm:
        result = agent_fn("What is SDG 13?", *extra_args, "hf-token", "gemini-key")

    assert result == "agent output"
    prompt_arg = mock_llm.call_args.args[0]
    assert "What is SDG 13?" in prompt_arg


def test_analysis_agent_includes_both_summaries_in_prompt():
    with patch("agents.llm_call", return_value="analysis output") as mock_llm:
        result = agents.analysis_agent(
            "query", "search summary text", "rag summary text", "hf-token", "gemini-key"
        )

    assert result == "analysis output"
    prompt_arg = mock_llm.call_args.args[0]
    assert "search summary text" in prompt_arg
    assert "rag summary text" in prompt_arg


def test_writer_agent_includes_all_upstream_context_in_prompt():
    with patch("agents.llm_call", return_value="final report") as mock_llm:
        result = agents.writer_agent(
            "query", "search summary", "rag summary", "analysis text",
            "hf-token", "gemini-key",
        )

    assert result == "final report"
    prompt_arg = mock_llm.call_args.args[0]
    for expected in ("search summary", "rag summary", "analysis text"):
        assert expected in prompt_arg
    # The anti-hallucination instruction should be present in the system prompt
    system_arg = mock_llm.call_args.args[1]
    assert "never invent" in system_arg.lower()
