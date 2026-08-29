"""
Tests for knowledge_base.py — chunking, embedding, and ChromaDB
retrieval logic. External calls (Wikipedia, Gemini embeddings,
ChromaDB) are mocked so these run offline and don't burn API quota.
"""
from unittest.mock import MagicMock, patch

import pytest

import knowledge_base as kb
import config


# ── chunk_text (pure function — no mocks needed) ───────────────

def test_chunk_text_splits_on_word_boundaries():
    text = " ".join(f"word{i}" for i in range(1000))
    chunks = kb.chunk_text(text)

    assert len(chunks) > 1
    # Every chunk should roughly respect CHUNK_SIZE (in words)
    for chunk in chunks:
        assert len(chunk.split()) <= config.CHUNK_SIZE


def test_chunk_text_applies_overlap_between_consecutive_chunks():
    text = " ".join(f"word{i}" for i in range(1000))
    chunks = kb.chunk_text(text)

    first_chunk_words = chunks[0].split()
    second_chunk_words = chunks[1].split()
    # The overlap region (last CHUNK_OVERLAP words of chunk 1) should
    # reappear at the start of chunk 2.
    overlap_expected = first_chunk_words[-config.CHUNK_OVERLAP:]
    overlap_actual = second_chunk_words[:config.CHUNK_OVERLAP]
    assert overlap_expected == overlap_actual


def test_chunk_text_skips_tiny_leftover_fragment():
    # Construct text whose final chunk is under the 50-char floor
    text = " ".join(["word"] * (config.CHUNK_SIZE - config.CHUNK_OVERLAP)) + " x"
    chunks = kb.chunk_text(text)
    # The trailing "x" alone should have been dropped, not returned as its own chunk
    assert all(len(c.strip()) > 50 for c in chunks)


def test_chunk_text_empty_string_returns_no_chunks():
    assert kb.chunk_text("") == []


# ── fetch_wikipedia ──────────────────────────────────────────

def test_fetch_wikipedia_returns_extract_text():
    fake_response = MagicMock()
    fake_response.json.return_value = {
        "query": {"pages": {"123": {"extract": "Climate change is a long-term shift."}}}
    }
    with patch("knowledge_base.requests.get", return_value=fake_response):
        result = kb.fetch_wikipedia("Climate_change")

    assert result == "Climate change is a long-term shift."


def test_fetch_wikipedia_caps_at_6000_chars():
    long_text = "x" * 10000
    fake_response = MagicMock()
    fake_response.json.return_value = {"query": {"pages": {"1": {"extract": long_text}}}}
    with patch("knowledge_base.requests.get", return_value=fake_response):
        result = kb.fetch_wikipedia("Some_page")

    assert len(result) == 6000


def test_fetch_wikipedia_returns_empty_string_on_network_error():
    with patch("knowledge_base.requests.get", side_effect=ConnectionError("no network")):
        result = kb.fetch_wikipedia("Climate_change")

    assert result == ""


# ── get_embedding ────────────────────────────────────────────

def test_get_embedding_calls_gemini_with_correct_task_type():
    fake_client = MagicMock()
    fake_embedding = MagicMock()
    fake_embedding.values = [0.1] * config.EMBEDDING_OUTPUT_DIM
    fake_client.models.embed_content.return_value = MagicMock(embeddings=[fake_embedding])

    with patch("knowledge_base.genai.Client", return_value=fake_client):
        result = kb.get_embedding("test text", "fake-key", task_type="RETRIEVAL_QUERY")

    assert len(result) == config.EMBEDDING_OUTPUT_DIM
    _, kwargs = fake_client.models.embed_content.call_args
    assert kwargs["config"].task_type == "RETRIEVAL_QUERY"


def test_get_embedding_truncates_text_to_2000_chars():
    fake_client = MagicMock()
    fake_embedding = MagicMock()
    fake_embedding.values = [0.0]
    fake_client.models.embed_content.return_value = MagicMock(embeddings=[fake_embedding])

    long_text = "a" * 5000
    with patch("knowledge_base.genai.Client", return_value=fake_client):
        kb.get_embedding(long_text, "fake-key")

    _, kwargs = fake_client.models.embed_content.call_args
    assert len(kwargs["contents"]) == 2000


# ── query_knowledge_base ─────────────────────────────────────

def test_query_knowledge_base_returns_placeholder_when_empty():
    empty_collection = MagicMock()
    empty_collection.count.return_value = 0

    result = kb.query_knowledge_base(empty_collection, "any query", "fake-key")
    assert result == "Knowledge base is empty."


def test_query_knowledge_base_formats_retrieved_context(sample_rag_context):
    collection = MagicMock()
    collection.count.return_value = 10
    collection.query.return_value = {
        "documents": [["Climate change refers to long-term shifts in temperatures."]],
        "metadatas": [[{
            "sdg_id": "SDG 13",
            "display_name": "Climate Action",
            "source": "Wikipedia: Climate_change",
        }]],
    }

    with patch("knowledge_base.get_embedding", return_value=[0.1, 0.2]):
        result = kb.query_knowledge_base(collection, "climate query", "fake-key")

    assert "SDG 13 – Climate Action" in result
    assert "Source: Wikipedia: Climate_change" in result


def test_query_knowledge_base_handles_failure_gracefully():
    collection = MagicMock()
    collection.count.return_value = 5

    with patch("knowledge_base.get_embedding", side_effect=RuntimeError("embedding API down")):
        result = kb.query_knowledge_base(collection, "climate query", "fake-key")

    assert result.startswith("RAG query failed:")


# ── build_knowledge_base ─────────────────────────────────────

def test_build_knowledge_base_skips_rebuild_if_already_populated():
    """If the collection already has data, we should NOT re-fetch Wikipedia
    or re-embed — this is the cost-control behaviour worth locking in."""
    fake_collection = MagicMock()
    fake_collection.count.return_value = 999  # already populated

    fake_client = MagicMock()
    fake_client.get_or_create_collection.return_value = fake_collection

    with patch("knowledge_base.chromadb.PersistentClient", return_value=fake_client), \
         patch("knowledge_base.fetch_wikipedia") as mock_fetch:
        result = kb.build_knowledge_base("fake-key")

    mock_fetch.assert_not_called()
    assert result is fake_collection


def test_build_knowledge_base_fetches_chunks_and_embeds_when_empty():
    """The actual populate path: empty collection -> fetch each SDG topic's
    Wikipedia text, chunk it, embed each chunk, and add it to the collection."""
    fake_collection = MagicMock()
    fake_collection.count.return_value = 0  # not yet populated

    fake_client = MagicMock()
    fake_client.get_or_create_collection.return_value = fake_collection

    with patch("knowledge_base.chromadb.PersistentClient", return_value=fake_client), \
         patch("knowledge_base.fetch_wikipedia", return_value="Some SDG article text " * 50), \
         patch("knowledge_base.get_embedding", return_value=[0.1, 0.2, 0.3]) as mock_embed:
        result = kb.build_knowledge_base("fake-key")

    assert result is fake_collection
    # One fetch per SDG topic (17 total)
    assert mock_embed.call_count >= len(config.SDG_TOPICS)
    # Each successful chunk should have been added to the collection
    assert fake_collection.add.called


def test_build_knowledge_base_skips_topic_on_empty_wikipedia_fetch():
    """If Wikipedia fetch fails for a topic (returns ''), that topic should
    be skipped entirely rather than crashing the whole knowledge base build."""
    fake_collection = MagicMock()
    fake_collection.count.return_value = 0

    fake_client = MagicMock()
    fake_client.get_or_create_collection.return_value = fake_collection

    with patch("knowledge_base.chromadb.PersistentClient", return_value=fake_client), \
         patch("knowledge_base.fetch_wikipedia", return_value=""), \
         patch("knowledge_base.get_embedding") as mock_embed:
        kb.build_knowledge_base("fake-key")

    mock_embed.assert_not_called()
    fake_collection.add.assert_not_called()


def test_build_knowledge_base_continues_past_single_embedding_failure():
    """One chunk's embedding call failing shouldn't abort the whole build."""
    fake_collection = MagicMock()
    fake_collection.count.return_value = 0

    fake_client = MagicMock()
    fake_client.get_or_create_collection.return_value = fake_collection

    with patch("knowledge_base.chromadb.PersistentClient", return_value=fake_client), \
         patch("knowledge_base.fetch_wikipedia", return_value="text " * 50), \
         patch("knowledge_base.get_embedding", side_effect=RuntimeError("embedding API down")):
        # Should not raise, despite every embedding call failing
        result = kb.build_knowledge_base("fake-key")

    assert result is fake_collection
    fake_collection.add.assert_not_called()
