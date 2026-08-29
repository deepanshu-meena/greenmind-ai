"""
Shared fixtures for the GreenMind AI test suite.
"""
import sys
import os

# Make the repo root importable when pytest is run from anywhere
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest


@pytest.fixture
def sample_ddgs_results():
    """A realistic DDGS().text() return value."""
    return [
        {
            "title": "SDG 13: Climate Action",
            "body": "Climate action refers to efforts to combat climate change.",
            "href": "https://example.com/sdg13",
        },
        {
            "title": "Renewable Energy Trends 2026",
            "body": "Solar and wind capacity grew significantly in 2026.",
            "href": "https://example.com/renewables",
        },
    ]


@pytest.fixture
def sample_web_results_text():
    """Formatted output as produced by web_search.search_web()."""
    return (
        "[1] SDG 13: Climate Action\n"
        "Climate action refers to efforts to combat climate change.\n"
        "Source: https://example.com/sdg13\n\n"
        "[2] Renewable Energy Trends 2026\n"
        "Solar and wind capacity grew significantly in 2026.\n"
        "Source: https://example.com/renewables"
    )


@pytest.fixture
def sample_rag_context():
    """Formatted output as produced by knowledge_base.query_knowledge_base()."""
    return (
        "\n[SDG 13 – Climate Action]\n"
        "Source: Wikipedia: Climate_change\n"
        "Climate change refers to long-term shifts in temperatures.\n"
        "────────────────────────────────────────────────────────\n"
        "\n[SDG 7 – Affordable and Clean Energy]\n"
        "Source: Wikipedia: Renewable_energy\n"
        "Renewable energy is energy from renewable resources.\n"
        "────────────────────────────────────────────────────────"
    )
