import streamlit as st
import os


def get_gemini_key() -> str:
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return os.getenv("GEMINI_API_KEY", "")


def get_hf_token() -> str:
    try:
        return st.secrets["HF_TOKEN"]
    except Exception:
        return os.getenv("HF_TOKEN", "")


# ── Models ──────────────────────────────────────────────────
# NOTE (Aug 2026): Google regularly retires older Gemini model IDs
# (gemini-1.5-*, gemini-2.0-* are already shut down as of mid-2026).
# GEMINI_FLASH_CANDIDATES is tried in order — if the first model ID
# 404s because it has been retired, the next one is tried automatically.
# If you hit a "models/... is not found for API version v1beta" error
# again in the future, just add the new model ID to the FRONT of this
# list (check https://ai.google.dev/gemini-api/docs/models for the
# current list) — no other code changes are needed.
IBM_GRANITE_MODEL   = "ibm-granite/granite-3.1-8b-instruct"
GRANITE_PROVIDER    = "auto"  # HF Inference Providers routing; "auto" picks any provider that serves the model

# Tried in order, newest/best first. On ANY error (404 retired model,
# regional availability, transient 5xx, quota on that specific model,
# etc.) agents.py automatically moves to the next one — so the app
# keeps working even if Google retires or rate-limits a given model.
# gemini-2.5-flash is scheduled to shut down Oct 16, 2026 — kept only
# as a late-chain safety net, not the primary model.
# "gemini-flash-latest" is a rolling alias Google keeps pointed at
# whatever their current default Flash model is, so it's kept last
# as an always-on catch-all.
GEMINI_FLASH_CANDIDATES = [
    "gemini-3.7-flash",       # newest Flash (Aug 13, 2026)
    "gemini-3.6-flash",       # stable GA workhorse
    "gemini-3.5-flash-lite",  # stable GA, fast/cheap
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash",       # legacy safety net (retires Oct 16, 2026)
    "gemini-flash-latest",    # rolling alias — ultimate catch-all
]
GEMINI_FLASH        = GEMINI_FLASH_CANDIDATES[0]   # kept for backwards-compat / display purposes

# gemini-embedding-001 replaced text-embedding-004, which Google
# deprecated on Jan 14, 2026. Full vectors are 3072-dim; we truncate
# to 768 via output_dimensionality (Matryoshka Representation
# Learning) to keep ChromaDB storage small and stay compatible with
# the original knowledge base size.
EMBEDDING_MODEL      = "gemini-embedding-001"
EMBEDDING_OUTPUT_DIM = 768

# ── ChromaDB ─────────────────────────────────────────────────
CHROMA_PATH        = "./data/greenmind_db"
COLLECTION_NAME    = "sdg_knowledge"
CHUNK_SIZE         = 400   # words per chunk
CHUNK_OVERLAP      = 40
TOP_K              = 4     # chunks retrieved per query

# ── SDG Knowledge Base Topics ───────────────────────────────
# (sdg_id, display_name, wikipedia_search_term)
SDG_TOPICS = [
    ("SDG 1",  "No Poverty",                      "Extreme_poverty"),
    ("SDG 2",  "Zero Hunger",                     "Food_security"),
    ("SDG 3",  "Good Health and Well-being",      "Universal_health_coverage"),
    ("SDG 4",  "Quality Education",               "Education_for_All"),
    ("SDG 5",  "Gender Equality",                 "Gender_equality"),
    ("SDG 6",  "Clean Water and Sanitation",      "Water_scarcity"),
    ("SDG 7",  "Affordable and Clean Energy",     "Renewable_energy"),
    ("SDG 8",  "Decent Work and Economic Growth", "Sustainable_development"),
    ("SDG 9",  "Industry Innovation",             "Green_technology"),
    ("SDG 10", "Reduced Inequalities",            "Economic_inequality"),
    ("SDG 11", "Sustainable Cities",              "Sustainable_city"),
    ("SDG 12", "Responsible Consumption",         "Sustainable_consumption"),
    ("SDG 13", "Climate Action",                  "Climate_change"),
    ("SDG 14", "Life Below Water",                "Marine_conservation"),
    ("SDG 15", "Life on Land",                    "Biodiversity"),
    ("SDG 16", "Peace Justice Institutions",      "Rule_of_law"),
    ("SDG 17", "Partnerships for the Goals",      "Global_partnership"),
]
