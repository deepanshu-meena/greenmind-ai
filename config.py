import streamlit as st
import os

def get_gemini_key() -> str:
    try:
        return st.secrets["GEMINI_API_KEY"]
    except:
        return os.getenv("GEMINI_API_KEY", "")

def get_hf_token() -> str:
    try:
        return st.secrets["HF_TOKEN"]
    except:
        return os.getenv("HF_TOKEN", "")

# ── Models ──────────────────────────────────────────────────
IBM_GRANITE_MODEL  = "ibm-granite/granite-3.1-8b-instruct"
GEMINI_FLASH       = "gemini-1.5-flash"
EMBEDDING_MODEL    = "models/text-embedding-004"

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
