"""
app.py  ·  GreenMind AI
Multi-Agent SDG Intelligence System
Powered by IBM Granite + Gemini + RAG + DuckDuckGo
1M1B × IBM SkillsBuild AI for Sustainability Internship (AICTE)
"""

import streamlit as st
from config          import get_gemini_key, get_hf_token
from knowledge_base  import build_knowledge_base
from orchestrator    import run_pipeline

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="GreenMind AI",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0d1b2a; color: #e8f5e9; }

    /* Title */
    .main-title {
        font-size: 2.8rem; font-weight: 800;
        background: linear-gradient(90deg, #00c853, #64dd17, #00bfa5);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-align: center; margin-bottom: 0.2rem;
    }
    .sub-title {
        text-align: center; color: #80cbc4;
        font-size: 1rem; margin-bottom: 1.5rem;
    }

    /* Agent cards */
    .agent-card {
        background: #1b2d3e; border-left: 4px solid #00c853;
        border-radius: 8px; padding: 1rem 1.2rem; margin: 0.5rem 0;
    }
    .agent-label {
        font-size: 0.78rem; font-weight: 700;
        color: #00c853; letter-spacing: 1px; text-transform: uppercase;
    }

    /* Metric cards */
    .metric-row { display: flex; gap: 1rem; margin-bottom: 1rem; }
    .metric-card {
        background: #1b2d3e; border-radius: 8px;
        padding: 0.8rem 1rem; flex: 1; text-align: center;
        border: 1px solid #2e4a3e;
    }
    .metric-val { font-size: 1.5rem; font-weight: 700; color: #00c853; }
    .metric-lbl { font-size: 0.75rem; color: #80cbc4; }

    /* Sidebar */
    section[data-testid="stSidebar"] { background-color: #112233; }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #00c853, #00bfa5);
        color: #0d1b2a; font-weight: 700; border: none;
        border-radius: 8px; padding: 0.6rem 2rem;
        font-size: 1rem; width: 100%;
    }
    .stButton > button:hover { opacity: 0.9; }

    /* Input */
    .stTextArea textarea { background: #1b2d3e; color: #e8f5e9;
        border: 1px solid #2e4a3e; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# ── Header ───────────────────────────────────────────────────
st.markdown('<div class="main-title">🌍 GreenMind AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">'
    'Multi-Agent SDG Intelligence System · '
    'Powered by IBM Granite + RAG + Agentic AI · '
    '1M1B × IBM SkillsBuild (AICTE)'
    '</div>',
    unsafe_allow_html=True,
)
st.divider()


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    gemini_key = st.text_input(
        "Gemini API Key", value=get_gemini_key(),
        type="password", help="Used for embeddings & fallback LLM"
    )
    hf_token = st.text_input(
        "HuggingFace Token", value=get_hf_token(),
        type="password", help="Required for IBM Granite model"
    )

    st.divider()
    st.markdown("### 🤖 Agent Pipeline")
    st.markdown("""
    1. 🔍 **Search Agent** — Live web search
    2. 📚 **RAG Agent** — SDG Knowledge Base
    3. 🔬 **Analysis Agent** — Cross-source analysis
    4. ✍️ **Writer Agent** — Intelligence report
    """)

    st.divider()
    st.markdown("### 🌐 SDG Coverage")
    st.markdown("Knowledge base covers all **17 UN SDGs** via Wikipedia.")

    st.divider()
    st.markdown("### 📌 Sample Questions")
    samples = [
        "What is the impact of climate change on food security?",
        "How can AI help achieve SDG 7 clean energy goals?",
        "What are the biggest threats to ocean biodiversity?",
        "How does deforestation affect climate change?",
        "What is the Paris Agreement target for 2050?",
    ]
    for s in samples:
        if st.button(s, key=s):
            st.session_state["query_input"] = s


# ── Knowledge Base Init ──────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_knowledge_base(api_key: str):
    return build_knowledge_base(api_key)


# ── Main Interface ───────────────────────────────────────────
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    default_q = st.session_state.get("query_input", "")
    query = st.text_area(
        "🌱 Ask about Climate, Sustainability, or any UN SDG",
        value=default_q,
        placeholder="e.g. How can AI help achieve net-zero carbon emissions by 2050?",
        height=90,
    )
    run_btn = st.button("🚀 Generate Intelligence Report", use_container_width=True)

st.divider()

# ── Run Pipeline ─────────────────────────────────────────────
if run_btn and query.strip():

    if not gemini_key:
        st.error("⚠️ Please enter your Gemini API Key in the sidebar.")
        st.stop()

    # Build / load knowledge base
    with st.spinner("🌱 Loading SDG Knowledge Base (first run may take ~60s)…"):
        try:
            collection = load_knowledge_base(gemini_key)
        except Exception as e:
            st.error(
                f"Knowledge base error: {e}\n\n"
                "This is almost always an invalid/expired Gemini API key "
                "(get a fresh one free at aistudio.google.com/apikey), "
                "or a temporary Google API outage — try again in a minute."
            )
            st.stop()

    # Agent pipeline with live status updates
    status_box = st.empty()
    agent_outputs = {}

    def update_status(msg: str):
        status_box.info(msg)

    with st.spinner(""):
        try:
            result = run_pipeline(
                query      = query,
                collection = collection,
                hf_token   = hf_token,
                gemini_key = gemini_key,
                status_callback = update_status,
            )
        except Exception as e:
            st.error(
                f"Pipeline error: {e}\n\n"
                "GreenMind already auto-switches across several Gemini "
                "models before giving up, so this usually means the Gemini "
                "API key is invalid/missing rather than a single model "
                "being unavailable. Double-check the key in the sidebar."
            )
            st.stop()

    status_box.empty()

    # ── Metrics Row ──────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(
            "🤖 LLM Used",
            "IBM Granite*" if hf_token else "Gemini",
            help="*Falls back automatically to Gemini for any step where "
                 "Granite is unavailable via HuggingFace.",
        )
    with m2:
        st.metric("📚 RAG Chunks", "Retrieved ✅")
    with m3:
        st.metric("🌐 Web Sources", "5 sources")
    with m4:
        st.metric("🎯 SDGs Covered", "17 SDGs")

    st.divider()

    # ── Agent Outputs (expandable) ───────────────────────────
    st.markdown("### 🔄 Agent Pipeline — Intermediate Outputs")
    with st.expander("🔍 Agent 1 — Search Agent Output", expanded=False):
        st.markdown(result["search_summary"])

    with st.expander("📚 Agent 2 — RAG Knowledge Agent Output", expanded=False):
        st.markdown(result["rag_summary"])
        with st.expander("📄 Raw RAG Context Retrieved", expanded=False):
            st.code(result["rag_context"], language="text")

    with st.expander("🔬 Agent 3 — Analysis Agent Output", expanded=False):
        st.markdown(result["analysis"])

    st.divider()

    # ── Final Report ─────────────────────────────────────────
    st.markdown("### 📋 SDG Intelligence Report")
    # NOTE: Wrapping multi-line markdown inside a raw <div ...>…</div> via
    # unsafe_allow_html is a long-standing Streamlit rendering gotcha
    # (streamlit/streamlit#859) — content after the first line can render
    # as literal "##"/"**" text instead of being parsed as markdown,
    # because the whole block is swallowed as one raw-HTML block. A
    # bordered container avoids that entirely and keeps normal markdown
    # rendering (headers, bold, code fences, etc.) fully intact.
    with st.container(border=True):
        st.markdown(result["report"])

    # Full downloadable report includes a sources/provenance footer so the
    # .md file is self-contained even outside the Streamlit UI.
    full_report = result["report"] + result.get("sources_footer", "")

    with st.expander("📎 Sources & Report Provenance", expanded=False):
        st.markdown(result.get("sources_footer", "_No source metadata available._"))

    # Download button
    st.download_button(
        label="⬇️ Download Report",
        data=full_report,
        file_name="greenmind_report.md",
        mime="text/markdown",
    )

elif run_btn and not query.strip():
    st.warning("Please enter a question first.")

else:
    # Landing state
    st.markdown("""
    <div style="text-align:center; padding: 2rem; color: #80cbc4;">
        <h3>🌱 How it works</h3>
        <p>Enter any climate, sustainability, or SDG-related question above.<br>
        Four specialised AI agents will research, analyse, and generate<br>
        a comprehensive intelligence report powered by IBM Granite & RAG.</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.info("**🔍 Search Agent**\nSearches live web for current data")
    with c2:
        st.info("**📚 RAG Agent**\nQueries all 17 SDG knowledge base")
    with c3:
        st.info("**🔬 Analysis Agent**\nCross-checks & identifies insights")
    with c4:
        st.info("**✍️ Writer Agent**\nGenerates structured SDG report")
