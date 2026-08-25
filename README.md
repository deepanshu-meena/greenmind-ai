<div align="center">

# 🌍 GreenMind AI
### Multi-Agent SDG Intelligence System

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red?style=flat-square&logo=streamlit)](https://streamlit.io)
[![IBM Granite](https://img.shields.io/badge/IBM%20Granite-3.1%208B-054ADA?style=flat-square&logo=ibm)](https://huggingface.co/ibm-granite)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-RAG-orange?style=flat-square)](https://chromadb.ai)
[![AICTE](https://img.shields.io/badge/AICTE-Certified%20Internship-green?style=flat-square)](https://internship.aicte-india.org)
[![SDG 13](https://img.shields.io/badge/Primary%20SDG-13%20Climate%20Action-3F7E44?style=flat-square)](https://sdgs.un.org/goals/goal13)

**An AI-powered sustainability intelligence assistant that answers any climate or SDG-related question by combining live web search with a scientific RAG knowledge base — generating a structured, downloadable action report in under 60 seconds.**

> *"How might we use AI to synthesise fragmented SDG and climate knowledge in real time, so that policymakers, students, and communities can make more informed and sustainable decisions?"*

*Built for: 1M1B × IBM SkillsBuild AI for Sustainability Internship (AICTE, 2026)*

</div>

---

## 📌 Table of Contents
- [What I Built](#-what-i-built)
- [SDG Alignment](#-sdg-alignment)
- [Target Users](#-target-users)
- [Architecture](#-architecture)
- [Results with Actual Numbers](#-results-with-actual-numbers)
- [The 4-Agent Pipeline](#-the-4-agent-pipeline)
- [Technical Decisions](#-technical-decisions)
- [Responsible AI Considerations](#-responsible-ai-considerations)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Setup & Installation](#-setup--installation)
- [How to Use](#-how-to-use)
- [Expected Impact](#-expected-impact)
- [IBM Technology Used](#-ibm-technology-used)
- [Future Scope](#-future-scope)
- [Author](#-author)

---

## 🔨 What I Built

### The Problem
Climate change and sustainable development are among the most complex challenges of our time. The world has 17 UN Sustainable Development Goals with a 2030 deadline — yet actionable intelligence remains siloed across hundreds of UN reports, Wikipedia articles, IPCC assessments, and news sources.

**Who is affected?** Students, NGOs, policymakers, educators, and researchers who need to make sustainability decisions daily — but lack time to cross-reference scattered sources. Communities in climate-vulnerable regions are most impacted when decision-makers lack fast, reliable SDG intelligence.

**Why does this problem persist?** There is no single system that can answer *"what is the current state of SDG 13, what are the key challenges, and what can AI do about it?"* — instantly, accurately, and from verified sources.

**Why is AI needed?** Manual research across UN reports, Wikipedia, and live news takes hours. AI enables automation of retrieval, synthesis, cross-verification, and structured report generation — compressing hours of research into under 60 seconds, at scale, accessible to anyone.

### The Solution
GreenMind AI is a **4-agent AI pipeline** powered by IBM Granite that:
1. Searches the **live web** for the most current sustainability data
2. Queries a **RAG knowledge base** pre-loaded with all 17 SDG Wikipedia articles in ChromaDB
3. **Cross-analyses** both sources to extract critical, quantified insights
4. **Generates a structured SDG Intelligence Report** with 6 labelled sections, downloadable as Markdown

---

## 🎯 SDG Alignment

### Primary SDG
**🌡️ SDG 13 — Climate Action**
GreenMind AI directly addresses SDG 13 by making climate science, policy data, and action intelligence instantly accessible to anyone — enabling faster, evidence-based climate decisions.

### Secondary SDGs
| SDG | Goal | How GreenMind Addresses It |
|---|---|---|
| **SDG 7** | Affordable & Clean Energy | Surfaces clean energy data, transition pathways, and AI solutions |
| **SDG 15** | Life on Land | Covers biodiversity loss, deforestation, and ecosystem intelligence |
| **SDG 14** | Life Below Water | Ocean conservation data and marine biodiversity insights |
| **SDG 4** | Quality Education | Makes complex SDG knowledge accessible to anyone, anywhere, free |
| **SDG 11** | Sustainable Cities | Urban sustainability planning and smart city intelligence |
| **SDG 17** | Partnerships for Goals | Equips NGOs, researchers & policymakers with aligned intelligence |

---

## 👥 Target Users

| User | Problem They Face | How GreenMind Helps |
|---|---|---|
| **Students & Researchers** | No single tool for SDG intelligence | Get source-grounded reports in seconds |
| **NGOs & Non-Profits** | Limited resources for research | Free, instant sustainability intelligence |
| **Policymakers** | Need fast, verified climate data | Cross-verified reports from science + live web |
| **Educators** | Teaching SDGs requires current data | Downloadable, structured reports for classroom use |
| **Journalists** | Covering climate needs fast fact-checking | Multi-source verified intelligence reports |
| **Community Leaders** | Lack access to expert sustainability advice | Plain-language SDG reports on any local issue |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    USER (Streamlit UI)                       │
│              Enters sustainability/SDG question              │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      ORCHESTRATOR                            │
│           Runs 4 agents in sequence, passes outputs          │
└────┬──────────────┬──────────────┬──────────────┬───────────┘
     │              │              │              │
     ▼              ▼              ▼              ▼
┌─────────┐   ┌──────────┐  ┌──────────┐  ┌──────────┐
│ AGENT 1 │   │ AGENT 2  │  │ AGENT 3  │  │ AGENT 4  │
│ Search  │   │   RAG    │  │ Analysis │  │  Writer  │
│  Agent  │   │  Agent   │  │  Agent   │  │  Agent   │
└────┬────┘   └────┬─────┘  └────┬─────┘  └────┬─────┘
     │              │             │              │
     ▼              ▼             │              ▼
┌─────────┐   ┌──────────┐       │      ┌──────────────┐
│Duck-    │   │ChromaDB  │       │      │  Structured  │
│DuckGo   │   │Persistent│───────┘      │ SDG Report   │
│Search   │   │RAG Store │              │ (Download)   │
└─────────┘   └────┬─────┘              └──────────────┘
                   │
         ┌─────────▼──────────┐
         │   Knowledge Base   │
         │  17 SDG Wikipedia  │
         │  Articles Embedded │
         │  Gemini Embeddings │
         │ gemini-embedding-001│
         └────────────────────┘

Primary LLM : IBM Granite 3.1 8B (via HuggingFace Inference Providers)
Fallback LLM: Gemini 3.7 Flash, auto-switching down through 3.6 Flash →
              3.5 Flash-Lite → 3.1 Flash-Lite → 2.5 Flash → gemini-flash-latest
              on any error, so no single retired/rate-limited model can
              break the app
```

---

## 📊 Results with Actual Numbers

| Metric | Value |
|---|---|
| 🌐 **SDGs Covered in Knowledge Base** | **17 / 17** |
| 📄 **Knowledge Chunks in ChromaDB** | **~60 chunks** (400 words each, 40-word overlap) |
| 🔍 **Live Web Sources per Query** | **5 sources** (DuckDuckGo, real-time) |
| 🎯 **RAG Chunks Retrieved per Query** | **Top 4** (cosine similarity search) |
| 🤖 **Agents in Pipeline** | **4 specialised agents** |
| 📋 **Report Sections Generated** | **6 structured sections** |
| ⏱️ **Average Pipeline Completion** | **~45–60 seconds** |
| 📐 **Embedding Dimensions** | **768-dim** (Gemini `gemini-embedding-001`, truncated via Matryoshka from its native 3072-dim) |
| 💾 **KB Rebuild Required** | **Once** (persistent ChromaDB, instant on reload) |

---

## 🤖 The 4-Agent Pipeline

### Agent 1 — 🔍 Search Agent
- Queries **DuckDuckGo** with user question + sustainability/SDG/2024 context
- Fetches 5 live web results (news, UN reports, research articles)
- IBM Granite extracts: key facts, recent statistics, credible sources
- **Output:** Structured summary of current real-world state (≤300 words)

### Agent 2 — 📚 RAG Knowledge Agent
- Embeds user query using **Gemini `gemini-embedding-001`** (768-dim vector, task-typed as `RETRIEVAL_QUERY`)
- Searches **ChromaDB** — retrieves Top 4 semantically relevant chunks from 17 SDG articles
- IBM Granite synthesises: scientific consensus, SDG targets, UN data, key metrics
- **Output:** Evidence-based scientific perspective grounded in SDG knowledge (≤300 words)

### Agent 3 — 🔬 Analysis Agent
- Receives combined outputs from Agent 1 (live) + Agent 2 (science)
- IBM Granite cross-examines: agreements, contradictions, knowledge gaps
- Quantifies impact where possible (%, temperatures, years, costs)
- Identifies specifically **where AI can intervene**
- **Output:** Critical analytical insights with SDG linkages (≤300 words)

### Agent 4 — ✍️ Writer Agent
- Synthesises all three agent outputs into one coherent context
- IBM Granite generates fully structured markdown report
- **6 sections:** Executive Summary · Key Facts · Relevant SDGs · Challenges · AI Solutions · Recommendations
- **Output:** Complete downloadable `.md` intelligence report

---

## 🧠 Technical Decisions

Every key technical choice is documented with reasoning and trade-offs.

📄 **[Read DECISIONS.md →](DECISIONS.md)**

| Decision | Chosen | Key Reason |
|---|---|---|
| Primary LLM | IBM Granite 3.1 8B | Required by internship; responsible AI; enterprise-grade |
| Fallback LLM | Gemini 3.7 Flash → 3.6 Flash → 3.5 Flash-Lite → 3.1 Flash-Lite → 2.5 Flash → `gemini-flash-latest` | Auto-switches on any error so a single retired/rate-limited model never breaks the app |
| Vector Store | ChromaDB | Free, local, persistent, no API key needed |
| Embedding Model | Gemini `gemini-embedding-001` | Successor to the retired `text-embedding-004`; 768-dim (truncated), high quality, already in stack |
| Knowledge Source | Wikipedia API | Free, public, comprehensive SDG coverage |
| Web Search | DuckDuckGo (`ddgs` package) | No API key required, privacy-first |
| UI Framework | Streamlit | Python-native, rapid ML app deployment |

---

## 🤝 Responsible AI Considerations

### Fairness
- GreenMind uses **Wikipedia** (open, community-maintained, globally neutral) and **DuckDuckGo** (no personalisation, no filter bubbles) as knowledge sources
- No demographic data, location data, or user profiling is involved at any stage
- The knowledge base covers **all 17 SDGs equally** — no SDG or region is deprioritised

### Transparency
- All **4 agent outputs are visible and expandable** in the Streamlit UI — users can inspect exactly how the final report was constructed at every step
- The system explicitly shows: which web sources were found, which RAG chunks were retrieved, what the analysis concluded — before generating the final report
- IBM Granite is the primary model; the UI indicates when Gemini fallback is used instead
- **The downloadable report is self-contained**: every `.md` export ends with a "Sources & Report Provenance" section listing the live web URLs and SDG Wikipedia articles actually retrieved for that query, plus a generation timestamp and which model produced it — so the report stands on its own outside the app, not just inside the expandable UI panels

### Ethics
- No harmful, discriminatory, or misleading content is generated — IBM Granite's responsible AI guardrails are active
- The system does not make political recommendations — it presents verified, source-grounded scientific and policy information only
- Agent system prompts explicitly instruct models to be factual, cite sources, and avoid speculation

### Privacy
- **Zero personal data is collected.** No login, no user tracking, no session storage, no cookies
- **BYOK (Bring Your Own Key)** architecture — API keys are user-owned, entered locally, never stored in the codebase or transmitted to any third party
- `.gitignore` ensures API keys in `secrets.toml` are never pushed to GitHub
- ChromaDB stores only Wikipedia text chunks — no user query history is persisted

---

## 🛠️ Tech Stack

| Layer | Technology | Version |
|---|---|---|
| Primary LLM | IBM Granite 3.1 8B Instruct (HuggingFace Inference Providers) | granite-3.1-8b-instruct |
| Fallback LLM | Google Gemini, newest-first with automatic fallback | gemini-3.7-flash → gemini-3.6-flash → gemini-3.5-flash-lite → gemini-3.1-flash-lite → gemini-2.5-flash → gemini-flash-latest |
| SDK | Google Gen AI SDK (`google-genai`) | ≥ 1.20.0 — the old `google-generativeai` package is deprecated |
| Embeddings | Gemini `gemini-embedding-001` | 768-dim vectors (Matryoshka-truncated from 3072) |
| Vector Store | ChromaDB (Persistent) | ≥ 0.5.0 |
| Knowledge Source | Wikipedia REST API | Public, free |
| Web Search | DuckDuckGo (`ddgs` package) | No key required |
| UI | Streamlit | ≥ 1.38.0 |
| Language | Python | 3.10+ |

### ⚠️ Troubleshooting: "models/gemini-... is not found for API version v1beta"

Google frequently retires Gemini model IDs (all `gemini-1.5-*` and
`gemini-2.0-*` models are already shut down as of mid-2026). If this
error ever comes back, it means Google retired every model currently
listed in `GEMINI_FLASH_CANDIDATES` in `config.py`. Fix: open
`config.py`, check the current model list at
[ai.google.dev/gemini-api/docs/models](https://ai.google.dev/gemini-api/docs/models),
and add the new model ID to the **front** of the `GEMINI_FLASH_CANDIDATES`
list. No other code changes are needed — `agents.py` will pick it up
automatically.

---

## 📁 Project Structure

```
greenmind-ai/
│
├── app.py                  # Streamlit UI — main entry point
├── orchestrator.py         # Coordinates 4-agent pipeline
├── agents.py               # All 4 agent definitions (IBM Granite + Gemini)
├── knowledge_base.py       # Wikipedia fetch → chunk → embed → ChromaDB
├── web_search.py           # DuckDuckGo wrapper
├── config.py               # API keys, model names, SDG topics, constants
│
├── DECISIONS.md            # Every technical decision explained with trade-offs
│
├── data/                   # Auto-created, ignored by Git
│   └── greenmind_db/       # Persistent ChromaDB vector store
│
├── .streamlit/
│   └── secrets.toml        # API keys (NEVER pushed to GitHub)
│
├── .gitignore              # Protects secrets and data folder
└── requirements.txt        # Python dependencies
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.10+
- **Gemini API Key** (free) → [aistudio.google.com](https://aistudio.google.com/app/apikey)
- **HuggingFace Token** (free) → [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) → Read access

### Step 1 — Clone
```bash
git clone https://github.com/deepanshu-meena/greenmind-ai.git
cd greenmind-ai
```

### Step 2 — Install
```bash
pip install -r requirements.txt
```

### Step 3 — Add API Keys
Edit `.streamlit/secrets.toml`:
```toml
GEMINI_API_KEY = "your-gemini-api-key-here"
HF_TOKEN       = "your-huggingface-token-here"
```

### Step 4 — Run
```bash
streamlit run app.py
```

> **⏳ First Run:** Knowledge base loads all 17 SDG Wikipedia articles (~60s). All subsequent runs load instantly.

---

## 🚀 How to Use

1. Open the app at `http://localhost:8501`
2. Enter API keys in the sidebar
3. Type any sustainability, climate, or SDG question
4. Click **"Generate Intelligence Report"**
5. Watch 4 agents work in real-time
6. Read or download the final structured report

**Sample Questions:**
- *"What is the impact of climate change on global food security?"*
- *"How can AI help achieve SDG 7 clean energy goals by 2030?"*
- *"What are the biggest threats to ocean biodiversity?"*
- *"How does deforestation contribute to carbon emissions?"*
- *"What is the current progress on the Paris Agreement 1.5°C target?"*

---

## 💥 Expected Impact

### If GreenMind AI is implemented at scale:

**Social Impact**
- Students and communities in climate-vulnerable regions gain free access to expert-level SDG intelligence
- Educators can generate current, structured SDG lesson material in seconds
- NGOs with limited research budgets can make faster, evidence-based sustainability decisions

**Environmental Impact**
- Faster intelligence → faster climate action decisions by policymakers
- Awareness tool for SDG 13 helps communities understand and act on local climate risks
- Reduces carbon cost of lengthy manual research processes

**Economic Impact**
- Free tool eliminates research costs for small NGOs and community organisations
- Policy decisions backed by AI intelligence reduce cost of uninformed sustainability choices
- Scalable to any country, language, or SDG topic without additional infrastructure cost

**Who Benefits Most:**
Students → NGOs → Local policymakers → Climate researchers → Educators → Community leaders

---

## 🔵 IBM Technology Used

| IBM Technology | How It's Used |
|---|---|
| **IBM Granite 3.1 8B Instruct** | Primary LLM powering all 4 agents |
| **HuggingFace Inference API** | Access layer for IBM Granite (free tier) |
| **IBM SkillsBuild Curriculum** | AI/ML, LLMs, RAG, and Agentic AI modules completed |

IBM Granite was chosen for its **enterprise-grade reliability**, **responsible AI principles**, and direct alignment with IBM's commitment to ethical and sustainable AI — making it the natural fit for an SDG-focused internship project.

---

## 🔮 Future Scope

- **Real-time SDG Progress Tracker** — integrate UN SDG API for live 2030 goal progress data
- **IBM WatsonX Integration** — upgrade from HuggingFace to full WatsonX platform
- **Multi-language Support** — SDG intelligence in Hindi and other regional languages
- **Voice Interface** — Whisper + TTS for voice-based sustainability queries
- **City-Level Reports** — customise intelligence for specific geographies
- **Carbon Footprint Module** — personal sustainability impact calculator

---

## 👨‍💻 Author

**Deepanshu Meena**
B.Tech Software Engineering, Delhi Technological University (DTU)

[![Email](https://img.shields.io/badge/Email-deepanshumeena545@gmail.com-red?style=flat-square&logo=gmail)](mailto:deepanshumeena545@gmail.com)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat-square&logo=linkedin)](https://linkedin.com/in/YOUR_PROFILE)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat-square&logo=github)](https://github.com/deepanshu-meena)

---

<div align="center">
<strong>🌍 Built for a Sustainable Future · Powered by IBM Granite & Agentic AI</strong><br>
<em>1M1B × IBM SkillsBuild AI for Sustainability Internship | AICTE 2026</em><br>
<em>Primary SDG: SDG 13 — Climate Action</em>
</div>
