# 🧠 DECISIONS.md — GreenMind AI
### Every Technical Decision Explained with Reasoning & Trade-offs

> This document follows the principle: *"Context. Thinking. Impact."*
> Every choice here has a reason, an alternative considered, and a trade-off acknowledged.

---

## Decision 1 — Primary LLM: IBM Granite 3.1 8B

**Chosen:** `ibm-granite/granite-3.1-8b-instruct` via HuggingFace Inference API

**Alternatives Considered:**
| Model | Reason Not Chosen |
|---|---|
| GPT-4o (OpenAI) | Paid API, not free for student use, not aligned with IBM internship requirement |
| Gemini 1.5 Pro | Good quality but overkill for agent tasks; Granite is required by 1M1B internship |
| LLaMA 3.1 8B | Open source but not IBM-aligned; no enterprise sustainability credentials |
| Mistral 7B | Strong model but no IBM sustainability/responsible AI branding |

**Why IBM Granite:**
- Directly required by the 1M1B × IBM SkillsBuild internship program
- IBM Granite is trained with **responsible AI principles** — critical for an SDG-focused project
- IBM has made public commitments to sustainability and ethical AI, aligning with project values
- Enterprise-grade reliability and safety guardrails built-in
- Available free via HuggingFace Inference API — no cost for student developers

**Trade-offs Accepted:**
- Granite 3.1 8B is smaller than GPT-4o — output quality slightly lower on very complex queries
- HuggingFace free tier has rate limits — mitigated by Gemini fallback
- Response slightly slower than Gemini Flash — acceptable for a report-generation use case

**Impact:** Project is aligned with internship requirements AND demonstrates responsible AI use — both evaluated by mentors.

---

## Decision 2 — Fallback LLM: Google Gemini 1.5 Flash

**Chosen:** `gemini-1.5-flash` via Google Generative AI SDK

**Why a Fallback at All:**
HuggingFace free inference has rate limits. Without a fallback, the app breaks under load. A fallback ensures 100% uptime for demos and evaluations.

**Why Gemini Flash specifically:**
- Already integrated in other projects (ATS Optimizer, Career Dashboard) — same API key
- Extremely fast (sub-2-second responses) — ideal as fallback
- Free tier is generous (60 RPM, 1500 RPD)
- Strong instruction-following for structured report generation

**Trade-offs Accepted:**
- Two different model behaviours possible (Granite vs Gemini) — mitigated by strong system prompts that enforce consistent output format
- Slight inconsistency in tone between models — acceptable since structure is enforced by prompts

---

## Decision 3 — Vector Store: ChromaDB (Persistent)

**Chosen:** `chromadb.PersistentClient` stored in `./data/greenmind_db/`

**Alternatives Considered:**
| Vector Store | Reason Not Chosen |
|---|---|
| Pinecone | Paid after free tier; requires cloud account; adds dependency |
| Weaviate | More complex setup; cloud-first; overkill for this scale |
| FAISS (Facebook) | In-memory only by default; no metadata support; harder to query |
| Qdrant | Excellent but requires Docker for local use; too heavy for a student project |
| pgvector (Postgres) | Requires PostgreSQL setup; too much infrastructure overhead |

**Why ChromaDB:**
- **Free and fully local** — no cloud account, no API key, no cost
- **Persistent storage** — knowledge base survives app restarts (critical for demo reliability)
- **Cosine similarity** built-in — correct metric for semantic text retrieval
- **Metadata filtering** — can filter chunks by SDG ID, source, or topic
- **Python-native API** — clean integration with the rest of the stack
- **No Docker needed** — installs via pip, works immediately

**Trade-offs Accepted:**
- Not scalable to millions of vectors (sufficient for ~60 SDG chunks)
- No cloud sync (fine for local + Streamlit Cloud deployment)
- Slightly slower than FAISS for very large collections (irrelevant at our scale)

**Impact:** Knowledge base loads once, persists across all sessions, zero ongoing cost.

---

## Decision 4 — Embedding Model: Gemini text-embedding-004

**Chosen:** `models/text-embedding-004` — 768-dimensional vectors

**Alternatives Considered:**
| Embedding Model | Reason Not Chosen |
|---|---|
| OpenAI text-embedding-3-small | Paid per token; not free; requires separate API key |
| sentence-transformers (all-MiniLM-L6-v2) | Free but downloads 80MB model on first run; slower cold start |
| Cohere Embed | Free tier limited; adds another API key dependency |
| IBM Granite Embeddings | Not available as a free standalone embedding API yet |

**Why Gemini text-embedding-004:**
- **Same API key as fallback LLM** — no extra credential for the user
- 768-dim vectors — strong semantic quality for SDG knowledge retrieval
- Fast inference — embedding 60 chunks takes ~15 seconds
- Supports `task_type="retrieval_document"` — optimised for RAG use case
- Free tier sufficient: 1500 embeddings/day

**Trade-offs Accepted:**
- Depends on Gemini API — if Gemini is down, embedding also fails (acceptable risk)
- 768-dim uses more ChromaDB storage than 384-dim models (negligible at our scale)

**Chunking Strategy Chosen:** Word-based, 400 words per chunk, 40-word overlap
- 400 words preserves enough context for semantic meaning
- 40-word overlap prevents information loss at chunk boundaries
- Word-based (not character-based) ensures chunks don't split mid-sentence awkwardly

---

## Decision 5 — Knowledge Source: Wikipedia REST API

**Chosen:** Wikipedia `w/api.php` with `extracts` + `explaintext` parameters

**Alternatives Considered:**
| Knowledge Source | Reason Not Chosen |
|---|---|
| UN Official SDG Reports (PDFs) | Requires PDF parsing; complex extraction; large file sizes |
| IPCC Reports | Extremely large (500+ pages); complex; overkill for conversational RAG |
| Paid databases (Scopus, etc.) | Paid; not accessible for student project |
| Hardcoded text chunks | Static; outdated quickly; poor coverage |
| ArXiv papers on SDGs | Too technical/academic; less accessible for policy/general queries |

**Why Wikipedia:**
- **Completely free and public** — no copyright issues, safe to use in GitHub projects
- **Comprehensive SDG coverage** — dedicated Wikipedia articles for all 17 SDGs
- **Regularly updated** — reflects current consensus and recent developments
- **Clean plain text output** — `explaintext=True` gives structured, parseable content
- **REST API** — simple HTTP request, no library dependency needed

**Trade-offs Accepted:**
- Wikipedia articles can have inaccuracies — mitigated by cross-referencing with live web search (Agent 1)
- Articles vary in depth (SDG 13 Climate Action is much richer than SDG 16) — acceptable
- Capped at 6,000 characters per article to manage embedding costs — some detail is lost

**Impact:** Zero cost, zero copyright risk, comprehensive coverage, GitHub-safe.

---

## Decision 6 — Web Search: DuckDuckGo Search API

**Chosen:** `duckduckgo-search` Python library

**Alternatives Considered:**
| Search API | Reason Not Chosen |
|---|---|
| Google Custom Search API | Free tier only 100 queries/day; requires API key + Custom Search Engine setup |
| Bing Search API | Paid after free trial; Microsoft Azure account needed |
| SerpAPI | Paid; $50/month after trial |
| Tavily Search API | Paid; specifically for LLMs but requires key |
| NewsAPI | Limited to news; not general sustainability queries |

**Why DuckDuckGo:**
- **Zero API key required** — users can run the app immediately without extra setup
- **No rate limits** for reasonable use
- **Privacy-first** — no user tracking, aligns with project ethics
- **Good result quality** for sustainability/SDG queries
- `duckduckgo-search` library is actively maintained and pip-installable

**Trade-offs Accepted:**
- Results occasionally less relevant than Google — mitigated by strong Agent 1 prompt that filters noise
- No news freshness filter available — results may include older articles
- No image or structured data — text-only results (sufficient for our use case)

---

## Decision 7 — UI Framework: Streamlit

**Chosen:** Streamlit with custom CSS overrides

**Alternatives Considered:**
| Framework | Reason Not Chosen |
|---|---|
| Flask + HTML/CSS/JS | Requires frontend development; much more code; not Deepanshu's focus |
| Gradio | Limited customisation; less control over layout and styling |
| FastAPI + React | Full-stack; overkill for a portfolio/demo project |
| Dash (Plotly) | More suited for data dashboards; less suited for chat/report interfaces |

**Why Streamlit:**
- **Python-native** — no JavaScript required; consistent with AI/ML stack
- **Rapid deployment** — from code to running app in minutes
- **Streamlit Cloud** — free deployment with GitHub integration
- **`st.session_state`** — supports stateful interactions (sidebar sample queries)
- **`st.status()`** — real-time agent progress updates look impressive in demos
- **Custom CSS** — enough flexibility for dark sustainability-themed UI
- Industry-standard for ML/AI demo apps — familiar to technical recruiters

**Trade-offs Accepted:**
- Not as polished as a React frontend — fine for internship submission and portfolio
- Limited multi-page navigation — single-page app is sufficient for this use case
- Streamlit Cloud has 1GB RAM limit — sufficient for ChromaDB at our scale

---

## Decision 8 — Agent Architecture: Custom Sequential Pipeline (not LangChain Agents)

**Chosen:** Custom `orchestrator.py` that calls each agent function sequentially

**Alternatives Considered:**
| Approach | Reason Not Chosen |
|---|---|
| LangChain AgentExecutor | Version instability; complex debugging; hides what's actually happening |
| CrewAI | Newer framework; less stable; adds heavy dependency |
| AutoGen (Microsoft) | Complex setup; better for multi-turn conversations, not one-shot reports |
| LangGraph | Excellent but steep learning curve; overkill for 4 sequential agents |

**Why Custom Pipeline:**
- **Full control** — every agent input/output is explicit and inspectable in the UI
- **Easier to debug** — no framework magic hiding errors
- **Shows understanding** of agentic architecture (more impressive than using a framework)
- **Lighter dependency** — no `langchain` (100+ MB) needed
- **Predictable** — sequential pipeline is reliable for report generation use case
- Interviewers can ask *"how do your agents work?"* and the answer is clear

**Trade-offs Accepted:**
- Agents cannot loop back or retry on failure (acceptable for v1)
- No parallel agent execution — sequential is slower but more readable
- Cannot handle very long multi-turn conversations — single-shot report generation only

**Impact:** Architecture is defensible in interviews, easy to explain, and demonstrates real understanding of agentic AI beyond just using a library.

---

<div align="center">
<em>Every decision above was made by weighing accuracy, cost, reliability, and explainability.</em><br>
<strong>Context → Thinking → Impact</strong>
</div>
