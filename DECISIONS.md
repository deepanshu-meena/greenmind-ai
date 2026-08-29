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

## Decision 2 — Fallback LLM: Google Gemini, with automatic multi-model fallback

**Chosen:** A ranked list of Gemini models (`gemini-3.7-flash` →
`gemini-3.6-flash` → `gemini-3.5-flash-lite` → `gemini-3.1-flash-lite` →
`gemini-2.5-flash` → `gemini-flash-latest`), called via the current
`google-genai` SDK. `agents.py` tries each candidate in order and only
moves on when a model returns an error or an empty response.

**Why a Fallback at All:**
HuggingFace free inference has rate limits, and IBM Granite 3.1 8B is
not consistently available through HF's auto-routed Inference
Providers (it is a community-requested model, not a guaranteed
serverless one). Without a fallback, the app breaks under load — or
whenever Granite isn't currently hosted by a provider. A fallback
ensures the app keeps producing reports even when Granite can't be
reached.

**Why auto-switch across *several* Gemini models instead of one:**
Google retires specific Gemini model IDs on a rolling basis with as
little as two weeks' notice for preview models (all `gemini-1.5-*`
and `gemini-2.0-*` IDs, including the `gemini-1.5-flash` this project
originally shipped with, are already shut down as of mid-2026). A
single hardcoded model name is a guaranteed future outage. Trying a
ranked list and only advancing on error means one retirement, one
regional outage, or one per-model quota limit no longer takes the
whole app down — newest/best model is always tried first, with
progressively cheaper/older stable models as safety nets, and
`gemini-flash-latest` (a rolling alias Google keeps pointed at its
current default Flash model) as the final catch-all.

**Trade-offs Accepted:**
- Two different model families' behaviours possible (Granite vs one
  of six Gemini candidates) — mitigated by strong system prompts that
  enforce consistent output format
- A worst-case query now makes up to ~12 model calls (2 retries × 6
  candidates) before failing outright — acceptable because normal
  operation succeeds on the first or second candidate, and this only
  matters when the Gemini key itself is bad
- The candidate list is a manually maintained config value, not
  auto-discovered from Google's `ListModels` endpoint — simpler and
  more predictable, at the cost of needing an occasional one-line
  update in `config.py` when Google's newest model changes

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

## Decision 4 — Embedding Model: Gemini `gemini-embedding-001`

**Chosen:** `gemini-embedding-001`, truncated to 768-dimensional vectors
via `output_dimensionality` (Matryoshka Representation Learning)

**Update (2026):** The project originally used `models/text-embedding-004`.
Google deprecated that model on January 14, 2026 and its replacement,
`gemini-embedding-001`, natively produces 3072-dim vectors. We keep
the original 768-dim footprint by passing `output_dimensionality=768`
in `EmbedContentConfig` — MRL means this truncation preserves most of
the semantic quality of the full vector while keeping ChromaDB
storage and query cost the same as before.

**Alternatives Considered:**
| Embedding Model | Reason Not Chosen |
|---|---|
| OpenAI text-embedding-3-small | Paid per token; not free; requires separate API key |
| sentence-transformers (all-MiniLM-L6-v2) | Free but downloads 80MB model on first run; slower cold start |
| Cohere Embed | Free tier limited; adds another API key dependency |
| IBM Granite Embeddings | Not available as a free standalone embedding API yet |
| Full 3072-dim `gemini-embedding-001` | 4× the ChromaDB storage for negligible retrieval-quality gain at our ~60-chunk scale |

**Why Gemini `gemini-embedding-001`:**
- **Same API key as fallback LLM** — no extra credential for the user
- Successor to the deprecated `text-embedding-004`, so this is the
  only currently-supported first-party Gemini embedding path
- Truncatable to 768-dim — strong semantic quality for SDG knowledge
  retrieval at the original storage footprint
- Supports `task_type="RETRIEVAL_DOCUMENT"` / `"RETRIEVAL_QUERY"` —
  optimised for RAG use case (documents and queries are now embedded
  with the correct, distinct task type — the original code used
  `"retrieval_document"` for both)
- Free tier sufficient for this project's scale

**Trade-offs Accepted:**
- Depends on Gemini API — if Gemini is down, embedding also fails (acceptable risk)
- Truncated embeddings are marginally lower-fidelity than the full
  3072-dim vector (acceptable trade for 4× less storage at our scale)
- Called via the new `google-genai` SDK, not the deprecated
  `google-generativeai` package (see Decision 9)

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

**Chosen:** `ddgs` Python library (formerly published as `duckduckgo-search`,
which was renamed upstream — the old package name now just re-exports
from `ddgs` with a deprecation warning)

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
- `ddgs` (the renamed successor to `duckduckgo-search`) is actively maintained and pip-installable

**Trade-offs Accepted:**
- Results occasionally less relevant than Google — mitigated by strong Agent 1 prompt that filters noise
- No news freshness filter available — results may include older articles
- No image or structured data — text-only results (sufficient for our use case)
- DuckDuckGo's free/unauthenticated endpoint rate-limits aggressively
  under repeated use (`RatelimitException`, HTTP 202) — mitigated with
  a short retry-with-backoff in `web_search.py`; if all retries are
  exhausted, the pipeline continues using RAG + web-search-error text
  rather than failing the whole report

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

## Decision 9 — SDK: `google-genai` instead of the deprecated `google-generativeai`

**Chosen:** `google-genai` (`from google import genai`) for both
generation and embeddings

**Context:**
The project originally used `google.generativeai`
(`import google.generativeai as genai`). Google deprecated that
package on August 31, 2025 in favor of a single unified SDK,
`google-genai`, that covers Gemini, embeddings, image, and video
models with one consistent client interface
(`genai.Client(...).models.generate_content(...)` /
`.embed_content(...)`).

**Why migrate now rather than leave the deprecated SDK in place:**
- The deprecated package no longer receives updates, so it has no
  path to support current/future model IDs as Google's naming and
  feature set evolve
- The unified client interface is what every current Gemini code
  sample and piece of documentation uses, which matters for a
  student portfolio project meant to be readable by reviewers
- No functional downside — same free tier, same API key, near-identical
  call shape

**Trade-offs Accepted:**
- One-time migration effort across `agents.py` and `knowledge_base.py`
- Reviewers/graders comparing against older Gemini tutorials online
  will see a different import style (`from google import genai`
  instead of `import google.generativeai as genai`) — noted here and
  in the README so it isn't mistaken for an error

---

## Decision 10 — Report Quality Fixes: grounding, recency, and rendering

**Context:** After the SDK/model migration (Decisions 2, 4, 9), a live
test run surfaced three separate issues worth documenting because they
affect the *quality* of the output, not just whether the app runs:

**10a — Writer Agent was only seeing ~20% of the research it was given.**
`writer_agent()` truncated `search_summary`, `rag_summary`, and
`analysis` to 400 characters each before writing the report — but
Agents 1–3 are each instructed to write up to 300 words (~1,800
characters). The final report was effectively being written from the
first two sentences of each upstream agent's output, which pushed it
toward generic, textbook-style language instead of the specific
findings actually retrieved. **Fix:** pass the full summaries through.

**10b — The web search query was permanently biased toward 2024–2025.**
`orchestrator.py` hardcoded `f"{query} sustainability SDG 2024 2025"`
regardless of when the app is actually run, actively steering
DuckDuckGo results toward stale content the longer the project exists
past that window. Same issue in the Search Agent's prompt, which
literally asked for "Recent developments (2023-2024)". **Fix:** both
now derive the current year at runtime (`datetime.now(timezone.utc).year`).

**10c — Report rendering used a fragile Streamlit HTML pattern.**
The original `app.py` wrapped the markdown report in a raw
`<div class="report-box">{report}</div>` via
`unsafe_allow_html=True`. Per CommonMark's HTML-block rules (and a
long-standing, confirmed Streamlit issue —
[streamlit/streamlit#859](https://github.com/streamlit/streamlit/issues/859)),
multi-line content opening with a raw HTML tag and no blank line
separator can get swallowed as one literal HTML block, so headers,
bold text, and code fences inside it may render as plain `##`/`**`
text instead of being parsed as markdown. **Fix:** replaced with
`st.container(border=True)` + plain `st.markdown(report)` — a
Streamlit-native bordered box that doesn't fight the markdown
renderer.

**10d — No source attribution in the exported file.**
The Streamlit UI shows retrieved web links and RAG chunks in
expandable panels, but the downloaded `.md` report itself had no
citations — a report shared outside the app (e.g. submitted as a
capstone deliverable) had specific-sounding statistics with no way to
verify where they came from. **Fix:** `orchestrator.py` now builds a
"Sources & Report Provenance" footer (live web URLs actually
retrieved, SDG Wikipedia articles actually used by RAG, generation
timestamp, model used) and appends it to every downloaded report.
The Writer Agent's system prompt was also tightened to ground
statistics in the provided research rather than inventing
precise-sounding numbers.

**Trade-offs Accepted:**
- Passing full (rather than truncated) summaries into the Writer
  Agent's prompt increases token usage per report — acceptable given
  all Gemini fallback candidates and Granite 8B have ample context
  windows for a few thousand extra characters
- The sources footer is best-effort: if DuckDuckGo was rate-limited
  for that query, the footer says so explicitly rather than silently
  omitting web sources

---

<div align="center">
<em>Every decision above was made by weighing accuracy, cost, reliability, and explainability.</em><br>
<strong>Context → Thinking → Impact</strong>
</div>
