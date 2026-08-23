"""
agents.py
Four specialised agents for GreenMind AI.
Primary LLM  : IBM Granite 3.1 8B via HuggingFace Inference API
Fallback LLM : Google Gemini 1.5 Flash
"""

import google.generativeai as genai
from huggingface_hub import InferenceClient
from config import IBM_GRANITE_MODEL, GEMINI_FLASH


# ── LLM Wrappers ─────────────────────────────────────────────

def _granite(prompt: str, system: str, hf_token: str) -> str:
    """Call IBM Granite via HuggingFace Inference API."""
    client = InferenceClient(token=hf_token)
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": prompt},
    ]
    resp = client.chat.completions.create(
        model=IBM_GRANITE_MODEL,
        messages=messages,
        max_tokens=900,
        temperature=0.3,
    )
    return resp.choices[0].message.content


def _gemini(prompt: str, system: str, gemini_key: str) -> str:
    """Call Gemini 1.5 Flash as fallback."""
    genai.configure(api_key=gemini_key)
    model = genai.GenerativeModel(
        model_name=GEMINI_FLASH,
        system_instruction=system,
    )
    return model.generate_content(prompt).text


def llm_call(prompt: str, system: str,
             hf_token: str, gemini_key: str) -> str:
    """Try IBM Granite first; fall back to Gemini on error."""
    if hf_token:
        try:
            return _granite(prompt, system, hf_token)
        except Exception:
            pass
    return _gemini(prompt, system, gemini_key)


# ── Agent 1 : Search Agent ───────────────────────────────────

def search_agent(query: str, web_results: str,
                 hf_token: str, gemini_key: str) -> str:
    system = (
        "You are a Climate & Sustainability Search Agent. "
        "Analyse web search results and extract the most relevant, "
        "factual, up-to-date information about climate, sustainability, "
        "and the UN SDGs. Be concise and precise."
    )
    prompt = f"""Query: {query}

Web Search Results:
{web_results}

Extract and summarise:
1. Key facts and statistics
2. Recent developments (2023-2024)
3. Most credible sources
4. Specific data points or metrics

Provide a structured summary (max 300 words)."""
    return llm_call(prompt, system, hf_token, gemini_key)


# ── Agent 2 : RAG Knowledge Agent ────────────────────────────

def rag_agent(query: str, rag_context: str,
              hf_token: str, gemini_key: str) -> str:
    system = (
        "You are a UN SDG Scientific Knowledge Agent. "
        "You have access to official SDG and climate science documents. "
        "Provide evidence-based, factual insights grounded in the "
        "retrieved knowledge. Always reference the SDG being discussed."
    )
    prompt = f"""Query: {query}

Retrieved SDG Knowledge Base Context:
{rag_context}

Based on this knowledge, provide:
1. Scientific consensus and key findings
2. Relevant SDG targets and indicators
3. Quantitative thresholds or goals (e.g., 1.5°C, 30x30, etc.)
4. Most critical challenges identified

Be precise and evidence-based (max 300 words)."""
    return llm_call(prompt, system, hf_token, gemini_key)


# ── Agent 3 : Analysis Agent ─────────────────────────────────

def analysis_agent(query: str, search_summary: str, rag_summary: str,
                   hf_token: str, gemini_key: str) -> str:
    system = (
        "You are a Sustainability Impact Analysis Agent. "
        "Cross-examine information from multiple sources, "
        "identify patterns, quantify impacts, and surface "
        "the most critical AI-driven insights for sustainability."
    )
    prompt = f"""Original Query: {query}

🔍 Web Research Summary:
{search_summary}

📚 Scientific Knowledge Summary:
{rag_summary}

Perform a cross-source analysis:
1. Key agreements between sources
2. Data gaps or contradictions
3. Most impactful insight (quantified where possible)
4. Role AI can play in solving this challenge
5. Which SDGs are directly linked

(max 300 words)"""
    return llm_call(prompt, system, hf_token, gemini_key)


# ── Agent 4 : Report Writer Agent ────────────────────────────

def writer_agent(query: str, search_summary: str,
                 rag_summary: str, analysis: str,
                 hf_token: str, gemini_key: str) -> str:
    system = (
        "You are a UN SDG Intelligence Report Writer. "
        "Create comprehensive, well-structured, actionable reports "
        "that connect AI solutions to real sustainability challenges. "
        "Write for a policy and technology audience. "
        "Always use proper markdown formatting."
    )
    prompt = f"""Topic: {query}

Background from research:
- Web Intelligence: {search_summary[:400]}
- Scientific Knowledge: {rag_summary[:400]}
- Cross Analysis: {analysis[:400]}

Write a complete SDG Intelligence Report with these EXACT sections:

## 🌍 Executive Summary

## 📊 Key Facts & Statistics

## 🎯 Relevant UN Sustainable Development Goals

## ⚠️ Current Challenges

## 🤖 How AI Can Help (IBM Granite / Technology Solutions)

## 💡 Actionable Recommendations

## 🔗 Related SDG Targets

Make it factual, structured, and actionable. Use bullet points within sections."""
    return llm_call(prompt, system, hf_token, gemini_key)
