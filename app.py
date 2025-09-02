import json, os
from pathlib import Path

import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Vedic Ethics Robot", page_icon="🤖", layout="centered")
st.title("🤖 Vedic Ethics Robot (RAG MVP)")
st.caption("Retrieves curated passages, then reasons with a transparent template. Zero-cost, no external APIs.")

# ------------------------
# Load corpus
# ------------------------
DATA_PATH = Path("data/corpus.jsonl")
if not DATA_PATH.exists():
    st.error("Missing data/corpus.jsonl. Create it with a few passages first.")
    st.stop()

docs = []
with open(DATA_PATH, "r", encoding="utf-8") as f:
    for line in f:
        try:
            docs.append(json.loads(line))
        except json.JSONDecodeError:
            pass

texts = [d["passage"] for d in docs]
meta  = [(d.get("id",""), d.get("source",""), d.get("theme",[])) for d in docs]

# Build TF-IDF index
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(texts)

# ------------------------
# Simple guardrails
# ------------------------
RISKY_KEYWORDS = [
    "medical", "diagnose", "law", "illegal", "violence", "self-harm", "weapon",
    "suicide", "harm yourself", "attack", "revenge", "hack", "exploit"
]
def is_risky(q: str) -> bool:
    ql = q.lower()
    return any(k in ql for k in RISKY_KEYWORDS)

# ------------------------
# Retrieval
# ------------------------
def retrieve(query, k=3):
    qv = vectorizer.transform([query])
    sims = cosine_similarity(qv, X).ravel()
    idx = sims.argsort()[::-1][:k]
    results = []
    for i in idx:
        score = float(sims[i])
        results.append({
            "score": score,
            "id": meta[i][0],
            "source": meta[i][1],
            "themes": meta[i][2],
            "passage": texts[i]
        })
    return results

# ------------------------
# Reasoning template (rule-based for MVP)
# ------------------------
def reason(query, passages):
    # Very lightweight “structure” without an external LLM
    principles = []
    for p in passages:
        label = ", ".join(p["themes"]) if p["themes"] else "principle"
        principles.append(f"- From {p['source']}: _{label}_ — “{p['passage']}”")

    # naive option generation for demo
    options = [
        "Act cautiously with minimal reversible steps.",
        "Seek more information or a second perspective.",
        "Defer to a qualified human if stakes are high."
    ]
    tradeoffs = [
        "- Caution reduces harm but may delay benefits.",
        "- Gathering info takes time but improves accuracy.",
        "- Deferring improves safety but reduces autonomy."
    ]

    # simple “confidence” proxy = average similarity
    conf = np.mean([p["score"] for p in passages]) if passages else 0.0
    conf_txt = f"{conf:.2f}"

    # recommendation logic
    if conf < 0.05:
        rec = "Uncertain. Acquire more context, then reconsider."
    else:
        rec = "Prioritize non-harm and truthfulness; take a reversible step and review impact. If people’s safety/rights are involved, escalate to a human."

    citations = [f"{p['id']} — {p['source']}" for p in passages]
    return {
        "context": query.strip(),
        "principles": principles,
        "options": options,
        "tradeoffs": tradeoffs,
        "recommendation": rec,
        "confidence": conf_txt,
        "citations": citations
    }

# ------------------------
# UI
# ------------------------
q = st.text_area("Ask an ethical question or describe a situation:", height=120, placeholder="e.g., A teammate lied to a client. What is the right course of action?")
col1, col2 = st.columns([1,1])
with col1:
    think = st.button("Think")
with col2:
    st.caption("This MVP retrieves up to 3 passages and formats a transparent answer. No medical/legal instructions.")

if think:
    if not q.strip():
        st.warning("Type a question first.")
    elif is_risky(q):
        st.error("This appears high-risk (medical/legal/self-harm/violence). I cannot advise specific actions. Please consult qualified help or escalate to a responsible human.")
    else:
        topk = retrieve(q, k=3)
        result = reason(q, topk)

        st.subheader("Recommendation")
        st.write(result["recommendation"])
        st.caption(f"Confidence (rough): {result['confidence']}")

        st.subheader("Principles (retrieved)")
        for pr in result["principles"]:
            st.markdown(pr)

        st.subheader("Options")
        for op in result["options"]:
            st.markdown(f"- {op}")

        st.subheader("Trade-offs")
        for t in result["tradeoffs"]:
            st.markdown(t)

        st.subheader("Citations")
        for c in result["citations"]:
            st.markdown(f"- {c}")

st.divider()
st.caption("Disclaimer: Educational prototype. Not a substitute for professional advice. Passages are simplified—add translators/commentaries and broaden sources over time.")
