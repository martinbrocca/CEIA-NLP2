import os
import requests
import streamlit as st
import pinecone
from dotenv import load_dotenv

"""
Streamlit chatbot using Pinecone as RAG DB and GROQ for embeddings / completions.

Environment variables required (in .env or environment):
- PINECONE_API_KEY
- PINECONE_ENV (e.g. "us-west1-gcp")
- PINECONE_INDEX (name of the existing Pinecone index)
- GROQ_API_KEY
- GROQ_API_URL (optional; default: "https://api.groq.ai")

Install dependencies:
pip install streamlit pinecone-client requests python-dotenv

Save as: chatbot.py
Run:
streamlit run chatbot.py
"""

load_dotenv()

# Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = os.getenv("GROQ_API_URL", "https://api.groq.ai")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "my-index")

if not GROQ_API_KEY:
    st.error("Missing GROQ_API_KEY environment variable.")
    st.stop()
if not PINECONE_API_KEY or not PINECONE_ENV:
    st.error("Missing PINECONE_API_KEY or PINECONE_ENV environment variables.")
    st.stop()

HEADERS = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json",
}

@st.cache_resource
def init_pinecone(api_key: str, environment: str):
    pinecone.init(api_key=api_key, environment=environment)
    return pinecone

pinecone = init_pinecone(PINECONE_API_KEY, PINECONE_ENV)

@st.cache_resource
def get_index(index_name: str):
    try:
        return pinecone.Index(index_name)
    except Exception as e:
        st.error(f"Could not open Pinecone index '{index_name}': {e}")
        st.stop()

index = get_index(PINECONE_INDEX)

def embed_text(text: str, model: str = "embed-1"):
    """
    Request an embedding from GROQ. Adjust endpoint/model names if your GROQ plan uses different names.
    """
    url = f"{GROQ_API_URL.rstrip('/')}/v1/embeddings"
    payload = {"input": text, "model": model}
    resp = requests.post(url, headers=HEADERS, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    # Try common shapes: data["data"][0]["embedding"] or data["embedding"]
    if isinstance(data, dict):
        if "data" in data and isinstance(data["data"], list) and len(data["data"]) > 0:
            return data["data"][0].get("embedding")
        if "embedding" in data:
            return data["embedding"]
    raise ValueError("Unexpected embedding response shape: " + str(data))

def query_pinecone(embedding, top_k: int = 4):
    # Query the pinecone index and return matched metadata texts
    try:
        res = index.query(vector=embedding, top_k=top_k, include_metadata=True)
    except TypeError:
        # Fallback for client variations
        res = index.query(queries=[embedding], top_k=top_k, include_metadata=True)
        if isinstance(res, dict) and "results" in res:
            res = res["results"][0]
    matches = res.get("matches") if isinstance(res, dict) else getattr(res, "matches", [])
    results = []
    for m in matches:
        md = m.get("metadata") if isinstance(m, dict) else getattr(m, "metadata", {})
        text = md.get("text") or md.get("content") or md.get("source") or ""
        results.append({"id": m.get("id") if isinstance(m, dict) else getattr(m, "id", None),
                        "score": m.get("score") if isinstance(m, dict) else getattr(m, "score", None),
                        "text": text})
    return results

def build_prompt(query: str, contexts: list):
    context_text = "\n\n---\n\n".join([c["text"] for c in contexts if c["text"]])
    prompt = (
        "You are a helpful assistant. Use the provided context to answer the user's question. "
        "If the answer is not contained in the context, say you don't know and do not hallucinate.\n\n"
        f"CONTEXT:\n{context_text}\n\n"
        f"USER QUESTION:\n{query}\n\n"
        "Answer concisely and cite the context where appropriate."
    )
    return prompt

def call_chat_completion(prompt: str, model: str = "groq-1", max_tokens: int = 512, temperature: float = 0.0):
    """
    Call GROQ chat/completion endpoint. Response parsing includes common variants:
    - {"choices":[{"message": {"content": "..."}}]}
    - {"choices":[{"text":"..."}]}
    """
    url = f"{GROQ_API_URL.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    resp = requests.post(url, headers=HEADERS, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    # Common shapes
    if isinstance(data, dict) and "choices" in data and isinstance(data["choices"], list) and len(data["choices"]) > 0:
        choice = data["choices"][0]
        if isinstance(choice, dict):
            # OpenAI-like chat shape
            if "message" in choice and isinstance(choice["message"], dict) and "content" in choice["message"]:
                return choice["message"]["content"]
            # Completion-like shape
            if "text" in choice:
                return choice["text"]
    # Fallback: if "output" or "content" keys exist somewhere
    if isinstance(data, dict):
        if "output" in data and isinstance(data["output"], str):
            return data["output"]
    raise ValueError("Unexpected completion response shape: " + str(data))

# Streamlit UI
st.title("RAG Chatbot (Pinecone + GROQ)")

if "messages" not in st.session_state:
    st.session_state["messages"] = []  # list of {"role": "user"/"assistant", "text": ...}

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Context top_k", min_value=1, max_value=10, value=4)
    model = st.selectbox("LLM model", options=["groq-1", "gpt-3.5-turbo", "gpt-4"], index=0)

# Chat display
for m in st.session_state.messages:
    if m["role"] == "user":
        st.markdown(f"**You:** {m['text']}")
    else:
        st.markdown(f"**Assistant:** {m['text']}")

# Input
query = st.text_input("Ask a question based on the RAG knowledge base", key="input")
if st.button("Send") and query.strip():
    st.session_state.messages.append({"role": "user", "text": query})

    with st.spinner("Embedding query and retrieving context..."):
        emb = embed_text(query)
        contexts = query_pinecone(emb, top_k=top_k)

    if not contexts:
        answer = "No context retrieved from the RAG DB. The index may be empty or the query didn't match documents."
    else:
        prompt = build_prompt(query, contexts)
        with st.spinner("Generating answer..."):
            answer = call_chat_completion(prompt, model=model)

    # Optionally attach source list
    sources = "\n".join([f"- id: {c['id']} (score: {c['score']})" for c in contexts])
    answer_with_sources = f"{answer}\n\nSources:\n{sources}" if contexts else answer

    st.session_state.messages.append({"role": "assistant", "text": answer_with_sources})
    # clear input
    st.session_state.input = ""

# Footer / index info
st.markdown("---")
st.caption(f"Pinecone index: {PINECONE_INDEX}")