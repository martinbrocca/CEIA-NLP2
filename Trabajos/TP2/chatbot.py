# chatbot.py – A Streamlit app for resume-based Q&A using LangChain, Groq, and Pinecone
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_core.messages import HumanMessage, AIMessage
from operator import itemgetter



import os
from pathlib import Path
import PyPDF2
from dotenv import load_dotenv

load_dotenv()

# ========================= CONFIG =========================
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
INDEX_NAME = "resumes-bge-large"
RESUMES_FOLDER = "Trabajos/TP2/resumes"

# ========================= EMBEDDINGS (FULLY FIXED: RunnableLambda sanitizer) =========================
@st.cache_resource
def get_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings

    class CleanEmbeddings:
        def __init__(self):
            self.emb = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")

        def _sanitize(self, text):
            if isinstance(text, dict):
                text = str(text)
            if not isinstance(text, str):
                text = str(text)
            return text.replace("\n", " ")

        def embed_query(self, text):
            return self.emb.embed_query(self._sanitize(text))

        def embed_documents(self, texts):
            cleaned = [self._sanitize(t) for t in texts]
            return self.emb.embed_documents(cleaned)

    return CleanEmbeddings()

# ========================= RESUME LOADER =========================
def extract_candidate_name(filename: str) -> str:
    name = Path(filename).stem
    for noise in [" - Resume", " Resume", " - CV", " CV", " 2024", " 2025", "-2024", "(Final)", " - final"]:
        name = name.replace(noise, "")
    if " - " in name:
        name = name.split(" - ")[0]
    name = name.replace("_", " ").replace("-", " ")
    parts = [p for p in name.split() if p and p[0].isupper()]
    return " ".join(parts[:3]) if len(parts) >= 2 else name.strip().title()

def load_resumes() -> list[Document]:
    docs = []
    folder = Path(RESUMES_FOLDER)
    folder.mkdir(exist_ok=True)
    if not any(folder.glob("*.pdf")):
        st.warning("No PDFs in ./resumes folder")
        return docs

    for pdf_path in sorted(folder.glob("*.pdf")):
        try:
            with open(pdf_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                text = "\n".join(page.extract_text() or "" for page in reader.pages)
            candidate = extract_candidate_name(pdf_path.name)
            docs.append(Document(page_content=text, metadata={"candidate": candidate, "source": pdf_path.name}))
            st.success(f"Loaded: **{candidate}**")
        except Exception as e:
            st.error(f"Error: {e}")
    return docs

# ========================= CHUNKING & VECTORSTORE =========================
def get_splitter():
    return RecursiveCharacterTextSplitter(chunk_size=1600, chunk_overlap=400, keep_separator=True)

@st.cache_resource
def get_vectorstore(_docs: list[Document]):
    if not _docs:
        return None
    chunks = get_splitter().split_documents(_docs)
    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = i
        c.metadata["candidate"] = c.metadata.get("candidate", "Unknown")

    pc = Pinecone(api_key=PINECONE_API_KEY)
    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(name=INDEX_NAME, dimension=1024, metric="cosine",
                        spec=ServerlessSpec(cloud="aws", region="us-east-1"))

    return PineconeVectorStore.from_documents(chunks, get_embeddings(), index_name=INDEX_NAME)

@st.cache_resource
def get_retriever():
    vs = get_vectorstore(st.session_state.raw_docs)
    return vs.as_retriever(search_kwargs={"k": 10}) if vs else None

# ========================= LLM & CHAIN =========================
@st.cache_resource
def get_llm():
    return ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0, groq_api_key=GROQ_API_KEY)

system_prompt = """You are a brutally accurate HR screening bot. Your job is to compare all resumes uploaded to the system.

STRICT RULES - FOLLOW OR BE FIRED:
- Base every single statement ONLY on the retrieved context
- NEVER guess, assume, or "think" anything not explicitly written
- ALWAYS start your answer with the candidate name(s) involved
- If only one has it → say clearly which one
- If neither has it → say "Neither candidate"
- If you're unsure → say "Not clearly mentioned"
- NEVER say "however", "actually", "but", or flip-flop mid-sentence
- Answer in one short, direct paragraph maximum

Context:
{context}"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}")
])

def convert_history(messages):
    out = []
    for role, content in messages:
        if role == "user":
            out.append(HumanMessage(content=content))
        elif role == "assistant":
            out.append(AIMessage(content=content))
    return out

def format_docs(docs):
    if not docs:
        return "No relevant information."
    seen = set()
    parts = []
    for d in docs:
        cand = d.metadata.get("candidate", "Unknown")
        src = d.metadata.get("source", "")
        if (cand, src) not in seen:
            seen.add((cand, src))
            parts.append(f"\n**{cand}** – {src}\n")
        parts.append(d.page_content.strip())
    return "\n".join(parts)

# ========================= STREAMLIT =========================
st.title("Resume QA Bot – HR Assistant")

if st.button("Re-index resumes", type="primary"):
    with st.spinner("Indexing..."):
        st.session_state.raw_docs = load_resumes()
        if st.session_state.raw_docs:
            get_vectorstore.clear()
            st.success("Done!")

if "raw_docs" not in st.session_state:
    st.session_state.raw_docs = []
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if question := st.chat_input("Ask about the candidates..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    retriever = get_retriever()
    if not retriever:
        st.error("Index first!")
    else:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                chain = (
                {
                    "context": itemgetter("question") | retriever | format_docs,
                    "question": itemgetter("question"),
                    "chat_history": itemgetter("chat_history"),
                }
                | prompt
                | get_llm()
                | StrOutputParser()
            )
                raw_history = st.session_state.messages[-10:]

                history = []
                for m in raw_history:
                    if m["role"] == "user":
                        history.append(HumanMessage(content=m["content"]))
                    elif m["role"] == "assistant":
                        history.append(AIMessage(content=m["content"]))
                response = chain.invoke({"question": question, "chat_history": history})
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})