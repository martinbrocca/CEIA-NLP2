# chatbot_multiagent.py – Multi-agent system for resume Q&A
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_core.messages import HumanMessage, AIMessage
from operator import itemgetter
from typing import TypedDict, Annotated, Literal
import operator
import re

# LangGraph imports
from langgraph.graph import StateGraph, END

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

# ========================= STATE DEFINITION =========================
class AgentState(TypedDict):
    """Estado del grafo de agentes"""
    question: str
    chat_history: list
    context: str
    selected_agent: str
    candidate_names: list[str]
    answer: str

# ========================= EMBEDDINGS =========================
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

def get_candidate_retriever(candidate_name: str):
    """Retriever filtrado por candidato específico"""
    vs = get_vectorstore(st.session_state.raw_docs)
    if not vs:
        return None
    return vs.as_retriever(
        search_kwargs={
            "k": 5,
            "filter": {"candidate": candidate_name}
        }
    )

# ========================= LLM =========================
@st.cache_resource
def get_llm():
    return ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0, groq_api_key=GROQ_API_KEY)

# ========================= ROUTER NODE =========================
def router_node(state: AgentState) -> AgentState:
    """
    Nodo router: analiza la pregunta y decide qué agente usar.
    Usa regex para detectar nombres de candidatos.
    """
    question = state["question"].lower()
    candidate_names = state.get("candidate_names", [])
    
    # Buscar menciones de candidatos específicos usando regex
    for candidate in candidate_names:
        # Crear pattern flexible: busca nombre completo o partes del nombre
        name_parts = candidate.lower().split()
        
        # Buscar nombre completo
        if re.search(rf'\b{re.escape(candidate.lower())}\b', question):
            state["selected_agent"] = candidate
            return state
        
        # Buscar por apellido (última palabra del nombre)
        if len(name_parts) > 1:
            last_name = name_parts[-1]
            if re.search(rf'\b{re.escape(last_name)}\b', question):
                state["selected_agent"] = candidate
                return state
    
    # Detectar preguntas comparativas
    comparison_patterns = [
        r'\b(compar|versus|vs|diferencia|mejor|peor)\b',
        r'\b(ambos|todos|entre)\b',
        r'\b(quién|quien|cual|cuál)\b.*\b(mejor|mas|más)\b'
    ]
    
    for pattern in comparison_patterns:
        if re.search(pattern, question):
            state["selected_agent"] = "general"
            return state
    
    # Por defecto: agente general
    state["selected_agent"] = "general"
    return state

# ========================= CONDITIONAL EDGE =========================
def route_question(state: AgentState) -> str:
    """
    Conditional edge: decide a qué nodo ir basado en el agente seleccionado.
    """
    selected = state.get("selected_agent", "general")
    
    # Si es un candidato específico, ir a su agente
    if selected in state.get("candidate_names", []):
        return selected
    
    # Caso general (comparaciones o preguntas generales)
    return "general"

# ========================= AGENT NODES =========================
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

def create_candidate_agent(candidate_name: str):
    """
    Crea un nodo agente específico para un candidato.
    """
    def candidate_agent_node(state: AgentState) -> AgentState:
        # Obtener retriever filtrado por este candidato
        retriever = get_candidate_retriever(candidate_name)
        
        if not retriever:
            state["answer"] = f"No hay información disponible para {candidate_name}."
            return state
        
        # Recuperar documentos
        docs = retriever.invoke(state["question"])
        context = format_docs(docs)
        state["context"] = context
        
        # Prompt específico para este candidato
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are an expert HR assistant specialized in analyzing {candidate_name}'s resume.

STRICT RULES:
- Answer ONLY based on {candidate_name}'s resume information provided in the context
- Be specific and detailed about {candidate_name}'s qualifications
- If the information is not in the context, say "Not mentioned in {candidate_name}'s resume"
- Keep answers concise but informative
- Always refer to the candidate by name: {candidate_name}

Context about {candidate_name}:
{{context}}"""),
            MessagesPlaceholder("chat_history"),
            ("human", "{question}")
        ])
        
        # Crear chain
        chain = (
            prompt
            | get_llm()
            | StrOutputParser()
        )
        
        # Generar respuesta
        response = chain.invoke({
            "question": state["question"],
            "context": context,
            "chat_history": state.get("chat_history", [])
        })
        
        state["answer"] = response
        return state
    
    return candidate_agent_node

def general_agent_node(state: AgentState) -> AgentState:
    """
    Agente general: maneja comparaciones y preguntas sobre todos los candidatos.
    """
    retriever = get_retriever()
    
    if not retriever:
        state["answer"] = "No hay información disponible."
        return state
    
    # Recuperar documentos de todos los candidatos
    docs = retriever.invoke(state["question"])
    context = format_docs(docs)
    state["context"] = context
    
    # Prompt para comparaciones
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a brutally accurate HR screening bot comparing multiple resumes.

STRICT RULES:
- Base every statement ONLY on the retrieved context
- NEVER guess or assume anything not explicitly written
- For comparisons: clearly state which candidate has what
- If comparing: mention ALL relevant candidates
- If only one has it → say which one clearly
- If neither has it → say "Neither candidate"
- If unsure → say "Not clearly mentioned"
- Answer in one concise paragraph

Context:
{context}"""),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}")
    ])
    
    # Crear chain
    chain = (
        prompt
        | get_llm()
        | StrOutputParser()
    )
    
    # Generar respuesta
    response = chain.invoke({
        "question": state["question"],
        "context": context,
        "chat_history": state.get("chat_history", [])
    })
    
    state["answer"] = response
    return state

# ========================= GRAPH BUILDER =========================
def build_agent_graph(candidate_names: list[str]):
    """
    Construye el grafo de agentes con un agente por candidato + agente general.
    """
    workflow = StateGraph(AgentState)
    
    # Agregar nodo router
    workflow.add_node("router", router_node)
    
    # Agregar nodo para cada candidato
    for candidate in candidate_names:
        workflow.add_node(candidate, create_candidate_agent(candidate))
    
    # Agregar nodo general
    workflow.add_node("general", general_agent_node)
    
    # Definir punto de entrada
    workflow.set_entry_point("router")
    
    # Agregar conditional edges desde router
    workflow.add_conditional_edges(
        "router",
        route_question,
        {
            **{candidate: candidate for candidate in candidate_names},
            "general": "general"
        }
    )
    
    # Todos los agentes van a END
    for candidate in candidate_names:
        workflow.add_edge(candidate, END)
    workflow.add_edge("general", END)
    
    return workflow.compile()

# ========================= STREAMLIT APP =========================
st.title(" Multi-Agent Resume QA System")
st.markdown("*Sistema de agentes inteligente para analizar resumes*")

# Sidebar con información
with st.sidebar:
    st.header("📋 Sistema de Agentes")
    st.markdown("""
    ### Cómo funciona:
    1. **Router**: Analiza tu pregunta
    2. **Agentes específicos**: Un agente por candidato
    3. **Agente general**: Para comparaciones
    
    ### Ejemplos de preguntas:
    - "¿Qué experiencia tiene Juan?"
    - "Compara las habilidades de todos"
    - "¿Quién tiene más experiencia en Python?"
    """)
    
    if st.button(" Re-indexar Resumes", type="primary"):
        with st.spinner("Indexando resumes..."):
            st.session_state.raw_docs = load_resumes()
            if st.session_state.raw_docs:
                get_vectorstore.clear()
                
                # Extraer nombres de candidatos
                candidate_names = list(set(
                    doc.metadata.get("candidate", "Unknown") 
                    for doc in st.session_state.raw_docs
                ))
                st.session_state.candidate_names = [c for c in candidate_names if c != "Unknown"]
                
                st.success(f" {len(st.session_state.raw_docs)} resumes indexados!")
                st.info(f"**Candidatos detectados:**\n" + "\n".join(f"- {c}" for c in st.session_state.candidate_names))

# Inicializar estado
if "raw_docs" not in st.session_state:
    st.session_state.raw_docs = []
if "messages" not in st.session_state:
    st.session_state.messages = []
if "candidate_names" not in st.session_state:
    st.session_state.candidate_names = []

# Mostrar historial de chat
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        # Mostrar qué agente respondió
        if m["role"] == "assistant" and "agent_used" in m:
            st.caption(f"🤖 Agente: **{m['agent_used']}**")

# Input del usuario
if question := st.chat_input("Pregunta sobre los candidatos..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Verificar que haya resumes indexados
    if not st.session_state.raw_docs:
        with st.chat_message("assistant"):
            st.error(" Por favor, indexa los resumes primero usando el botón en la barra lateral.")
    else:
        with st.chat_message("assistant"):
            with st.spinner(" Analizando con sistema de agentes..."):
                # Construir grafo de agentes
                graph = build_agent_graph(st.session_state.candidate_names)
                
                # Preparar chat history
                history = []
                for m in st.session_state.messages[-10:]:
                    if m["role"] == "user":
                        history.append(HumanMessage(content=m["content"]))
                    elif m["role"] == "assistant":
                        history.append(AIMessage(content=m["content"]))
                
                # Ejecutar grafo
                result = graph.invoke({
                    "question": question,
                    "chat_history": history,
                    "candidate_names": st.session_state.candidate_names,
                    "context": "",
                    "selected_agent": "",
                    "answer": ""
                })
                
                # Mostrar respuesta
                response = result["answer"]
                agent_used = result["selected_agent"]
                
                st.markdown(response)
                st.caption(f"🤖 Agente utilizado: **{agent_used}**")
                
                # Guardar en historial
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response,
                    "agent_used": agent_used
                })

# Mostrar diagrama del grafo (opcional)
with st.expander("📊 Ver Diagrama del Sistema de Agentes"):
    if st.session_state.candidate_names:
        st.markdown("### Flujo del Sistema:")
        st.code(f"""
START
  ↓
[Router] ← Analiza la pregunta
  ↓
  ├─→ [{st.session_state.candidate_names[0] if st.session_state.candidate_names else 'Candidato 1'}] (Agente específico)
  {'├─→ [' + st.session_state.candidate_names[1] + '] (Agente específico)' if len(st.session_state.candidate_names) > 1 else ''}
  {'├─→ [' + st.session_state.candidate_names[2] + '] (Agente específico)' if len(st.session_state.candidate_names) > 2 else ''}
  └─→ [General] (Comparaciones y preguntas generales)
  ↓
[END] → Respuesta al usuario
        """, language="text")
    else:
        st.info("Indexa los resumes para ver el diagrama del sistema.")