"""
chatbot_multi.py
Interfaz de Streamlit para el sistema Multi-Agente RAG

CEIA - NLP2 - Trabajo Práctico 3
"""

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
import os
from pathlib import Path
from dotenv import load_dotenv

from multi_agent import MultiAgentRAG
from core.llm import LLMFactory

# Cargar variables de entorno
env_locations = [
    Path(__file__).parent / ".env",
    Path(__file__).parent.parent.parent / ".env",
    Path.cwd() / ".env",
]

for env_path in env_locations:
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
        break

# Configuración
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
RESUMES_FOLDER = "Trabajos/TP3/resumes"
INDEX_NAME = "resumes-bge-large-tp3"

# Configuración de página
st.set_page_config(
    page_title="Sistema Multi-Agente - CVs",
    page_icon="robot",
    layout="wide"
)

# Inicializar sistema multi-agente
if "rag" not in st.session_state:
    st.session_state.rag = MultiAgentRAG(
        pinecone_api_key=PINECONE_API_KEY,
        groq_api_key=GROQ_API_KEY,
        anthropic_api_key=ANTHROPIC_API_KEY,
        index_name=INDEX_NAME
    )

# Inicializar estado
if "messages" not in st.session_state:
    st.session_state.messages = []

if "indexed" not in st.session_state:
    st.session_state.indexed = False

# Header
st.title("Sistema Multi-Agente para Análisis de CVs")

st.markdown("""
Sistema avanzado con arquitectura multi-agente usando **LangGraph**:
- **Router inteligente**: Decide qué agentes deben responder (Regex + LLM híbrido)
- **Agentes especializados**: Cada candidato tiene su propio agente experto
- **Confidence scores**: Cada agente reporta su nivel de confianza
- **Agregación inteligente**: Sintetiza respuestas de múltiples agentes
""")

# Sidebar
with st.sidebar:
    st.header("Configuración")
    
    # Selector de modelo
    st.subheader("Modelo LLM")
    model_names = LLMFactory.get_model_names()
    
    selected_display_name = st.selectbox(
        "Selecciona el modelo:",
        options=list(model_names.keys()),
        index=0,
        help="Usado para router, agents y aggregator"
    )
    selected_model = model_names[selected_display_name]
    
    # Temperatura
    st.subheader("Creatividad")
    temperature = st.slider(
        "Temperature:",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.1,
        help="0.0 = preciso, 1.0 = creativo"
    )
    
    st.divider()
    
    # Gestión de CVs
    st.subheader("Gestión de CVs")
    
    if st.button("Re-indexar CVs", type="primary", use_container_width=True):
        try:
            progress = st.empty()
            progress.info("Paso 1/4: Cargando PDFs...")
            
            from core.utils import load_resume_pdfs
            docs = load_resume_pdfs(RESUMES_FOLDER)
            
            if not docs:
                st.error("No se encontraron PDFs en la carpeta de resumes")
            else:
                progress.success(f"Cargados {len(docs)} CVs")
                
                progress.info("Paso 2/4: Creando índice en Pinecone...")
                st.session_state.rag.docs = docs
                created = st.session_state.rag.vectorstore_manager.create_index(force_refresh=True)
                
                if created:
                    progress.success("Índice creado")
                
                progress.info("Paso 3/4: Generando embeddings e indexando...")
                st.session_state.rag.vectorstore_manager.add_documents(
                    docs, 
                    st.session_state.rag.embeddings
                )
                progress.success("Documentos indexados")
                
                progress.info("Paso 4/4: Construyendo grafo multi-agente...")
                num_docs, candidates = st.session_state.rag.load_and_index(
                    RESUMES_FOLDER,
                    force_refresh=False  # Ya se creó arriba
                )
                
                # Construir grafo
                st.session_state.rag.build_graph(
                    model_name=selected_model,
                    temperature=temperature
                )
                
                progress.empty()
                st.session_state.indexed = True
                
                st.success(f"Sistema listo: {num_docs} CVs, {len(candidates)} agentes")
                
                st.info("Agentes especializados creados:")
                for candidate in sorted(candidates):
                    location = next(
                        (d.metadata.get("location", "") for d in docs if d.metadata.get("candidate") == candidate),
                        ""
                    )
                    st.write(f"Agent_{candidate}" + (f" ({location})" if location else ""))
                
        except Exception as e:
            st.error(f"Error al indexar: {str(e)}")
            import traceback
            with st.expander("Ver detalles del error"):
                st.code(traceback.format_exc())
    
    # Información del sistema
    st.divider()
    st.caption(f"**Modelo:** {selected_display_name}")
    st.caption(f"**Temperature:** {temperature}")
    
    # Estado de API keys
    with st.expander("Estado de API Keys"):
        st.write(f"Groq: {'✓' if GROQ_API_KEY else '✗'}")
        st.write(f"Anthropic: {'✓' if ANTHROPIC_API_KEY else '✗'}")
        st.write(f"Pinecone: {'✓' if PINECONE_API_KEY else '✗'}")
    
    # Agentes disponibles
    if st.session_state.indexed:
        candidates = st.session_state.rag.candidates
        st.caption(f"**Agentes activos:** {len(candidates)}")
        
        with st.expander("Ver agentes"):
            for name in sorted(candidates):
                st.write(f"🤖 {name}")

# Chat interface
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        
        # Mostrar metadata si existe
        if "metadata" in m and m["metadata"]:
            with st.expander("Detalles de ejecución", expanded=False):
                meta = m["metadata"]
                
                # Router decision
                st.write(f"**Router:** {meta.get('router_reasoning', 'N/A')}")
                st.write(f"**Método:** {meta.get('router_method', 'N/A')}")
                st.write(f"**Agentes ejecutados:** {meta.get('agents_executed', 0)}")
                
                # Agent details
                if "agent_details" in meta:
                    st.write("**Confidence scores:**")
                    for agent, details in meta["agent_details"].items():
                        conf_pct = int(details["confidence"] * 100)
                        docs = details["retrieved_docs"]
                        
                        # Color según confidence
                        if conf_pct >= 70:
                            color = "🟢"
                        elif conf_pct >= 40:
                            color = "🟡"
                        else:
                            color = "🔴"
                        
                        st.write(f"{color} {agent.capitalize()}: {conf_pct}% ({docs} docs)")

# Input
if question := st.chat_input("Pregunta sobre los candidatos..."):
    
    if not st.session_state.indexed:
        st.error("Por favor, indexa los CVs primero usando el botón en la barra lateral.")
    else:
        # Agregar mensaje del usuario
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        
        # Procesar con multi-agente
        with st.chat_message("assistant"):
            with st.spinner("Ejecutando sistema multi-agente..."):
                try:
                    # Preparar historial
                    recent_messages = st.session_state.messages[-10:]
                    chat_history = []
                    
                    for msg in recent_messages:
                        if msg["role"] == "user":
                            chat_history.append(HumanMessage(content=msg["content"]))
                        elif msg["role"] == "assistant":
                            chat_history.append(AIMessage(content=msg["content"]))
                    
                    # Ejecutar grafo multi-agente
                    result = st.session_state.rag.query(
                        question=question,
                        chat_history=chat_history,
                        model_name=selected_model,
                        temperature=temperature
                    )
                    
                    response = result["response"]
                    metadata = result["metadata"]
                    
                    # Mostrar respuesta
                    st.markdown(response)
                    
                    # Mostrar metadata en expander
                    with st.expander("Detalles de ejecución", expanded=True):
                        st.write(f"**Router:** {metadata.get('router_reasoning', 'N/A')}")
                        st.write(f"**Método:** {metadata.get('router_method', 'N/A')}")
                        st.write(f"**Agentes ejecutados:** {metadata.get('agents_executed', 0)}")
                        
                        if "agent_details" in metadata:
                            st.write("**Confidence scores:**")
                            for agent, details in metadata["agent_details"].items():
                                conf_pct = int(details["confidence"] * 100)
                                docs = details["retrieved_docs"]
                                
                                if conf_pct >= 70:
                                    color = "🟢"
                                elif conf_pct >= 40:
                                    color = "🟡"
                                else:
                                    color = "🔴"
                                
                                st.write(f"{color} {agent.capitalize()}: {conf_pct}% ({docs} docs)")
                    
                    # Guardar en historial
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "metadata": metadata
                    })
                    
                except Exception as e:
                    st.error(f"Error al procesar la query: {str(e)}")
                    import traceback
                    with st.expander("Traceback"):
                        st.code(traceback.format_exc())

# Footer
st.divider()
st.caption("CEIA - NLP2 - TP3 | Multi-Agent System con LangGraph, Groq/Anthropic y Pinecone")