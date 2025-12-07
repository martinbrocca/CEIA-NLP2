"""
chatbot.py
Interfaz de Streamlit para el sistema RAG de análisis de currículums
Soporta Groq y Anthropic (Claude)

CEIA - NLP2 - Trabajo Práctico 2
"""

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
import os
from pathlib import Path
from dotenv import load_dotenv

from simple_rag import SimpleRAG
from core.llm import LLMFactory

# Cargar variables de entorno - buscar en múltiples ubicaciones
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
RESUMES_FOLDER = "Trabajos/TP2/resumes"
INDEX_NAME = "resumes-bge-large"

# Configuración de página
st.set_page_config(
    page_title="Asistente de CVs",
    page_icon="briefcase",
    layout="wide"
)

# Inicializar RAG system
if "rag" not in st.session_state:
    st.session_state.rag = SimpleRAG(
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
st.title("Asistente de Análisis de Currículums")

st.markdown("""
Sistema de preguntas y respuestas sobre CVs:
  - Presiona "Re-indexar CVs" para cargar e indexar los currículums desde la carpeta `resumes/`     
  - Haz preguntas sobre los candidatos usando el chat abajo

Ejemplos de preguntas:
  - "¿Quién tiene experiencia en Machine Learning?"
  - "¿Qué candidato tiene conocimientos en Python y SQL?"
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
        help="Modelos Groq o Anthropic (Claude)"
    )
    selected_model = model_names[selected_display_name]
    
    # Temperatura
    st.subheader("Creatividad")
    temperature = st.slider(
        "Temperature (0 = preciso, 1 = creativo):",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.1,
        help="Temperatura más alta = respuestas más creativas"
    )
    
    # Instrucciones personalizadas
    st.subheader("Instrucciones personalizadas")
    custom_instructions = st.text_area(
        "Agrega instrucciones adicionales:",
        value="",
        height=100,
        placeholder="Ejemplo: Responde siempre en bullet points.",
        help="Estas instrucciones se añadirán al prompt del sistema"
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
                    progress.success("Índice creado y listo")
                else:
                    progress.info("Índice ya existía")
                
                progress.info("Paso 3/4: Generando embeddings e indexando...")
                st.session_state.rag.vectorstore_manager.add_documents(
                    docs, 
                    st.session_state.rag.embeddings
                )
                progress.success("Documentos indexados")
                
                progress.info("Paso 4/4: Configurando retriever...")
                st.session_state.rag._retriever = st.session_state.rag.vectorstore_manager.get_retriever(
                    st.session_state.rag.embeddings,
                    k=10
                )
                
                progress.empty()
                st.session_state.indexed = True
                
                candidates = list(set(
                    doc.metadata.get("candidate", "Unknown")
                    for doc in docs
                ))
                
                st.success(f"Indexación completada: {len(docs)} CVs, {len(candidates)} candidatos")
                
                st.info("Candidatos cargados:")
                for candidate in sorted(candidates):
                    location = next(
                        (d.metadata.get("location", "") for d in docs if d.metadata.get("candidate") == candidate),
                        ""
                    )
                    st.write(f"- {candidate}" + (f" ({location})" if location else ""))
                
        except Exception as e:
            st.error(f"Error al indexar: {str(e)}")
            import traceback
            with st.expander("Ver detalles del error"):
                st.code(traceback.format_exc())
    
    # Información del sistema
    st.divider()
    st.caption(f"**Modelo:** {selected_display_name}")
    st.caption(f"**Temperature:** {temperature}")
    
    # Debug: Estado de API keys
    with st.expander("Estado de API Keys"):
        st.write(f"Groq: {'✓' if GROQ_API_KEY else '✗ NO configurada'}")
        st.write(f"Anthropic: {'✓' if ANTHROPIC_API_KEY else '✗ NO configurada'}")
        st.write(f"Pinecone: {'✓' if PINECONE_API_KEY else '✗ NO configurada'}")
        if ANTHROPIC_API_KEY:
            st.caption(f"Claude key: {ANTHROPIC_API_KEY[:20]}...")
    
    # Candidatos cargados
    if st.session_state.indexed:
        candidates = st.session_state.rag.candidates
        st.caption(f"**Candidatos:** {len(candidates)}")
        
        with st.expander("Ver candidatos"):
            for name in sorted(candidates):
                st.write(f"- {name}")

# Chat interface
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if question := st.chat_input("Pregunta sobre los candidatos..."):
    
    if not st.session_state.indexed:
        st.error("Por favor, indexa los CVs primero usando el botón en la barra lateral.")
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        
        with st.chat_message("assistant"):
            with st.spinner("Pensando..."):
                try:
                    recent_messages = st.session_state.messages[-10:]
                    chat_history = []
                    
                    for msg in recent_messages:
                        if msg["role"] == "user":
                            chat_history.append(HumanMessage(content=msg["content"]))
                        elif msg["role"] == "assistant":
                            chat_history.append(AIMessage(content=msg["content"]))
                    
                    response = st.session_state.rag.query(
                        question=question,
                        chat_history=chat_history,
                        model_name=selected_model,
                        temperature=temperature,
                        custom_instructions=custom_instructions
                    )
                    
                    st.markdown(response)
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response
                    })
                    
                except Exception as e:
                    st.error(f"Error al procesar la query: {str(e)}")

# Footer
st.divider()
st.caption("CEIA - NLP2 - Trabajo Práctico 2 | Powered by LangChain, Groq, Anthropic y Pinecone")