"""
simple_rag.py
Clase de RAG simple para TP2
Soporta Groq y Anthropic (Claude)
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter

from core import (
    CleanEmbeddings,
    PineconeManager,
    LLMFactory,
    load_resume_pdfs,
    format_docs_simple
)


class SimpleRAG:
    """Sistema de RAG simple para análisis de currículums"""
    
    def __init__(self, pinecone_api_key, groq_api_key=None, anthropic_api_key=None, index_name="resumes-bge-large"):
        """
        Inicializa el sistema RAG
        
        Args:
            pinecone_api_key: API key de Pinecone
            groq_api_key: API key de Groq (opcional)
            anthropic_api_key: API key de Anthropic (opcional)
            index_name: Nombre del índice de Pinecone
        """
        self.embeddings = CleanEmbeddings()
        self.vectorstore_manager = PineconeManager(
            api_key=pinecone_api_key,
            index_name=index_name
        )
        self.groq_api_key = groq_api_key
        self.anthropic_api_key = anthropic_api_key
        self.docs = []
        self._retriever = None
    
    def load_and_index(self, folder_path, force_refresh=False):
        """Carga PDFs e indexa en Pinecone"""
        self.docs = load_resume_pdfs(folder_path)
        
        if not self.docs:
            return 0, []
        
        self.vectorstore_manager.create_index(force_refresh=force_refresh)
        self.vectorstore_manager.add_documents(self.docs, self.embeddings)
        
        self._retriever = self.vectorstore_manager.get_retriever(
            self.embeddings,
            k=10
        )
        
        candidate_names = list(set(
            doc.metadata.get("candidate", "Unknown")
            for doc in self.docs
        ))
        
        return len(self.docs), candidate_names
    
    def _build_prompt(self, custom_instructions=""):
        """Construye el prompt del sistema"""
        base_prompt = """Eres un asistente de screening de RR.HH. que analiza currículums.

REGLAS:
- Basa tus respuestas en la información del contexto proporcionado
- Haz inferencias razonables de los datos (ej: si la dirección dice "TX, United States", la persona vive en EE.UU.)
- SIEMPRE comienza tu respuesta mencionando el/los nombre(s) del/los candidato(s) involucrados
- Si solo un candidato tiene la información, indica claramente cuál
- Si ningún candidato la tiene, di "Ninguno de los candidatos"
- Si no está claro, di "No se menciona claramente"
- Mantén las respuestas concisas y directas (máximo 1-2 párrafos)"""
        
        if custom_instructions.strip():
            base_prompt += f"\n\nINSTRUCCIONES ADICIONALES:\n{custom_instructions}"
        
        base_prompt += "\n\nContexto:\n{context}"
        
        return base_prompt
    
    def query(
        self,
        question,
        chat_history=None,
        model_name="llama-3.3-70b-versatile",
        temperature=0.0,
        custom_instructions=""
    ):
        """Procesa una query usando RAG"""
        if not self._retriever:
            raise ValueError("No se han cargado documentos. Llama a load_and_index() primero.")
        
        if chat_history is None:
            chat_history = []
        
        system_prompt = self._build_prompt(custom_instructions)
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{question}")
        ])
        
        llm = LLMFactory.create_llm(
            model_name=model_name,
            temperature=temperature,
            groq_api_key=self.groq_api_key,
            anthropic_api_key=self.anthropic_api_key
        )
        
        chain = (
            {
                "context": itemgetter("question") | self._retriever | format_docs_simple,
                "question": itemgetter("question"),
                "chat_history": itemgetter("chat_history"),
            }
            | prompt
            | llm
            | StrOutputParser()
        )
        
        response = chain.invoke({
            "question": question,
            "chat_history": chat_history
        })
        
        return response
    
    @property
    def candidates(self):
        """Retorna lista de candidatos cargados"""
        return list(set(
            doc.metadata.get("candidate", "Unknown")
            for doc in self.docs
        ))
    
    @property
    def num_documents(self):
        """Retorna número de documentos cargados"""
        return len(self.docs)