"""
multi_agent.py
Sistema Multi-Agente RAG usando LangGraph para TP3

Arquitectura:
Usuario → Router → [Agent1, Agent2, ...] → Aggregator → Respuesta
"""

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage

from state import AgentState
from graph_nodes import RouterNode, SpecialistAgentNode, AggregatorNode
from core import CleanEmbeddings, PineconeManager, LLMFactory, load_resume_pdfs


class MultiAgentRAG:
    """
    Sistema RAG multi-agente con routing inteligente y agregación
    """
    
    def __init__(self, pinecone_api_key, groq_api_key=None, anthropic_api_key=None, 
                 index_name="resumes-bge-large"):
        """
        Inicializa el sistema multi-agente
        
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
        self.candidates = []
        self.graph = None
        self._specialist_agents = {}
    
    def load_and_index(self, folder_path, force_refresh=False):
        """
        Carga PDFs e indexa en Pinecone
        
        Args:
            folder_path: Ruta a carpeta con PDFs
            force_refresh: Si True, elimina índice existente
            
        Returns:
            tuple: (num_docs, list_of_candidates)
        """
        # Cargar documentos
        self.docs = load_resume_pdfs(folder_path)
        
        if not self.docs:
            return 0, []
        
        # Crear/recrear índice
        self.vectorstore_manager.create_index(force_refresh=force_refresh)
        
        # Indexar documentos
        self.vectorstore_manager.add_documents(self.docs, self.embeddings)
        
        # Extraer candidatos únicos
        self.candidates = list(set(
            doc.metadata.get("candidate", "Unknown")
            for doc in self.docs
        ))
        
        return len(self.docs), self.candidates
    
    def build_graph(self, model_name="llama-3.3-70b-versatile", temperature=0.0):
        """
        Construye el grafo de LangGraph con todos los nodos
        
        Args:
            model_name: ID del modelo LLM
            temperature: Temperatura del LLM
        """
        # Crear LLM para router y aggregator
        llm = LLMFactory.create_llm(
            model_name=model_name,
            temperature=temperature,
            groq_api_key=self.groq_api_key,
            anthropic_api_key=self.anthropic_api_key
        )
        
        # Crear LLM para agentes (puede ser diferente, ej: más rápido)
        agent_llm = llm  # Por ahora, mismo LLM
        
        # Crear nodos
        router = RouterNode(llm=llm, candidates=self.candidates)
        aggregator = AggregatorNode(llm=llm)
        
        # Crear agentes especialistas (uno por candidato)
        self._specialist_agents = {}
        for candidate in self.candidates:
            # Retriever filtrado por candidato
            retriever = self.vectorstore_manager.get_retriever(
                self.embeddings,
                k=5,  # Top-5 por candidato (menos que TP2 porque es filtrado)
                filter_dict={"candidate": candidate}
            )
            
            agent = SpecialistAgentNode(
                candidate_name=candidate,
                retriever=retriever,
                llm=agent_llm
            )
            
            self._specialist_agents[candidate.lower()] = agent
        
        # Construir grafo
        workflow = StateGraph(AgentState)
        
        # Agregar nodos
        workflow.add_node("router", router)
        workflow.add_node("run_agents", self._run_agents_node)
        workflow.add_node("aggregator", aggregator)
        
        # Definir edges
        workflow.set_entry_point("router")
        workflow.add_edge("router", "run_agents")
        workflow.add_edge("run_agents", "aggregator")
        workflow.add_edge("aggregator", END)
        
        # Compilar
        self.graph = workflow.compile()
    
    def _run_agents_node(self, state: AgentState) -> AgentState:
        """
        Nodo que ejecuta los agentes especialistas seleccionados
        Ejecución SECUENCIAL (no paralela)
        
        Args:
            state: Estado con selected_agents ya definido
            
        Returns:
            Estado actualizado con agent_responses
        """
        selected_agents = state["selected_agents"]
        agent_responses = {}
        
        # Ejecutar cada agente secuencialmente
        for agent_name in selected_agents:
            agent = self._specialist_agents.get(agent_name.lower())
            
            if agent:
                # Ejecutar agente
                response = agent(state)
                agent_responses[agent_name] = response
            else:
                # Agente no encontrado (no debería pasar)
                agent_responses[agent_name] = {
                    "answer": f"Error: Agente {agent_name} no encontrado",
                    "confidence": 0.0,
                    "sources": [],
                    "retrieved_docs": 0
                }
        
        # Actualizar estado
        state["agent_responses"] = agent_responses
        state["metadata"]["agents_executed"] = len(agent_responses)
        
        return state
    
    def query(self, question, chat_history=None, model_name="llama-3.3-70b-versatile", 
              temperature=0.0, custom_instructions=""):
        """
        Procesa una query usando el grafo multi-agente
        
        Args:
            question: Pregunta del usuario
            chat_history: Historial conversacional (lista de Messages)
            model_name: ID del modelo LLM
            temperature: Temperatura del LLM
            custom_instructions: Instrucciones adicionales (no usado en multi-agente por ahora)
            
        Returns:
            dict: {
                "response": str,
                "metadata": {
                    "router_decision": list,
                    "router_reasoning": str,
                    "agents_executed": int,
                    "agent_details": dict
                }
            }
        """
        if not self.candidates:
            raise ValueError("No se han cargado documentos. Llama a load_and_index() primero.")
        
        # Construir grafo si no existe o si cambió el modelo
        if self.graph is None:
            self.build_graph(model_name=model_name, temperature=temperature)
        
        # Preparar estado inicial
        if chat_history is None:
            chat_history = []
        
        initial_state = {
            "question": question,
            "chat_history": chat_history,
            "selected_agents": [],
            "router_reasoning": "",
            "agent_responses": {},
            "final_response": "",
            "metadata": {}
        }
        
        # Ejecutar grafo
        final_state = self.graph.invoke(initial_state)
        
        # Preparar respuesta con metadata
        return {
            "response": final_state["final_response"],
            "metadata": {
                "router_decision": final_state["selected_agents"],
                "router_reasoning": final_state["router_reasoning"],
                "router_method": final_state["metadata"].get("router_method", "unknown"),
                "agents_executed": final_state["metadata"].get("agents_executed", 0),
                "agent_details": {
                    name: {
                        "confidence": resp["confidence"],
                        "retrieved_docs": resp["retrieved_docs"]
                    }
                    for name, resp in final_state["agent_responses"].items()
                }
            }
        }
    
    @property
    def num_documents(self):
        """Retorna número de documentos cargados"""
        return len(self.docs)
