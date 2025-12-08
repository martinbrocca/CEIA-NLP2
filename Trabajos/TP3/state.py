"""
state.py
Define el estado compartido entre nodos del grafo LangGraph
"""

from typing import TypedDict, Annotated, Sequence
from operator import add
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """
    Estado compartido entre todos los nodos del grafo.
    
    Se pasa entre nodos y se actualiza en cada paso:
    Router → Agents → Aggregator
    """
    
    # Input original
    question: str
    chat_history: Annotated[Sequence[BaseMessage], add]
    
    # Decisión del router
    selected_agents: list[str]  # Ej: ["Candidato1", "Candidato2"]
    router_reasoning: str  # Por qué eligió estos agentes
    
    # Respuestas de cada agente especialista
    agent_responses: dict[str, dict]  # {"martin": {"answer": "...", "confidence": 0.9, "sources": [...]}}
    
    # Respuesta final del aggregator
    final_response: str
    
    # Metadata para debugging y métricas
    metadata: dict  # {
                    #   "router_method": "regex" | "llm",
                    #   "agents_executed": 2,
                    #   "total_time": 1.5,
                    #   "token_usage": {...}
                    # }


class AgentResponse(TypedDict):
    """Estructura de respuesta de cada agente especialista"""
    answer: str
    confidence: float  # 0.0 a 1.0
    sources: list[str]  # IDs de los chunks usados
    retrieved_docs: int  # Cuántos docs recuperó
