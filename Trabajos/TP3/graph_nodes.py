"""
graph_nodes.py
Implementación de los nodos del grafo LangGraph:
- Router: Decide qué agentes invocar
- Specialist Agents: Responden sobre candidatos específicos
- Aggregator: Sintetiza respuestas de múltiples agentes
"""

import re
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from state import AgentState, AgentResponse
from core import format_docs_simple


class RouterNode:
    """
    Nodo router: Decide qué agentes especialistas deben ejecutarse
    Estrategia híbrida: Regex primero, LLM como fallback
    """
    
    def __init__(self, llm, candidates, default_agent="martin"):
        """
        Args:
            llm: Instancia de LLM para routing inteligente
            candidates: Lista de nombres de candidatos disponibles (nombres completos)
            default_agent: Agente a usar por defecto cuando no se menciona ningún candidato
        """
        self.llm = llm
        # Guardar nombres completos para display
        self.full_candidates = candidates
        # Extraer primeros nombres para matching (lowercase)
        self.candidates = [c.split()[0].lower() for c in candidates]
        self.default_agent = default_agent.lower()
        
        # Palabras clave para detección de comparaciones
        self.comparison_keywords = [
            "compar", "versus", "vs", "mejor", "peor", "más", "menos",
            "diferencia", "ambos", "todos", "cuál", "cual", "quien", "quién"
        ]
    
    def _regex_route(self, question: str) -> tuple[list[str], str]:
        """
        Routing rápido basado en regex
        
        Returns:
            (selected_agents, reasoning)
        """
        question_lower = question.lower()
        
        # Detectar nombres de candidatos mencionados
        mentioned = [c for c in self.candidates if c in question_lower]
        
        # Caso 1: Menciona candidatos específicos
        if mentioned:
            # Si menciona múltiples o palabras comparativas → todos los mencionados
            if len(mentioned) > 1 or any(kw in question_lower for kw in self.comparison_keywords):
                return mentioned, f"Regex: Detectó comparación entre {', '.join(mentioned)}"
            # Si menciona solo uno → ese candidato
            else:
                return mentioned, f"Regex: Detectó pregunta específica sobre {mentioned[0]}"
        
        # Caso 2: Pregunta comparativa sin nombres específicos → todos
        if any(kw in question_lower for kw in self.comparison_keywords):
            return self.candidates, "Regex: Detectó pregunta comparativa, consultando todos"
        
        # Caso 3: No menciona candidatos ni es comparación → agente por defecto
        return [self.default_agent], f"Regex: Sin mención específica, usando agente por defecto ({self.default_agent})"
    
    def _llm_route(self, question: str) -> tuple[list[str], str]:
        """
        Routing inteligente usando LLM
        
        Returns:
            (selected_agents, reasoning)
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Eres un router que decide qué agentes especialistas deben responder una pregunta.

Candidatos disponibles: {candidates}
Agente por defecto: {default_agent}

Reglas:
- Si la pregunta es sobre UN candidato específico → retorna solo ese nombre
- Si es comparación o menciona "todos" → retorna "all"
- Si NO menciona candidatos específicos → retorna "default"

Formato de respuesta: SOLO una de estas opciones:
- Un nombre: "martin" o "ariadna"
- Múltiples: "martin,ariadna"
- Todos: "all"
- Por defecto: "default"

NO agregues explicaciones."""),
            ("human", "{question}")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        
        result = chain.invoke({
            "question": question,
            "candidates": ", ".join(self.candidates),
            "default_agent": self.default_agent
        }).strip().lower()
        
        # Parse resultado
        if result == "all":
            return self.candidates, "LLM: Pregunta general/comparativa, consultando todos"
        elif result == "default":
            return [self.default_agent], f"LLM: Sin mención específica, usando agente por defecto ({self.default_agent})"
        else:
            selected = [c.strip() for c in result.split(",") if c.strip() in self.candidates]
            if not selected:
                # Fallback: usar agente por defecto
                return [self.default_agent], f"LLM: Respuesta inválida, usando agente por defecto ({self.default_agent})"
            return selected, f"LLM: Seleccionó agentes específicos: {', '.join(selected)}"
    
    def __call__(self, state: AgentState) -> AgentState:
        """
        Ejecuta el routing híbrido
        
        Args:
            state: Estado actual del grafo
            
        Returns:
            Estado actualizado con selected_agents y router_reasoning
        """
        question = state["question"]
        
        # Paso 1: Intentar regex
        selected, reasoning = self._regex_route(question)
        method = "regex"
        
        # Paso 2: Si regex no dio match, usar LLM
        if selected is None:
            selected, reasoning = self._llm_route(question)
            method = "llm"
        
        # Actualizar estado
        state["selected_agents"] = selected
        state["router_reasoning"] = reasoning
        state["metadata"] = {
            "router_method": method,
            "agents_to_call": len(selected)
        }
        
        return state


class SpecialistAgentNode:
    """
    Nodo de agente especialista: Responde preguntas sobre un candidato específico
    """
    
    def __init__(self, candidate_name, retriever, llm):
        """
        Args:
            candidate_name: Nombre del candidato (ej: "Martin")
            retriever: Retriever filtrado para este candidato
            llm: Instancia de LLM para generar respuestas
        """
        self.candidate_name = candidate_name
        self.retriever = retriever
        self.llm = llm
        
        # Prompt especializado para este agente
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", f"""Eres un agente especialista que analiza el CV del candidato {candidate_name}.

Tu trabajo:
- Responder preguntas SOBRE {candidate_name} basándote en su CV
- IMPORTANTE: Siempre responde en TERCERA PERSONA mencionando el nombre del candidato
- Ejemplo: "{candidate_name} tiene experiencia en..." o "{candidate_name} maneja..."
- Si no tienes información suficiente, di "No se menciona información sobre esto en el CV de {candidate_name}"
- Sé conciso y preciso
- NUNCA respondas en primera persona ("Yo tengo...", "Manejo...")

NO menciones otros candidatos, solo enfócate en {candidate_name}.

Contexto del CV de {candidate_name}:
{{context}}"""),
            ("human", "{question}")
        ])
    
    def _calculate_confidence(self, docs, answer) -> float:
        """
        Calcula confidence score basado en:
        - Número de documentos recuperados
        - Presencia de indicadores de incertidumbre en la respuesta
        
        Returns:
            float entre 0.0 y 1.0
        """
        # Base: más docs recuperados = mayor confianza
        doc_score = min(len(docs) / 5.0, 1.0)  # Máximo en 5 docs
        
        # Penalizar si hay indicadores de incertidumbre
        uncertainty_phrases = [
            "no tengo información", "no se menciona", "no está claro",
            "no puedo determinar", "no aparece", "no especifica"
        ]
        
        has_uncertainty = any(phrase in answer.lower() for phrase in uncertainty_phrases)
        
        if has_uncertainty:
            return min(doc_score * 0.3, 0.4)  # Máximo 40% si hay incertidumbre
        
        return doc_score
    
    def __call__(self, state: AgentState) -> AgentResponse:
        """
        Ejecuta el agente especialista
        
        Args:
            state: Estado del grafo
            
        Returns:
            AgentResponse con answer, confidence, sources
        """
        question = state["question"]
        
        # Recuperar documentos relevantes para este candidato
        docs = self.retriever.invoke(question)
        
        if not docs:
            # No hay información
            return {
                "answer": f"No tengo información relevante sobre esto en el CV de {self.candidate_name}.",
                "confidence": 0.0,
                "sources": [],
                "retrieved_docs": 0
            }
        
        # Formatear contexto
        context = format_docs_simple(docs)
        
        # Generar respuesta
        chain = self.prompt | self.llm | StrOutputParser()
        answer = chain.invoke({
            "question": question,
            "context": context
        })
        
        # Calcular confidence
        confidence = self._calculate_confidence(docs, answer)
        
        # Extraer source IDs
        sources = [doc.metadata.get("chunk_id", f"chunk_{i}") for i, doc in enumerate(docs)]
        
        return {
            "answer": answer,
            "confidence": confidence,
            "sources": sources[:3],  # Top-3 sources
            "retrieved_docs": len(docs)
        }


class AggregatorNode:
    """
    Nodo aggregator: Sintetiza respuestas de múltiples agentes
    """
    
    def __init__(self, llm):
        """
        Args:
            llm: Instancia de LLM para sintetizar respuestas
        """
        self.llm = llm
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """Eres un agregador que sintetiza información de múltiples agentes especialistas.

Tu trabajo:
- Combinar las respuestas de los agentes en una respuesta coherente
- Si solo hay un agente, pasar su respuesta directamente (mejorada si es necesario)
- Si hay múltiples agentes, hacer comparaciones claras
- Priorizar agentes con mayor confidence
- Mantener respuestas concisas (máximo 2-3 párrafos)

Respuestas de los agentes:
{agent_responses}"""),
            ("human", "{question}")
        ])
    
    def _format_agent_responses(self, agent_responses: dict) -> str:
        """Formatea las respuestas de agentes para el prompt"""
        formatted = []
        
        for agent_name, response in agent_responses.items():
            conf_pct = int(response["confidence"] * 100)
            formatted.append(f"""
Agente: {agent_name.capitalize()}
Confidence: {conf_pct}%
Respuesta: {response["answer"]}
""")
        
        return "\n".join(formatted)
    
    def __call__(self, state: AgentState) -> AgentState:
        """
        Agrega respuestas de múltiples agentes
        
        Args:
            state: Estado con agent_responses poblado
            
        Returns:
            Estado actualizado con final_response
        """
        agent_responses = state["agent_responses"]
        question = state["question"]
        
        # Caso especial: Solo un agente
        if len(agent_responses) == 1:
            agent_name = list(agent_responses.keys())[0]
            response = agent_responses[agent_name]
            
            # Si confidence es muy baja, agregar disclaimer
            if response["confidence"] < 0.5:
                state["final_response"] = f"{response['answer']}\n\n(Nota: Confidence score: {int(response['confidence']*100)}%)"
            else:
                state["final_response"] = response["answer"]
            
            return state
        
        # Múltiples agentes: sintetizar
        formatted_responses = self._format_agent_responses(agent_responses)
        
        chain = self.prompt | self.llm | StrOutputParser()
        final_response = chain.invoke({
            "question": question,
            "agent_responses": formatted_responses
        })
        
        state["final_response"] = final_response
        
        return state