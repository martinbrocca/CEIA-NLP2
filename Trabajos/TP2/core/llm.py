"""
core/llm.py
Módulo de gestión de LLMs compartido entre TP2 y TP3
Soporta Groq y Anthropic (Claude)
"""

from langchain_groq import ChatGroq
from langchain_anthropic import ChatAnthropic


class LLMFactory:
    """Factory para crear instancias de LLMs de Groq o Anthropic"""
    
    GROQ_MODELS = {
        "llama-3.3-70b-versatile": {
            "name": "Llama 3.3 70B (Groq)",
            "description": "Modelo más potente de Groq"
        },
        "llama-3.1-70b-versatile": {
            "name": "Llama 3.1 70B (Groq)",
            "description": "Versión anterior del 70B"
        },
        "mixtral-8x7b-32768": {
            "name": "Mixtral 8x7B (Groq)",
            "description": "Modelo mixture-of-experts"
        },
        "llama-3.1-8b-instant": {
            "name": "Llama 3.1 8B (Groq - Rápido)",
            "description": "Modelo más rápido"
        }
    }
    
    ANTHROPIC_MODELS = {
        "claude-sonnet-4-20250514": {
            "name": "Claude Sonnet 4.5 (Anthropic)",
            "description": "Modelo más inteligente"
        },
        "claude-3-5-sonnet-20241022": {
            "name": "Claude 3.5 Sonnet Oct (Anthropic)",
            "description": "Versión octubre 2024"
        },
        "claude-3-5-sonnet-20240620": {
            "name": "Claude 3.5 Sonnet (Anthropic)",
            "description": "Balance calidad/velocidad"
        },
        "claude-3-5-haiku-20241022": {
            "name": "Claude 3.5 Haiku (Anthropic)",
            "description": "Rápido y económico"
        }
    }
    
    @staticmethod
    def get_available_models():
        """Retorna todos los modelos disponibles"""
        all_models = {}
        for model_id, info in LLMFactory.GROQ_MODELS.items():
            all_models[model_id] = {**info, "provider": "groq"}
        for model_id, info in LLMFactory.ANTHROPIC_MODELS.items():
            all_models[model_id] = {**info, "provider": "anthropic"}
        return all_models
    
    @staticmethod
    def get_model_names():
        """Retorna dict de nombres legibles a IDs"""
        models = LLMFactory.get_available_models()
        return {info["name"]: model_id for model_id, info in models.items()}
    
    @staticmethod
    def create_llm(model_name, temperature, groq_api_key=None, anthropic_api_key=None):
        """Crea instancia de LLM (Groq o Anthropic según el modelo)"""
        all_models = LLMFactory.get_available_models()
        
        if model_name not in all_models:
            raise ValueError(f"Modelo desconocido: {model_name}")
        
        provider = all_models[model_name]["provider"]
        
        if provider == "groq":
            if not groq_api_key:
                raise ValueError("Se requiere GROQ_API_KEY para usar modelos de Groq")
            return ChatGroq(
                model=model_name,
                temperature=temperature,
                groq_api_key=groq_api_key
            )
        elif provider == "anthropic":
            if not anthropic_api_key:
                raise ValueError("Se requiere ANTHROPIC_API_KEY para usar modelos de Claude")
            return ChatAnthropic(
                model=model_name,
                temperature=temperature,
                anthropic_api_key=anthropic_api_key
            )
        else:
            raise ValueError(f"Provider desconocido: {provider}")