"""
core package
Módulos compartidos entre TP2 y TP3
"""

from .embeddings import CleanEmbeddings
from .vectorstore import PineconeManager
from .llm import LLMFactory
from .utils import (
    extract_candidate_name,
    extract_location,
    load_resume_pdfs,
    format_docs_simple,
    format_docs_grouped
)

__all__ = [
    "CleanEmbeddings",
    "PineconeManager",
    "LLMFactory",
    "extract_candidate_name",
    "extract_location",
    "load_resume_pdfs",
    "format_docs_simple",
    "format_docs_grouped"
]
