"""
core/embeddings.py
Módulo de embeddings compartido entre TP2 y TP3

Proporciona una clase wrapper para HuggingFaceEmbeddings que sanitiza
el texto antes de generar embeddings, evitando errores con caracteres
especiales o formatos inesperados de la extracción de PDFs.
"""

from langchain_huggingface import HuggingFaceEmbeddings


class CleanEmbeddings:
    """
    Wrapper para HuggingFaceEmbeddings que sanitiza el texto.
    
    Attributes:
        emb: Instancia de HuggingFaceEmbeddings
        model_name: Nombre del modelo de embeddings
    """
    
    def __init__(self, model_name="BAAI/bge-large-en-v1.5"):
        """
        Inicializa el modelo de embeddings.
        
        Args:
            model_name: Modelo de HuggingFace a usar (default: BGE-large)
                       BGE-large tiene 1024 dimensiones y excelente calidad
        """
        self.model_name = model_name
        self.emb = HuggingFaceEmbeddings(model_name=model_name)

    def _sanitize(self, text):
        """
        Limpia el texto antes de embeddearlo.
        
        PyPDF2 puede extraer texto con caracteres especiales, diccionarios,
        o formatos inesperados. Esta función normaliza todo a string limpio.
        
        Args:
            text: Texto a sanitizar (puede ser str, dict, u otro tipo)
            
        Returns:
            str: Texto limpio sin saltos de línea
        """
        # Convertir a string si es necesario
        if isinstance(text, dict):
            text = str(text)
        if not isinstance(text, str):
            text = str(text)
        
        # Reemplazar saltos de línea por espacios
        # Los embeddings funcionan mejor con texto continuo
        return text.replace("\n", " ")

    def embed_query(self, text):
        """
        Genera embedding para una query de búsqueda.
        
        Args:
            text: Query a embeddear
            
        Returns:
            list[float]: Vector de embeddings (1024 dimensiones para BGE-large)
        """
        return self.emb.embed_query(self._sanitize(text))

    def embed_documents(self, texts):
        """
        Genera embeddings para múltiples documentos.
        
        Args:
            texts: Lista de documentos a embeddear
            
        Returns:
            list[list[float]]: Lista de vectores de embeddings
        """
        cleaned = [self._sanitize(t) for t in texts]
        return self.emb.embed_documents(cleaned)