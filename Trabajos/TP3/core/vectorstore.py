"""
core/vectorstore.py
Módulo de gestión de Pinecone compartido entre TP2 y TP3

Maneja la creación, actualización y consulta del índice de vectores
en Pinecone. Soporta filtrado por metadata para TP3.
"""

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter


class PineconeManager:
    """
    Gestor del índice de Pinecone.
    
    Attributes:
        api_key: API key de Pinecone
        index_name: Nombre del índice
        pc: Cliente de Pinecone
        dimension: Dimensión de los embeddings (1024 para BGE-large)
    """
    
    def __init__(self, api_key, index_name, dimension=1024):
        """
        Inicializa el gestor de Pinecone.
        
        Args:
            api_key: API key de Pinecone
            index_name: Nombre del índice a crear/usar
            dimension: Dimensión de los embeddings (default: 1024)
        """
        self.api_key = api_key
        self.index_name = index_name
        self.dimension = dimension
        self.pc = Pinecone(api_key=api_key)
    
    def create_index(self, force_refresh=False):
        """
        Crea el índice en Pinecone si no existe.
        
        Args:
            force_refresh: Si True, elimina el índice existente antes de crear uno nuevo
            
        Returns:
            bool: True si el índice fue creado/recreado, False si ya existía
        """
        import time
        
        # Eliminar índice existente si force_refresh=True
        if force_refresh and self.index_name in self.pc.list_indexes().names():
            self.pc.delete_index(self.index_name)
            # Esperar a que se elimine completamente
            while self.index_name in self.pc.list_indexes().names():
                time.sleep(1)
        
        # Crear índice si no existe
        if self.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="cosine",  # Mejor para embeddings de texto
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            
            # Esperar a que el índice esté completamente listo
            while self.index_name not in self.pc.list_indexes().names():
                time.sleep(1)
            
            # Esperar a que el estado sea "ready"
            index_info = self.pc.describe_index(self.index_name)
            while not index_info.status.ready:
                time.sleep(1)
                index_info = self.pc.describe_index(self.index_name)
            
            return True
        
        return False
    
    def delete_index(self):
        """Elimina el índice de Pinecone."""
        if self.index_name in self.pc.list_indexes().names():
            self.pc.delete_index(self.index_name)
    
    def add_documents(self, docs, embeddings):
        """
        Divide documentos en chunks y los agrega al índice.
        
        Args:
            docs: Lista de Documents de LangChain
            embeddings: Instancia de CleanEmbeddings
            
        Returns:
            PineconeVectorStore: Vectorstore con los documentos indexados
        """
        import time
        
        # Asegurar que el índice existe
        if self.index_name not in self.pc.list_indexes().names():
            raise ValueError(f"El índice '{self.index_name}' no existe. Llama a create_index() primero.")
        
        # Esperar a que el índice esté listo (importante después de creación)
        index_info = self.pc.describe_index(self.index_name)
        while not index_info.status.ready:
            time.sleep(1)
            index_info = self.pc.describe_index(self.index_name)
        
        # Text splitter optimizado para CVs
        # chunk_size=1600: Suficiente para secciones completas
        # chunk_overlap=400: 25% de overlap para evitar pérdida de info
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1600,
            chunk_overlap=400,
            keep_separator=True
        )
        
        # Dividir documentos en chunks
        chunks = splitter.split_documents(docs)
        
        # Agregar metadata de chunk_id
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i
            # Asegurar que candidate esté en metadata
            if "candidate" not in chunk.metadata:
                chunk.metadata["candidate"] = "Unknown"
        
        # Crear vectorstore e indexar
        vectorstore = PineconeVectorStore.from_documents(
            chunks,
            embeddings,
            index_name=self.index_name
        )
        
        return vectorstore
    
    def get_retriever(self, embeddings, k=10, filter_dict=None):
        """
        Crea un retriever del vectorstore.
        
        Args:
            embeddings: Instancia de CleanEmbeddings
            k: Número de documentos a recuperar (default: 10)
            filter_dict: Diccionario de filtros por metadata (ej: {"candidate": "Martin Brocca"})
                        Útil para TP3 donde cada agente filtra por su candidato
            
        Returns:
            VectorStoreRetriever: Retriever configurado
        """
        # Conectar al vectorstore existente
        vectorstore = PineconeVectorStore(
            index_name=self.index_name,
            embedding=embeddings
        )
        
        # Configurar búsqueda
        search_kwargs = {"k": k}
        if filter_dict:
            search_kwargs["filter"] = filter_dict
        
        return vectorstore.as_retriever(search_kwargs=search_kwargs)
