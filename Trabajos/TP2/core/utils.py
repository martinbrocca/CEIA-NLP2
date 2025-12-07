"""
core/utils.py
Utilidades compartidas entre TP2 y TP3

Funciones helper para procesamiento de PDFs, extracción de metadata
y formateo de documentos.
"""

import re
from pathlib import Path
import PyPDF2
from langchain_core.documents import Document


def extract_candidate_name(filename: str) -> str:
    """
    Se asume que el nombre del candidato está en el nombre del archivo PDF.

    Extrae el nombre del candidato del nombre del archivo PDF.
    
    Limpia el nombre removiendo texto común como "Resume", "CV", años, etc.
    
    Args:
        filename: Nombre del archivo PDF (ej: "Martin Brocca - Resume 2024.pdf")
        
    Returns:
        str: Nombre limpio del candidato (ej: "Martin Brocca")
        
    Examples:
        >>> extract_candidate_name("Martin Brocca - Resume 2024.pdf")
        "Martin Brocca"
        >>> extract_candidate_name("Jane_Doe_CV.pdf")
        "Jane Doe"
    """
    # Obtener nombre sin extensión
    name = Path(filename).stem
    
    # Remover ruido común en nombres de archivos
    noise_patterns = [
        " - Resume", " Resume", " - CV", " CV",
        " 2024", " 2025", "-2024", "(Final)", " - final"
    ]
    for noise in noise_patterns:
        name = name.replace(noise, "")
    
    # Si hay guión, tomar solo la primera parte
    if " - " in name:
        name = name.split(" - ")[0]
    
    # Limpiar guiones y underscores
    name = name.replace("_", " ").replace("-", " ")
    
    # Tomar solo palabras que empiezan con mayúscula (nombres propios)
    parts = [p for p in name.split() if p and p[0].isupper()]
    
    # Retornar máximo 3 palabras (nombre completo típico)
    return " ".join(parts[:3]) if len(parts) >= 2 else name.strip().title()


def extract_location(header_text: str) -> str:
    """
    Extrae la ubicación del candidato del header de su CV.
    Esta función suplementa la extracción básica de metadata debido a irregularidad 
    de formatos o complejidad de los CVs de presentar de forma clara la ubicación.

    Busca patrones comunes de ubicación (Ciudad, Estado/País, ZIP).
    Maneja espacios irregulares que PyPDF2 puede introducir al extraer texto.
    
    Args:
        header_text: Primeros ~300 caracteres del CV
        
    Returns:
        str: Ubicación formateada o "ubicación no detectada"
        
    Examples:
        >>> extract_location("Spring, TX, 77389, United States")
        "Spring, TX, 77389, United States"
        >>> extract_location("Buenos Aires, Argentina")
        "Buenos Aires, Argentina"
    """
    # Normalizar espacios múltiples que PyPDF2 puede introducir
    # Ejemplo: "Spring , TX , 77389" -> "Spring, TX, 77389"
    header_text = re.sub(r'\s+', ' ', header_text)
    
    # Patrones de ubicación ordenados de más específico a más genérico
    patterns = [
        # Ciudad, Estado, ZIP, País (ej: Spring, TX, 77389, United States)
        r'([A-Z][a-z]+\s*,\s*[A-Z]{2}\s*,\s*\d{5}\s*,\s*United States)',
        # Ciudad, Estado, ZIP (ej: Spring, TX, 77389)
        r'([A-Z][a-z]+\s*,\s*[A-Z]{2}\s*,\s*\d{5})',
        # Ciudad, Estado, País (ej: Spring, TX, United States)
        r'([A-Z][a-z]+\s*,\s*[A-Z]{2}\s*United States)',
        # Cualquier ciudad con TX (ej: Houston, TX, 77001)
        r'([A-Za-z\s]+\s*,\s*TX\s*,\s*\d{5})',
        # Ciudad, País (ej: Buenos Aires, Argentina)
        r'([A-Z][a-z\s]+\s*,\s*Argentina)',
    ]
    
    # Intentar cada patrón
    for pattern in patterns:
        match = re.search(pattern, header_text, re.IGNORECASE)
        if match:
            location = match.group(1).strip()
            # Limpiar espacios extra alrededor de comas
            location = re.sub(r'\s*,\s*', ', ', location)
            return location
    
    # Fallback: buscar líneas que contengan indicadores de ubicación
    lines = header_text.split('\n')
    for line in lines:
        line_clean = line.strip()
        # Buscar TX, United States o USA
        if 'TX' in line_clean or 'United States' in line_clean or 'USA' in line_clean:
            # Limpiar formato
            line_clean = re.sub(r'\s*,\s*', ', ', line_clean)
            return line_clean
    
    return "ubicación no detectada"


def load_resume_pdfs(folder_path: str) -> list[Document]:
    """
    Carga todos los PDFs de una carpeta y los convierte a Documents.
    
    Para cada PDF:
    1. Extrae el texto completo
    2. Identifica el nombre del candidato
    3. Extrae la ubicación del header
    4. Crea un Document con metadata
    
    Args:
        folder_path: Ruta a la carpeta con PDFs
        
    Returns:
        list[Document]: Lista de documentos con metadata
        
    Raises:
        FileNotFoundError: Si la carpeta no existe
    """
    docs = []
    folder = Path(folder_path)
    
    # Crear carpeta si no existe
    folder.mkdir(exist_ok=True, parents=True)
    
    # Verificar que haya PDFs
    pdf_files = list(folder.glob("*.pdf"))
    if not pdf_files:
        return docs

    # Procesar cada PDF
    for pdf_path in sorted(pdf_files):
        try:
            # Leer PDF
            with open(pdf_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                # Extraer texto de todas las páginas
                text = "\n".join(page.extract_text() or "" for page in reader.pages)
            
            # Extraer información del candidato
            candidate = extract_candidate_name(pdf_path.name)
            
            # Extraer ubicación del header (primeros 300 caracteres)
            header = text[:300]
            location = extract_location(header)
            
            # Crear documento con metadata
            docs.append(Document(
                page_content=text,
                metadata={
                    "candidate": candidate,      # Nombre del candidato
                    "source": pdf_path.name,     # Nombre del archivo
                    "location": location         # Ubicación extraída
                }
            ))
            
        except Exception as e:
            # Si hay error, continuar con el siguiente archivo
            print(f"Error procesando {pdf_path.name}: {e}")
            continue
    
    return docs


def format_docs_simple(docs) -> str:
    """
    Formatea documentos para contexto del LLM (versión simple para TP2).
    
    Args:
        docs: Lista de documentos recuperados por el retriever
        
    Returns:
        str: Contexto formateado
    """
    if not docs:
        return "No se encontró información relevante."
    
    # Agrupar por candidato
    candidates_info = {}
    for d in docs:
        cand = d.metadata.get("candidate", "Unknown")
        if cand not in candidates_info:
            candidates_info[cand] = {
                "location": d.metadata.get("location", "ubicación desconocida"),
                "source": d.metadata.get("source", ""),
                "chunks": []
            }
        candidates_info[cand]["chunks"].append(d.page_content.strip())
    
    # Formatear con headers claros
    parts = []
    for cand, info in candidates_info.items():
        # Header con información del candidato
        header = f"\n{'='*60}\n"
        header += f"CANDIDATO: {cand}\n"
        header += f"Ubicación: {info['location']}\n"
        header += f"Fuente: {info['source']}\n"
        header += f"{'='*60}\n"
        parts.append(header)
        
        # Chunks de contenido
        for chunk in info["chunks"]:
            parts.append(chunk)
            parts.append("\n---\n")
    
    return "\n".join(parts)


def format_docs_grouped(docs) -> dict:
    """
    Agrupa documentos por candidato (útil para TP3 multi-agente).
    
    Args:
        docs: Lista de documentos recuperados
        
    Returns:
        dict: {candidate_name: [chunks]}
    """
    grouped = {}
    for d in docs:
        cand = d.metadata.get("candidate", "Unknown")
        if cand not in grouped:
            grouped[cand] = []
        grouped[cand].append(d.page_content.strip())
    
    return grouped