![](./images/nlp_banner.png)



## Especialización en Inteligencia Artificial 
## Materia de Procesamiento del Lenguaje Natural 2 


### Alumno: Martin Brocca

### Contenido: 


### Trabajo Práctico Nº 1 – TinyGPT & Mixture of Experts
[Ver notebook principal](./Trabajos/TP1/TinyGPT.ipynb) 
    - Implementación y comparación de arquitecturas Transformer para generación de texto a nivel carácter: modelo base GPT vs. Mixture of Experts (MoE), entrenados en Shakespeare con análisis de perplejidad, comportamiento de generación y distribución de uso de expertos.
    Análisis Comparativo: Transformer Base vs. Mixture of Experts
    
        Implementación desde cero de dos arquitecturas de lenguaje:
           - Modelo Base: Transformer GPT estándar (2 capas, 4 heads, 128d embedding)
           - Modelo MoE: Arquitectura con 4 expertos especializados (top-k=2)
           - Dataset: Complete Works of Shakespeare (generación carácter a carácter)
           - Métricas: Perplejidad, entropía de vocabulario, análisis de logits, mode collapse detection
           - Resultado: MoE logra 30% mejor perplejidad pero requiere ajuste cuidadoso para evitar colapso de modoRetry


### Trabajo Práctico Nº 2 – Chatbot RAG para comparación de CVs (este proyecto)

[Ver código](./Trabajos/TP2/chatbot.py) · Ejecutar: `streamlit run Trabajos/TP2/chatbot.py`

    Objetiv\o:
        Desarrollar un asistente inteligente que permita responder consultas o comparar dos (o más) currículums en lenguaje natural.  
        Ejemplos de preguntas:

            - ¿Quién vive en Texas?
            - ¿Algún candidato tiene experiencia en hotelería?
            - ¿Quién tiene experiencia en pre-sales o ventas?
            - ¿Quién tiene certificación PMP?
            - ¿Quién ha trabajado con observabilidad / monitoring?
            - ¿En qué universidad estudiaron?

    Características técnicas (estado del arte 2025)
    - **Embedding**: `BAAI/bge-large-en-v1.5` (1024 dim) 
    - **Vector DB**: Pinecone (serverless, índice dedicado por dimensión)
    - **Chunking inteligente**: 1600 caracteres + 400 de overlap + metadata de candidato en cada chunk
    - **Retriever**: top-k=10 sin reranker 
    - **LLM**: Llama-3.3-70B (Groq) a temperatura 0.0 + prompt optimizado para responder sobre CVs
    - **Interfaz**: Streamlit

    Modo de uso
```bash
# 1. Clonar el repositorio y acceder a la carpeta
cd ./CEIA-NLP2

# 2. Activar el entorno
source .venv/bin/activate

# 3. Subir los CVs en la carpeta correcta de curriculumns
cp "mis-cv-*.pdf" Trabajos/TP2/resumes/

# 4. Ejecutar el chatbot
streamlit run Trabajos/TP2/chatbot.py
```


### Estructura del repositorio:
```bash
├── images/
│   ├── nlp_banner.png
│   └── nlp.png
├── resumes/                      ← (raíz antigua, opcional)
├── Trabajos/
│   ├── TP1/
│   │   ├── TinyGPT.ipynb
│   │   ├── TinyGPT_es.ipynb
│   │   ├── trainer.py
│   │   └── checkpoints/         ← pesos de modelos entrenados
│   └── TP2/
│       ├── chatbot.py           
│       ├── class_exercise.py
│       ├── test_torch.ipynb
│       └── resumes/             
│           ├── Ariadna Garmendia - Resume - 2024.pdf
│           ├── Martin Brocca - Solution Architect Resume.pdf
│           └── Resume - Martin Brocca.docx
├── main.py
├── pyproject.toml
├── uv.lock
└── README.md
```