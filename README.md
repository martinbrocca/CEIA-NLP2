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


### Trabajo Práctico Nº 2 – Chatbot RAG para comparación de CVs 

[Ir al TP](./Trabajos/TP2/README.md) · Ejecutar: `streamlit run Trabajos/TP2/chatbot.py`

    Objetivo:
        Desarrollar un asistente inteligente que permita responder consultas o comparar dos (o más) currículums en lenguaje natural.  
        Ejemplos de preguntas:

            - ¿Que skills tiene el candidato?
            - ¿Tiene experiencia en hotelería?
            - ¿Tiene experiencia en pre-sales o ventas?
            - ¿Quién tiene certificación PMP?
            - ¿Quién ha trabajado con observabilidad / monitoring?
            - ¿En qué universidad estudió?

### Trabajo Práctico Nº 3 – Chatbot RAG multiagente para comparación de CVs

[Ir al TP](./Trabajos/TP3/README.md) · Ejecutar: `streamlit run Trabajos/TP3/chat_bot_multi.py`

    Objetivo:
        Desarrollar un agentes que responda de manera eficiente sobre currículums de personas en forma eficiente. El sistema tiene un agente por persona y permita responder consultas o comparar dos (o más) currículums en lenguaje natural.  
        Ejemplos de preguntas:

            - ¿Quién vive en Texas?
            - ¿Algún candidato tiene experiencia en hotelería?
            - ¿Quién tiene experiencia en pre-sales o ventas?
            - ¿Quién tiene certificación PMP?
            - ¿Quién ha trabajado con observabilidad / monitoring?
            - ¿En qué universidad estudiaron?
