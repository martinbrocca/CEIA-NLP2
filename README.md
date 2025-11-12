![](./images/nlp_banner.png)



## Especialización en Inteligencia Artificial 
## Materia de Procesamiento del Lenguaje Natural 2 


### Alumno: Martin Brocca

### Contenido: 


[Trabajo Práctico N1](./Trabajos/TP1/TinyGPT.ipynb) :
    - Implementación y comparación de arquitecturas Transformer para generación de texto a nivel carácter: modelo base GPT vs. Mixture of Experts (MoE), entrenados en Shakespeare con análisis de perplejidad, comportamiento de generación y distribución de uso de expertos.
    Análisis Comparativo: Transformer Base vs. Mixture of Experts
    
        Implementación desde cero de dos arquitecturas de lenguaje:
           - Modelo Base: Transformer GPT estándar (2 capas, 4 heads, 128d embedding)
           - Modelo MoE: Arquitectura con 4 expertos especializados (top-k=2)
           - Dataset: Complete Works of Shakespeare (generación carácter a carácter)
           - Métricas: Perplejidad, entropía de vocabulario, análisis de logits, mode collapse detection
           - Resultado: MoE logra 30% mejor perplejidad pero requiere ajuste cuidadoso para evitar colapso de modoRetry


<!--   
[desafío 2](./desafios/desafio_2/Desafio_2.ipynb) :
    - Generación y ensayo de embeddings creados con la librería Gensim a partir de los modelos CBOW y Skipgram. Los embeddings se generaron a partir de un corpus en inglés basado en las novelas de Harry Potter. Para visualizar agrupación entre vectores se utilizó TSNE con librería Sklearn.
  
[desafío 3](./desafios/desafio_3/Desafio_3.ipynb) :
    - Modelo de lenguaje con tokenizacion por caracteres: Implementacion y comparacion de tres arquitecturas de redes neuronales recurrentes (SimpleRNN, LSTM y GRU) para la generación de texto a nivel de caracteres utilizando el corpus de las novelas de Harry Potter.
    
[desafío 4](./desafios/desafio_4/Desafio_4.ipynb) :
    - Implementación y comparación de arquitecturas Seq2Seq (con y sin atención) para traducción automática inglés-español. Experimentación sistemática con 4 configuraciones que explora el impacto crítico del tamaño de vocabulario, arquitectura del encoder y embeddings, alcanzando un modelo final con 66.41% de precisión y ~15 BLEU mediante encoder bidireccional y vocabulario optimizado de 5K palabras. -->

### Estructura del repositorio:
```bash
.
├── Trabajos
│   ├── TP1
│   │   └── TinyGPT_es.ipynb
│   |   ├── TinyGPT.ipynb
│   |   ├── trainer.py
├── images
│   └── nlp.png
└── README.md
