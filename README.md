# 📈 Asesor Financiero RAG (Retrieval-Augmented Generation)

Este proyecto implementa un sistema de Inteligencia Artificial diseñado para actuar como un asesor financiero personal. Utiliza una arquitectura **RAG** para consultar una base de conocimiento compuesta por 8 libros clásicos de educación financiera y generar respuestas fundamentadas, precisas y con citas directas al autor.

## 🚀 Características Principales

* **Procesamiento de Documentos Inteligente:** Carga de PDFs con inyección de metadatos (autor, título y tema) para asegurar la trazabilidad de cada consejo financiero.
* **Vectorización Acelerada por GPU:** Uso de embeddings multilingües (`google/embeddinggemma-300m`) ejecutados de manera local optimizando los recursos de hardware (CUDA).
* **Retrieval Avanzado (Pipeline Híbrido):** 
  * **MMR (Maximal Marginal Relevance):** Maximiza la diversidad de las fuentes (ej. contrastando las ideas de los diferentes autores).
  * **Contextual Compression:** Un LLM intermedio extrae únicamente la información vital de los fragmentos recuperados, eliminando el ruido ("basura contextual") antes de la inferencia final.
* **Generación con Atribución:** El modelo de lenguaje responde inyectando las citas exactas de dónde provino la información, evitando alucinaciones y sesgos.

## 🛠️ Stack Tecnológico

* **Framework Core:** [LangChain](https://www.langchain.com/)
* **Modelos de Lenguaje (LLM):** `gemini-2.5-flash-lite` y `gemma-3-27b-it` (vía Google Generative AI)
* **Modelos de Embeddings:** HuggingFace (`google/embeddinggemma-300m`)
* **Base de Datos Vectorial:** ChromaDB (Persistente en almacenamiento local)
* **Procesamiento de Texto:** `PyPDFLoader` y `RecursiveCharacterTextSplitter`

## ⚙️ Instalación y Configuración del Entorno

Para garantizar la reproducibilidad del proyecto y evitar conflictos con las dependencias (especialmente con librerías de GPU y LangChain), se recomienda usar **Conda**.

### Opción 1: Usando Conda (Recomendado)
Si tienes Anaconda o Miniconda instalado, puedes recrear el entorno exacto ejecutando:
```bash
# Clonar el repositorio
git clone <URL_DE_TU_REPOSITORIO>

# Crear el entorno a partir del archivo YAML
conda env create -f environment.yml

# Activar el entorno
conda activate Financial_advisor
```

### Opción 2: Usando Pip
Si prefieres usar entornos virtuales de Python estándar (`venv`):
```bash
# Crear y activar el entorno virtual
python -m venv env
source env/bin/activate  # En Windows usa: env\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

> [!warning] Advertencia: Variables de entorno
> Necesitarás una API Key de Google (Google AI Studio) para usar los modelos de generación.


## 🧠 Arquitectura del Pipeline de Datos

1. **Ingesta (Document Loading & Splitting):** Los 8 libros se leen, se les asignan metadatos y se dividen en *chunks* de 1200 caracteres con un *overlap* de 200 caracteres para no perder contexto.
2. **Base Vectorial (ChromaDB):** Los fragmentos se transforman en vectores de 768 dimensiones y se almacenan localmente en el directorio `./db/vectorial-db/` para evitar el re-procesamiento continuo.
3. **Recuperación (Retrieval):** 
   * Búsqueda inicial de los 25 fragmentos más relevantes (`fetch_k=25`).
   * Filtrado MMR para seleccionar los 4 más diversos (`k=4`).
   * Compresión contextual (`LLMChainExtractor`) para limpiar los textos.
4. **Respuesta (Question Answering):** La consulta del usuario y el contexto comprimido se envían al LLM final usando LCEL (`ChatPromptTemplate | model | StrOutputParser`) asegurando la cita de fuentes.

## 💻 Uso

Una vez configurado el entorno y la API Key, el proyecto se puede ejecutar de manera secuencial usando Jupyter Notebooks:

```python
# Ejemplo de consulta al Asesor Financiero
query = "¿Cuáles son los pasos a seguir para empezar a invertir?"

# El sistema recuperará el contexto comprimido y generará la respuesta
retrievedDocs = compressedRetriever.invoke(query)
reference = format_docs(retrievedDocs)

answer = chain.invoke({
    "query": query,
    "reference": reference,
})

print(answer)
```

<!-- ## 📈 Próximos Pasos (To-Do)
- [ ] Implementar un historial de memoria (`ConversationBufferMemory`) para sostener diálogos de múltiples turnos.
- [ ] Desarrollar una interfaz gráfica básica usando Streamlit o Gradio.
- [ ] Incorporar Búsqueda Léxica (BM25) para crear un Ensemble Retriever. -->
