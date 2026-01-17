# Sanskrit RAG: Local Document Intelligence System

A Retrieval-Augmented Generation (RAG) pipeline designed for querying Sanskrit literature. This system is engineered for **CPU-only inference**, allowing it to run on standard hardware without the need for a GPU.

## 🚀 System Architecture

The pipeline is split into a modular ingestion-retrieval-generation workflow, optimized for the Devanagari script and local RAM constraints.



### 1. Ingestion Layer
* **Parser:** Extracts text from `.docx` files while preserving Unicode/Sanskrit characters.
* **Chunking:** Uses `RecursiveCharacterTextSplitter` with paragraph-priority boundaries to keep Sanskrit verses (*Shlokas*) and their contexts intact.

### 2. Retrieval Layer
* **Embeddings:** `paraphrase-multilingual-MiniLM-L12-v2` is used to map Sanskrit queries to a 384-dimensional vector space.
* **Vector Store:** Local **FAISS** index for high-speed semantic similarity search on system memory.

### 3. Generation Layer
* **Model:** **Phi-3-mini-4k-instruct-q4**, a 3.8B parameter quantized Small Language Model (SLM) known for high reasoning efficiency on CPUs.
* **Inference Engine:** `llama-cpp-python` configured with multi-threaded execution to maximize local performance.

---

## 🛠️ Setup & Installation

### 1. Prerequisites
Ensure you have **Python 3.10+** and the **Microsoft Visual C++ Redistributable** installed on your Windows machine.

### 2. Installation
```bash
# 1. Create a virtual environment
python -m venv venv

# 2. Activate the environment
.\venv\Scripts\activate

# 3. Install core dependencies
pip install langchain-core langchain-community langchain-huggingface langchain-text-splitters 
pip install sentence-transformers faiss-cpu docx2txt streamlit

# 4. Install the CPU-optimized inference engine
pip install llama-cpp-python --extra-index-url [https://abetlen.github.io/llama-cpp-python/whl/cpu](https://abetlen.github.io/llama-cpp-python/whl/cpu)
```

### 3. Model Placement
Download `Phi-3-mini-4k-instruct-q4.gguf` from Hugging Face and place it in the `/models` directory of this project.

---

## 🏃 How to Run

1. **Initialize the Knowledge Base:**
   Process the source documents in `/data` to build the local vector database.
   ```bash
   python code/ingest.py
   ```

2. **Launch the Scholar Dashboard:**
   Start the Streamlit web interface for an interactive querying experience.
   ```bash
   streamlit run code/app.py
   ```

---

## 🧪 Verified Test Cases
The system has been validated against the included Sanskrit corpus:

| Topic | Verified Query Response |
| :--- | :--- |
| **Shankhanada** | Correctly identifies the servant's literalism and the puppy incident. |
| **Kalidasa** | Explains the grammatical debate regarding the verb 'badh' (*badhati* vs *badhate*). |
| **Devotee Lesson** | Cites the moral: "God helps those who help themselves" from the cart story. |

---

## 📈 Performance & Resource Footprint
* **Memory Usage:** ~2.1 GB Peak RAM.
* **Inference Speed:** ~12-18 tokens per second on standard quad-core CPUs.
* **Grounding:** The system utilizes a specialized "Sanskrit Scholar" prompt to minimize hallucinations and stick to the provided text.
