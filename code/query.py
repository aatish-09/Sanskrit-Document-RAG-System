import streamlit as st
import os
import multiprocessing
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- Page Config ---
st.set_page_config(page_title="Sanskrit RAG Scholar", page_icon="📜", layout="wide")

# --- Initialize Paths ---
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.abspath(os.path.join(base_dir, "..", "models", "Phi-3-mini-4k-instruct-q4.gguf"))
index_path = os.path.join(base_dir, "sanskrit_index")

# --- Sidebar: System Status ---
st.sidebar.title("⚙️ System Configuration")
st.sidebar.info("Model: Phi-3-mini (GGUF)")
st.sidebar.info(f"Mode: CPU-Only ({multiprocessing.cpu_count()} Threads)")

# --- Resource Loading (Cached for Speed) ---
@st.cache_resource
def load_resources():
    # Load Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    
    # Load Vector Store
    vector_db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    
    # Load LLM
    llm = LlamaCpp(
        model_path=model_path,
        n_ctx=4096,
        n_threads=multiprocessing.cpu_count(),
        temperature=0.1,
        verbose=False
    )
    return vector_db.as_retriever(search_kwargs={"k": 2}), llm

retriever, llm = load_resources()

# --- RAG Logic ---
template = """<|system|>
You are a Sanskrit Scholar. Use the following Sanskrit context to answer the user's question accurately.
If the answer is not in the context, politely state that the information is missing.
Context: {context} <|end|>
<|user|>
{question} <|end|>
<|assistant|>"""

prompt = PromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# --- UI Layout ---
st.title("📜 Sanskrit Document RAG System")
st.markdown("Query your collection of Sanskrit stories and moral lessons using natural language.")

query = st.text_input("Ask a question (e.g., 'What happened to Shankhanada?'):", placeholder="Enter your query here...")

if query:
    with st.spinner("Scholar is analyzing the documents..."):
        try:
            # Get response
            response = rag_chain.invoke(query)
            
            # Display Result
            st.subheader("Answer:")
            st.write(response)
            
            # Show Retrieved Context (For Transparency/Report)
            with st.expander("🔍 View Retrieved Sources"):
                docs = retriever.invoke(query)
                for i, doc in enumerate(docs):
                    st.markdown(f"**Source {i+1}:**")
                    st.text(doc.page_content)
        except Exception as e:
            st.error(f"Error: {e}")

# --- Footer ---
st.markdown("---")
st.caption("Developed for Sanskrit RAG System Assignment | CPU-Optimized Inference")