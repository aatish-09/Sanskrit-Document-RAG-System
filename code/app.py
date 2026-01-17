import streamlit as st
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import GPT4All
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Page Configuration
st.set_page_config(page_title="Sanskrit Scholar RAG", layout="wide")
st.title("🪔 Sanskrit Document Intelligence System")

# 1. Paths & Initialization
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "Phi-3-mini-4k-instruct-q4.gguf")
index_path = os.path.join(base_dir, "sanskrit_index")

@st.cache_resource
def load_system():
    # Load Knowledge Base
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    vector_db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    
    # Load GPT4All (The stable Windows CPU Engine)
    llm = GPT4All(model=model_path, device='cpu', n_threads=os.cpu_count())
    return vector_db.as_retriever(search_kwargs={"k": 2}), llm

retriever, llm = load_system()

# 2. RAG Logic
template = """System: You are a Sanskrit Scholar. Answer the question using ONLY the context provided.
Context: {context}
User: {question}
Scholar:"""
prompt = PromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 3. Sidebar for Metadata
with st.sidebar:
    st.header("Document Metadata")
    st.info("Source: Rag-docs.docx")
    st.markdown("### Key Stories Indexed:")
    st.write("- मूर्खभृत्यस्य (The Foolish Servant)")
    st.write("- चतुरस्य कालीदासस्य (Clever Kalidasa)")
    st.write("- वृद्धायाः चार्तुयम् (Old Woman's Trick)")

# 4. User Interaction
user_query = st.text_input("Enter your query in Sanskrit or English:", placeholder="e.g., What did Shankhanada do with the sugar?")

if user_query:
    with st.spinner("Analyzing Sanskrit Corpus..."):
        # Get response and retrieved context
        docs = retriever.invoke(user_query)
        response = rag_chain.invoke(user_query)
        
        st.subheader("Scholar's Response")
        st.write(response)
        
        with st.expander("View Source Context"):
            for i, doc in enumerate(docs):
                st.markdown(f"**Snippet {i+1}:**")
                st.text(doc.page_content)