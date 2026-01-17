import os
import docx2txt
from langchain_core.documents import Document 
# NEW IMPORT PATH FOR 2026
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ... (rest of your load_sanskrit_docs function remains the same)

def load_sanskrit_docs(file_path):
    """Extracts text from docx and splits into story-based documents."""
    # Preserves Unicode characters for Sanskrit [cite: 48-119]
    raw_text = docx2txt.process(file_path)
    
    # Identify stories by their titles from your provided document 
    # We split by titles to keep context together
    titles = ["मूर्खभृत्यस्य", "चतुरस्य कालीदासस्य", "वृद्धायाः चार्तुयम्", "शीतं बहु बाधति"]
    
    docs = []
    # Simple split based on double newlines to start
    paragraphs = raw_text.split('\n\n')
    
    for para in paragraphs:
        if len(para.strip()) > 20:
            docs.append(Document(page_content=para.strip(), metadata={"source": file_path}))
    return docs

if __name__ == "__main__":
    # 1. Setup paths [cite: 39-44]
    data_path = "../data/Rag-docs.docx"
    
    if os.path.exists(data_path):
        print("--- Step 1: Loading Sanskrit Documents ---")
        all_docs = load_sanskrit_docs(data_path)
        
        # 2. Chunking logic (keeping Shlokas intact) [cite: 68-69, 103]
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
        chunks = text_splitter.split_documents(all_docs)
        
        # 3. Embedding Setup (CPU Optimized) [cite: 15, 36]
        print("--- Step 2: Generating Embeddings (This may take a minute) ---")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        
        # 4. Create and Save Vector Store [cite: 18, 27]
        vector_db = FAISS.from_documents(chunks, embeddings)
        vector_db.save_local("sanskrit_index")
        print(f"--- Success! Created 'sanskrit_index' with {len(chunks)} chunks ---")
    else:
        print(f"Error: File not found at {data_path}")