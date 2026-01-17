import os
from llama_cpp import Llama

# Use the exact path from your error
model_path = r"E:\RAG_SAN\models\Llama-3.2-1B-Instruct-Q4_K_M.gguf"

print(f"--- Diagnostic Start ---")
print(f"Checking Path: {model_path}")

if not os.path.exists(model_path):
    print("❌ ERROR: File does not exist at this path.")
else:
    size = os.path.getsize(model_path) / (1024*1024)
    print(f"✅ File exists. Size: {size:.2f} MB")
    
    try:
        print("Attempting to load model into RAM (CPU)...")
        # Direct load to see the raw error
        temp_llm = Llama(model_path=model_path, verbose=True)
        print("✅ SUCCESS: Model loaded perfectly!")
    except Exception as e:
        print(f"❌ FAILED: The engine could not read the file.")
        print(f"Actual Error: {e}")