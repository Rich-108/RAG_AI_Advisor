from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os
# 1. Load and Split (Exactly like Day 2)
loader = PyPDFLoader("handbook.pdf")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = text_splitter.split_documents(docs)

# 2. Setup the "Translator" (The Embedding Model)
# We use 'all-MiniLM-L6-v2' because it is extremely fast on i3 CPUs.
print("--- Downloading small embedding model (approx 80MB)... ---")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 3. Create the Database on your SSD
print("--- Creating Vector Database in './chroma_db' folder... ---")
vector_db = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print(f"SUCCESS! Database created with {len(splits)} chunks.")
print("You should now see a new folder named 'chroma_db' in your sidebar.")