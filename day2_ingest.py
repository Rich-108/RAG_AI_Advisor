from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Load the PDF
print("--- Loading PDF ---")
loader = PyPDFLoader("handbook.pdf")
data = loader.load()

# 2. Split into chunks 
# We use 500 characters so your i3 stays fast during the search phase.
print("--- Splitting text into chunks ---")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, 
    chunk_overlap=50
)
chunks = text_splitter.split_documents(data)

# 3. Show Results
print(f"Total Pages Found: {len(data)}")
print(f"Total Chunks Created: {len(chunks)}")

# Preview the first chunk to see if it read correctly
if chunks:
    print("\n--- Preview of First Chunk ---")
    print(chunks[0].page_content)