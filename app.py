import streamlit as st
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate  # <--- Make sure this is imported

st.title("🎓 Department AI Advisor")

@st.cache_resource
def load_resources():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    llm = OllamaLLM(model="llama3.2:1b")
    return db, llm

db, llm = load_resources()

# --- NEW ADDITION START ---
# "Less strict" prompt for the 1B model (it gets scared easily!)
template = """You are a helpful assistant.
Read the following text text and answer the question.

Context:
{context}

Question: {question}

Answer:"""

QA_CHAIN_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=template,
)
# --- NEW ADDITION END ---

# Update your chain to use the prompt
retriever = db.as_retriever(search_kwargs={"k": 2}) # Reduced to 2 for better focus
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT} # <--- Add this!
)

query = st.text_input("Enter your question here:")

if query:
    with st.spinner("Searching..."):
        # This will now use your new strict rules
        response = qa_chain.invoke(query)
        st.markdown("### 🤖 Advisor Answer:")
        st.success(response["result"])

        st.markdown("---")
        # Keep debug accessible but tucked away
        with st.expander("�️ Debug: See what I read from the Handbook"):
            docs = db.similarity_search(query, k=2)
            for i, d in enumerate(docs):
                st.markdown(f"**Chunk {i+1}** (from page {d.metadata.get('page', '?')}):")
                st.info(d.page_content)