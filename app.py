import streamlit as st
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq # NEW: Use Groq instead of Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# 1. Page Config
st.set_page_config(page_title="Department AI Advisor", page_icon="🎓")
st.title("🎓 Department AI Advisor")

# 2. Setup Memory (Chat History)
if "messages" not in st.session_state:
    st.session_state.messages = []

# 3. Load Resources (Optimized for Cloud)
@st.cache_resource
def load_resources():
    # We use the same embeddings so the 'chroma_db' still works!
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    
    # Use Groq API instead of local Ollama
    # API key should be stored in .streamlit/secrets.toml
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except (FileNotFoundError, KeyError):
        api_key = "your_actual_api_key_here" 
    llm = ChatGroq(api_key=api_key, model_name="llama-3.1-8b-instant")
    return db, llm

db, llm = load_resources()

# 4. Define the Chain (Same logic as before)
template = """You are a helpful University Advisor. Use the handbook context to answer.
Context: {context}
Question: {question}
Answer:"""

QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)

# 5. The Chat Interface
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Ask me about the handbook:"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        response = qa_chain.invoke(prompt)
        answer = response["result"]
        st.write(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})