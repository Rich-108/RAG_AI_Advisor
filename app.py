
import streamlit as st
import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- 1. PAGE CONFIG (MUST BE THE FIRST ST COMMAND) ---
st.set_page_config(page_title="Department AI Advisor", layout="centered")

# --- 2. THE VISIBLE TITLE ---
st.title("🎓 Acadamic Insight Engine")
st.markdown("---")

# --- 3. PATH LOGIC FOR HUGGING FACE ---
current_dir = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(current_dir, "chroma_db")

# --- 4. RESOURCE LOADING ---
@st.cache_resource
def load_rag_system():
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        # Initialize Chroma with the absolute path
        db = Chroma(persist_directory=db_path, embedding_function=embeddings)
        
        api_key = st.secrets["GROQ_API_KEY"]
        llm = ChatGroq(api_key=api_key, model_name="llama-3.1-8b-instant", temperature=0)
        
        return db, llm
    except Exception as e:
        st.error(f"Initialization Error: {e}")
        return None, None

db, llm = load_rag_system()

# --- 5. RAG LOGIC & CHAT ---
if db and llm:
    # Strict prompt to stop the "Cake Recipe" problem
    prompt = ChatPromptTemplate.from_template("""
    You are a University Advisor. Answer ONLY using the context below.
    If the answer isn't there, say: "I am sorry, but that is not in the handbook."
    
    Context: {context}
    Question: {question}
    Answer:""")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    retriever = db.as_retriever(search_kwargs={"k": 3})
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Chat Interface Logic...
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_input := st.chat_input("Ask me about the handbook..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            # This 'invoke' now uses the RAG data
            response = rag_chain.invoke(user_input)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})