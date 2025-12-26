import streamlit as st
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA # Fixes the ModuleNotFoundError
from langchain.prompts import PromptTemplate

st.set_page_config(page_title="Department AI Advisor")
st.title("🎓 Department AI Advisor")

# 1. Load Resources (with Error Handling)
@st.cache_resource
def load_resources():
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
        
        # Pull API key from Secrets (Hugging Face / Streamlit Cloud)
        api_key = st.secrets["GROQ_API_KEY"]
        llm = ChatGroq(api_key=api_key, model_name="llama-3.1-8b-instant")
        
        return db, llm
    except Exception as e:
        st.error(f"Error loading resources: {e}")
        return None, None

db, llm = load_resources()

if db and llm:
    # 2. Chain Logic
    template = """You are a helpful University Advisor. Answer based ONLY on the context.
    Context: {context}
    Question: {question}
    Answer:"""
    
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(search_kwargs={"k": 2}),
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )

    # 3. Chat Interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("Ask about the handbook:"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            response = qa_chain.invoke(prompt)
            answer = response["result"]
            st.write(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})