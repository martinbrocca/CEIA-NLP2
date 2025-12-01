import streamlit as st
import os
import torch
from dotenv import load_dotenv

# LangChain Imports
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# --- Configuration and Initialization ---

load_dotenv()
# Get API keys from .env file
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Define the device dynamically for embeddings (CUDA, MPS, or CPU)
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

INDEX_NAME = "resume-embeddings-2"
MODEL_NAME = "llama-3.1-8b-instant" # Using the fast, supported Llama 3 8B model

@st.cache_resource
def get_retriever():
    """Initializes embeddings and Pinecone Vector Store, returns the retriever."""
    if not PINECONE_API_KEY:
        st.error("PINECONE_API_KEY not found. Please check your .env file.")
        return None

    # Initialize Embeddings model (using dynamic device)
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': device}
    )
    
    # Initialize the Vector Store (Note: Assumes index is already created and populated)
    vectorstore = PineconeVectorStore(
        index_name=INDEX_NAME, 
        embedding=embeddings, 
        pinecone_api_key=PINECONE_API_KEY
    )
    
    # Define the retriever (k=5 to ensure enough context is retrieved)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    return retriever

@st.cache_resource
def get_conversational_rag_chain(_retriever):
    """Initializes the LLM and creates the complete RAG chain."""
    if not GROQ_API_KEY:
        st.error("GROQ_API_KEY not found. Please check your .env file.")
        return None

    # 1. Instantiate Groq LLM
    llm = ChatGroq(
        model=MODEL_NAME,
        temperature=0,
        max_retries=2,
    )

    # 2. Contextualize Question Prompt (for history)
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, _retriever, contextualize_q_prompt
    )

    # 3. Answer Question Prompt
    qa_system_prompt = (
        "You are a helpful HR assistant analyzing resumes. "
        "Use the following pieces of retrieved context to answer "
        "the question. Always cite which resume (source) the information comes from. "
        "If you don't know the answer, say that you don't know. "
        "Use three sentences maximum and keep the answer concise."
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    # 4. State Management (Using Streamlit session_state for storage)
    if "store" not in st.session_state:
        st.session_state.store = {}
        
    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in st.session_state.store:
            st.session_state.store[session_id] = ChatMessageHistory()
        return st.session_state.store[session_id]

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    return conversational_rag_chain

# --- Streamlit UI Code ---

st.set_page_config(page_title="Resume RAG Analyst (Groq/Pinecone)", layout="wide")
st.title("📄 Resume RAG Analyst")
st.caption(f"Powered by **Groq** ({MODEL_NAME}) and **Pinecone** on **{device.upper()}**")

# Initialize chat history display
if "messages" not in st.session_state:
    st.session_state.messages = []

# Get the RAG chain and ensure everything is set up
retriever = get_retriever()
if retriever is None:
    st.stop()
    
conversational_rag_chain = get_conversational_rag_chain(retriever)
if conversational_rag_chain is None:
    st.stop()
    
# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input processing
if prompt := st.chat_input("Ask a question about the candidate's resume..."):
    # 1. Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Get AI response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing resumes..."):
            # Use a static session_id for simplicity, tied to the current Streamlit session
            session_id = "streamlit_session"
            
            response = conversational_rag_chain.invoke(
                {"input": prompt},
                config={"configurable": {"session_id": session_id}}
            )
            
            ai_response = response["answer"]
            st.markdown(ai_response)
            
    # 3. Add AI response to chat history
    st.session_state.messages.append({"role": "assistant", "content": ai_response})