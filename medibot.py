import os
import time
import streamlit as st
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate


dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=dotenv_path)
DB_FAISS_PATH = "vectorstore/db_faiss"


# ---------------- Cache vectorstore ----------------
@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    db = FAISS.load_local(
        DB_FAISS_PATH,
        embedding_model,
        allow_dangerous_deserialization=True
    )
    return db


# ---------------- Fast typing animation ----------------
def stream_response(text):
    """Simulate fast but smooth typing."""
    message_placeholder = st.empty()
    full_text = ""
    # use larger chunking for speed (instead of word by word)
    for chunk in text.split(". "):
        full_text += chunk + ". "
        message_placeholder.markdown(full_text + "▌")
        time.sleep(0.005)  # much faster
    message_placeholder.markdown(full_text)


# ---------------- Main chatbot ----------------
def main():
    st.title("🩺 AI Medical Chatbot")

    # Sidebar info
    with st.sidebar:
        st.header("⚙️ Model Info")
        st.write("""
        - **LLM:** Llama 3.1 (Groq)
        - **Vector DB:** FAISS
        - **Embeddings:** Sentence Transformers
        - **Architecture:** RAG (Retrieval-Augmented Generation)
        """)
        st.info("💡 Ask any medical-related question based on your PDF knowledge base.")

        # --- Download Chat Button ---
        if st.session_state.get("messages"):
            chat_text = "\n\n".join(
                [f"{m['role'].capitalize()}: {m['content']}" for m in st.session_state.messages]
            )
            st.download_button(
                label="💾 Download Chat (.txt)",
                data=chat_text.encode("utf-8"),
                file_name=f"medical_chat_{time.strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
            )

        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.experimental_rerun()

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Get user input
    user_prompt = st.chat_input("Ask your medical question here...")

    if user_prompt:
        # ✅ Auto rephrase one-word queries like "cancer" → "What is cancer?"
        if len(user_prompt.strip().split()) == 1:
            user_prompt = f"What is {user_prompt.strip()}?"

        with st.chat_message("user"):
            st.markdown(user_prompt)
        st.session_state.messages.append({"role": "user", "content": user_prompt})

        with st.spinner("🧠 Processing your question..."):
            try:
                vectorstore = get_vectorstore()
                if not vectorstore:
                    st.error("⚠️ Vector store not found.")
                    return

                GR0Q_API_KEY = os.environ.get("GR0Q_API_KEY")
                if not GR0Q_API_KEY:
                    st.error("⚠️ Missing GROQ_API_KEY in .env file")
                    return

                # Initialize LLM (faster generation)
                llm = ChatGroq(
                    model="llama-3.1-8b-instant",
                    temperature=0.3,
                    max_tokens=400,  # slightly reduced for speed
                    api_key=GR0Q_API_KEY,
                )

                # Structured medical prompt
                CUSTOM_PROMPT_TEMPLATE = """
                You are a professional AI medical assistant. Use the CONTEXT below 
                to answer clearly and concisely in a **structured** format.

                CONTEXT: {context}
                QUESTION: {question}

                Format:
                - **Diagnosis:** ...
                - **Symptoms:** ...
                - **Treatment:** ...
                - **Advice:** ...

                Keep responses short, factual, and only use information from the context.
                """

                prompt_template = PromptTemplate(
                    template=CUSTOM_PROMPT_TEMPLATE,
                    input_variables=["context", "question"]
                )

                retrieval_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
                combine_docs_chain = create_stuff_documents_chain(llm, retrieval_prompt)
                rag_chain = create_retrieval_chain(
                    vectorstore.as_retriever(search_kwargs={"k": 2}),  # smaller k = faster
                    combine_docs_chain,
                )

                # Run RAG chain
                response = rag_chain.invoke({"input": user_prompt})
                result = response["answer"]

                # Assistant message (fast streaming)
                with st.chat_message("assistant"):
                    stream_response(result)

                st.session_state.messages.append(
                    {"role": "assistant", "content": result}
                )

            except Exception as e:
                st.error(f"⚠️ Error: {str(e)}")


if __name__ == "__main__":
    main()

