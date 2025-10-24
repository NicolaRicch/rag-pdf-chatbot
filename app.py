import streamlit as st
import os
from chatbot_pdf_multi import update_vectorstore, ask_question, DATA_DIR, DB_DIR
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.memory import ConversationBufferMemory

# --- Streamlit App ---
st.set_page_config(page_title="RAG Multi-PDF Chatbot", layout="wide")
st.title("📚 Chatbot RAG su più PDF con Gemma")

# --- Sidebar hamburger ---
if 'sidebar_visible' not in st.session_state:
    st.session_state.sidebar_visible = False

def toggle_sidebar():
    st.session_state.sidebar_visible = not st.session_state.sidebar_visible

hamburger_button = st.button("☰", on_click=toggle_sidebar)

if st.session_state.sidebar_visible:
    with st.sidebar:
        st.header("File PDF caricati")

        # Lista file in DATA_DIR
        pdf_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith('.pdf')]

        if pdf_files:
            embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
            vectordb = Chroma(persist_directory=DB_DIR, embedding_function=embedding)
            
            for pdf in pdf_files:
                cols = st.columns([3, 1])
                cols[0].write(pdf)
                # Bottone rimuovi
                if cols[1].button("Rimuovi", key=f"remove_{pdf}"):
                    # Rimuovo file
                    os.remove(os.path.join(DATA_DIR, pdf))
                    st.success(f"File '{pdf}' rimosso dal filesystem.")
                    
                    # Rimuovo dal DB Chroma i documenti con source == pdf
                    metadatas = vectordb.get()['metadatas']
                    # Ricostruiamo lista senza i documenti di quel pdf
                    new_docs = []
                    new_metadatas = []
                    # Carichiamo tutti i doc
                    # Purtroppo Chroma non ha una funzione diretta per rimuovere singoli documenti,
                    # quindi dobbiamo ricostruire db (soluzione semplice: cancellare DB e ricreare con i file rimanenti)

                    # Quindi cancelliamo DB e reindicizziamo solo i pdf rimanenti:
                    vectordb.delete_collection()  # cancella tutto DB

                    for remaining_pdf in pdf_files:
                        if remaining_pdf != pdf and os.path.exists(os.path.join(DATA_DIR, remaining_pdf)):
                            update_vectorstore(os.path.join(DATA_DIR, remaining_pdf))

                    st.rerun()

        else:
            st.write("Nessun PDF caricato.")

# --- Main content ---

uploaded_file = st.file_uploader("Carica un PDF", type="pdf")

if uploaded_file:
    file_path = os.path.join(DATA_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())
    st.success(f"File '{uploaded_file.name}' caricato.")

    with st.spinner("Aggiornamento database..."):
        update_vectorstore(file_path)
    st.success("Database aggiornato!")

user_question = st.text_input("Fai una domanda sui tuoi documenti:")

# Inizializza la memoria se non esiste già
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )



if st.button("Chiedi") and user_question:
    with st.spinner("Cerco nei documenti..."):
        try:
            answer, sources, st.session_state.memory = ask_question(
                user_question,
                memory=st.session_state.memory
            )
            st.markdown(f"**Risposta:** {answer}")
            st.markdown(f"**Fonti:** {', '.join(set(sources))}")

            # Mostra cronologia chat (solo se esiste) --> opzionale
            # if st.session_state.memory.chat_memory.messages:
            #     with st.expander("🧠 Cronologia conversazione"):
            #         for m in st.session_state.memory.chat_memory.messages:
            #             role = "👤 Utente" if m.type == "human" else "🤖 Assistente"
            #             st.write(f"**{role}:** {m.content}")

        except Exception as e:
            st.error(f"Errore: {e}")

