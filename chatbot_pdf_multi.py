import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain


# Percorsi
DATA_DIR = "data"
DB_DIR = "chroma_db"

# Assicuriamoci che le cartelle esistano
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

#1 Carica e spezza un PDF
def load_pdf_and_split(path: str):
    loader = PyMuPDFLoader(path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    # aggiungiamo un metadato per sapere da quale file proviene
    for c in chunks:
        c.metadata["source"] = os.path.basename(path)
    return chunks

#2 Crea o aggiorna il database Chroma
def update_vectorstore(pdf_path):
    embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # carica o crea un database esistente
    vectordb = Chroma(persist_directory=DB_DIR, embedding_function=embedding)

    # controlla se il file è già stato indicizzato
    existing_sources = {meta.get("source") for meta in vectordb.get()["metadatas"] if meta.get("source")}
    filename = os.path.basename(pdf_path)

    if filename in existing_sources:
        print(f"✅ '{filename}' già indicizzato, salto l'importazione.")
        return vectordb

    # altrimenti carica e aggiungi nuovi documenti
    print(f"📥 Indicizzo nuovo file: {filename}")
    new_docs = load_pdf_and_split(pdf_path)
    vectordb.add_documents(new_docs)
    vectordb.persist()

    print(f"✅ Aggiunto '{filename}' al database.")
    return vectordb

#3 Prompt e modello
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Sei un assistente virtuale professionale e chiaro.
Rispondi solo usando le informazioni contenute nei documenti seguenti.
Se non sai la risposta, scrivi: "Non ho abbastanza informazioni nei documenti per rispondere."

CONTENUTO:
{context}

DOMANDA:
{question}

RISPOSTA:
"""
)

llm = Ollama(model="gemma:2b")


#4 Crea catena RAG
def create_rag_chain_with_memory(vectordb, memory=None):
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # Crea memoria se non esiste
    if memory is None:
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

    # Crea catena conversazionale RAG
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt_template},
        return_source_documents=True,
        output_key="answer"
    )

    return chain, memory



#5 Funzione principale per domande
def ask_question(question: str, memory=None):
    embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma(persist_directory=DB_DIR, embedding_function=embedding)

    chain, memory = create_rag_chain_with_memory(vectordb, memory)
    result = chain.invoke({"question": question})

    answer = result["answer"]
    sources = [doc.metadata.get("source") for doc in result.get("source_documents", [])]
    return answer, sources, memory


