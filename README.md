# Chatbot RAG Multi-PDF
Questo progetto implementa un chatbot basato su RAG (Retrieval-Augmented Generation) capace di rispondere a domande sui contenuti di più file PDF. Utilizza il modello Gemma tramite Ollama, embeddings con SentenceTransformer e un database vettoriale Chroma.

## Funzionalità principali:
- Caricamento di uno o più PDF nella cartella `data`.
- Spezzamento dei PDF in chunk di testo per un recupero più preciso.
- Indicizzazione automatica dei chunk nel database Chroma (`chroma_db`).
- Possibilità di visualizzare i PDF caricati e rimuoverli dal menu a scomparsa (sidebar).
- Chat con il modello LLM che utilizza i documenti come riferimento.
- Memoria della conversazione: il chatbot ricorda le domande precedenti e le risposte.

# Come usare il progetto:

## Clonare il repository
git clone https://github.com/NicolaRicch/rag-pdf-chatbot.git
cd chatbot-pdf-rag

## Installare le dipendenze
pip install -r requirements.txt

## Caricare i PDF
Inserisci i PDF nella cartella data/ oppure caricali tramite l’interfaccia Streamlit (se decidi di riattivarla).

## Avviare il chatbot
streamlit run app.py

## Interagire con il bot:
Inserisci la tua domanda nell’apposito campo.

Il bot risponderà usando le informazioni contenute nei PDF.

La risposta include anche le fonti (nome dei PDF).

Puoi controllare e rimuovere i PDF direttamente dal menu a scomparsa.

## Struttura del progetto:
CHATBOT PDF RAG/
├─ chatbot_pdf_multi.py # Logica del chatbot, gestione PDF e database
├─ app.py # Interfaccia Streamlit
├─ data/ # PDF caricati
├─ chroma_db/ # Database Chroma
├─ requirements.txt # Dipendenze
└─ README.md


## Tecnologie utilizzate

- Python 3.10+
- LangChain
- SentenceTransformer Embeddings
- Chroma Vector Store
- Gemma LLM via Ollama
- Streamlit (opzionale)

## Obiettivo del progetto:
Questo progetto è stato sviluppato come esercizio per approfondire tecnologie moderne di NLP, gestione di documenti e chatbot intelligenti. Può essere esteso per integrare altre fonti o modelli LLM.