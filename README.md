# 📚 RAG Multi-PDF Chatbot con memoria

Questo progetto è un chatbot basato su LangChain, Ollama, Streamlit e Chroma, in grado di:

- Caricare e indicizzare più file PDF
- Rispondere alle domande usando un approccio RAG (Retrieval-Augmented Generation)
- Ricordare le conversazioni precedenti grazie a una memoria conversazionale

# ⚙️ Installazione

```bash
git clone https://github.com/NicolaRicch/rag-pdf-chatbot.git
cd rag-pdf-chatbot
pip install -r requirements.txt

# ▶️ Esecuzione locale
streamlit run app.py