# RAG Dashboard

A local, open-source Retrieval-Augmented Generation (RAG) tool. Upload PDFs, build a vector database from them, and chat with your documents — all running on your own machine with no API keys required.

---

## How it works

1. Upload one or more PDFs — text is extracted, chunked, and embedded into a local vector store.
2. When you ask a question, the most relevant chunks are retrieved and sent as context to a locally-running LLM via Ollama.
3. The LLM answers using only your document content as context.

---

## Software Requirements

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.11 | Other 3.x versions may work but are untested |
| Ollama | Latest | [https://ollama.com](https://ollama.com) |
| gemma4 model | Latest | Auto-downloaded on first run if Ollama is running |

> **Ollama must be installed and running** before starting the app. On first launch, the app will automatically pull the `gemma4` model if it is not already installed (this may take a few minutes depending on your connection).

---

## Installation

### 1. Install Ollama

Download and install Ollama from [https://ollama.com](https://ollama.com) for your operating system.

After installing, start the Ollama application. You can verify it is running by opening a terminal and running:

```
ollama list
```

### 2. Clone or download this repository

```bash
git clone <repository-url>
cd rag-dashboard
```

Or download the ZIP and extract it.

### 3. Create a virtual environment

**Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

If you get an execution policy error, run this first:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

This installs Flask, sentence-transformers, scikit-learn, PyMuPDF, and all other required packages. Note that `sentence-transformers` will download the `all-MiniLM-L6-v2` embedding model on first use (~90 MB).

---

## Running the App

Make sure Ollama is running, then:

```bash
python app.py
```

Open your browser and navigate to:

```
http://localhost:5000
```

On first run the app will check if the `gemma4` model is available in Ollama and pull it automatically if not. Pull progress is shown in the terminal. The model is approximately 5 GB.

---

## Using the Dashboard

### Sidebar
- **Ollama status** — shows whether Ollama is running and the model is ready.
- **Saved Models** — lists all previously saved vector stores. Click any model to load it into the active session.
- **New Session** — clears the current in-memory session (saved models on disk are not affected).

### Documents panel
- Drop a PDF onto the upload zone or click to browse.
- Each upload extracts text, builds embeddings, and saves the model automatically.
- Uploading multiple PDFs in one session adds them all to the same model.

### Vector DB Tester panel
- Use the slider to control how many chunks are retrieved per query (1–20). This setting also applies to the chat.
- Enter any query and click **Test** to see the raw chunks that would be passed to the LLM, along with their similarity scores.

### Chat panel
- Ask questions in plain English about your uploaded documents.
- The LLM answers based on the retrieved context chunks only.

---

## Project Structure

```
.
├── app.py                  # Flask application and API routes
├── requirements.txt
├── uploads/                # Uploaded PDFs (created automatically)
├── saved_models/           # Persisted vector stores (created automatically)
│   └── <model-id>/
│       ├── metadata.json
│       ├── chunks.json
│       └── embeddings.npy
├── templates/
│   └── dashboard.html      # Single-page dashboard UI
└── utils/
    ├── clean_text.py       # Text extraction and chunking
    └── model_store.py      # Save / load / delete vector stores
```

---

## Notes

- All data stays on your machine. No data is sent to any external server.
- The embedding model (`all-MiniLM-L6-v2`) runs locally via `sentence-transformers`.
- Saved models persist between sessions and are stored in `saved_models/`.
- To use a different Ollama model, change `OLLAMA_MODEL` at the top of `app.py`.
