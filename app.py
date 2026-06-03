from flask import Flask, request, render_template, jsonify
import os
import fitz
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import requests
import json
import threading

from utils.clean_text import extract_clean_sentences
from utils.model_store import save_model, update_model, load_model_data, list_models, delete_model

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MODELS_FOLDER"] = "saved_models"

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "gemma4"

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

_session = {
    "chunks": [],
    "embeddings": [],
    "index": None,
    "top_k": 5,
    "source_files": [],
    "model_id": None,
}


def _rebuild_index():
    if not _session["embeddings"]:
        _session["index"] = None
        return
    k = min(_session["top_k"], len(_session["embeddings"]))
    idx = NearestNeighbors(n_neighbors=k, metric="cosine")
    idx.fit(np.array(_session["embeddings"]))
    _session["index"] = idx


def _check_ollama():
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def _check_model():
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        if r.status_code == 200:
            models = r.json().get("models", [])
            return any(m["name"].startswith(OLLAMA_MODEL) for m in models)
    except Exception:
        pass
    return False


def _pull_model():
    print(f"\nPulling {OLLAMA_MODEL} from Ollama...")
    try:
        r = requests.post(
            f"{OLLAMA_BASE_URL}/api/pull",
            json={"name": OLLAMA_MODEL},
            stream=True,
            timeout=600,
        )
        for line in r.iter_lines():
            if line:
                d = json.loads(line)
                status = d.get("status", "")
                total = d.get("total", 0)
                completed = d.get("completed", 0)
                if total:
                    pct = completed / total * 100
                    print(f"\r  {status}: {pct:.1f}%", end="", flush=True)
                else:
                    print(f"\r  {status}   ", end="", flush=True)
        print(f"\n{OLLAMA_MODEL} ready.")
    except Exception as e:
        print(f"\nFailed to pull {OLLAMA_MODEL}: {e}")


def _startup():
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    os.makedirs(app.config["MODELS_FOLDER"], exist_ok=True)

    if not _check_ollama():
        print("WARNING: Ollama is not running. Start Ollama before using the chat feature.")
        return

    if not _check_model():
        _pull_model()
    else:
        print(f"Ollama ready — {OLLAMA_MODEL} available.")


threading.Thread(target=_startup, daemon=True).start()


def _extract_text(pdf_path):
    parts = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            parts.append(page.get_text())
    return " ".join(parts)


def _auto_save():
    model_id = _session["model_id"]
    name = _session["source_files"][0] if _session["source_files"] else "Unnamed"

    if model_id:
        update_model(
            app.config["MODELS_FOLDER"],
            model_id,
            _session["chunks"],
            _session["embeddings"],
            _session["source_files"],
        )
    else:
        model_id = save_model(
            app.config["MODELS_FOLDER"],
            name,
            _session["chunks"],
            _session["embeddings"],
            _session["source_files"],
        )
        _session["model_id"] = model_id

    return model_id


@app.route("/")
def dashboard():
    return render_template("dashboard.html")


@app.route("/api/status")
def api_status():
    running = _check_ollama()
    model_ready = _check_model() if running else False
    return jsonify({
        "ollama_running": running,
        "model_ready": model_ready,
        "model_name": OLLAMA_MODEL,
        "chunks_loaded": len(_session["chunks"]),
        "source_files": _session["source_files"],
        "top_k": _session["top_k"],
        "model_id": _session["model_id"],
    })


@app.route("/api/upload", methods=["POST"])
def api_upload():
    if "pdf" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    pdf_file = request.files["pdf"]
    if not pdf_file.filename:
        return jsonify({"error": "No file selected"}), 400

    filename = pdf_file.filename
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    pdf_file.save(file_path)

    text = _extract_text(file_path)
    chunks = extract_clean_sentences(text)

    for chunk in chunks:
        _session["chunks"].append(chunk)
        _session["embeddings"].append(embedding_model.encode(chunk).tolist())

    if filename not in _session["source_files"]:
        _session["source_files"].append(filename)

    _rebuild_index()
    model_id = _auto_save()

    return jsonify({
        "success": True,
        "filename": filename,
        "chunks_added": len(chunks),
        "total_chunks": len(_session["chunks"]),
        "model_id": model_id,
    })


@app.route("/api/ask", methods=["POST"])
def api_ask():
    data = request.get_json()
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "No question provided"}), 400
    if _session["index"] is None:
        return jsonify({"error": "No documents loaded. Upload a PDF first."}), 400
    if not _check_ollama():
        return jsonify({"error": "Ollama is not running. Start Ollama and try again."}), 503

    q_emb = embedding_model.encode(question).reshape(1, -1)
    distances, indices = _session["index"].kneighbors(q_emb)
    top_chunks = [_session["chunks"][i] for i in indices[0]]
    context = " ".join(top_chunks)

    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"

    try:
        r = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            },
            timeout=120,
        )
        answer = r.json()["message"]["content"]
    except Exception as e:
        return jsonify({"error": f"Ollama error: {str(e)}"}), 500

    return jsonify({"answer": answer, "chunks_used": top_chunks})


@app.route("/api/test_chunks", methods=["POST"])
def api_test_chunks():
    data = request.get_json()
    query = data.get("query", "").strip()

    if not query:
        return jsonify({"error": "No query provided"}), 400
    if _session["index"] is None:
        return jsonify({"error": "No documents loaded. Upload a PDF first."}), 400

    q_emb = embedding_model.encode(query).reshape(1, -1)
    distances, indices = _session["index"].kneighbors(q_emb)

    results = []
    for rank, (idx, dist) in enumerate(zip(indices[0], distances[0]), start=1):
        results.append({
            "rank": rank,
            "chunk": _session["chunks"][idx],
            "similarity": round(float(1 - dist), 4),
        })

    return jsonify({"results": results, "query": query})


@app.route("/api/settings", methods=["POST"])
def api_settings():
    data = request.get_json()
    if "top_k" in data:
        k = max(1, min(20, int(data["top_k"])))
        _session["top_k"] = k
        _rebuild_index()
    return jsonify({"top_k": _session["top_k"]})


@app.route("/api/models", methods=["GET"])
def api_list_models():
    models = list_models(app.config["MODELS_FOLDER"])
    return jsonify({"models": models})


@app.route("/api/models/<model_id>/load", methods=["POST"])
def api_load_model(model_id):
    data = load_model_data(app.config["MODELS_FOLDER"], model_id)
    if not data:
        return jsonify({"error": "Model not found"}), 404

    _session["chunks"] = data["chunks"]
    _session["embeddings"] = data["embeddings"]
    _session["source_files"] = data["metadata"]["source_files"]
    _session["model_id"] = model_id
    _rebuild_index()

    return jsonify({
        "success": True,
        "model_id": model_id,
        "chunk_count": len(_session["chunks"]),
        "source_files": _session["source_files"],
    })


@app.route("/api/models/<model_id>", methods=["DELETE"])
def api_delete_model(model_id):
    success = delete_model(app.config["MODELS_FOLDER"], model_id)
    if _session["model_id"] == model_id:
        _session["chunks"] = []
        _session["embeddings"] = []
        _session["index"] = None
        _session["source_files"] = []
        _session["model_id"] = None
    return jsonify({"success": success})


@app.route("/api/session/clear", methods=["POST"])
def api_clear_session():
    _session["chunks"] = []
    _session["embeddings"] = []
    _session["index"] = None
    _session["source_files"] = []
    _session["model_id"] = None
    return jsonify({"success": True})


if __name__ == "__main__":
    app.run(debug=True)
