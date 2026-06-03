import os
import json
import shutil
import uuid
import numpy as np
from datetime import datetime


def save_model(models_dir, name, chunks, embeddings, source_files):
    model_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]
    model_dir = os.path.join(models_dir, model_id)
    os.makedirs(model_dir, exist_ok=True)

    metadata = {
        "id": model_id,
        "name": name,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "source_files": source_files,
        "chunk_count": len(chunks),
    }

    with open(os.path.join(model_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    with open(os.path.join(model_dir, "chunks.json"), "w") as f:
        json.dump(chunks, f)

    np.save(os.path.join(model_dir, "embeddings.npy"), np.array(embeddings))

    return model_id


def update_model(models_dir, model_id, chunks, embeddings, source_files):
    model_dir = os.path.join(models_dir, model_id)
    if not os.path.exists(model_dir):
        return False

    meta_path = os.path.join(model_dir, "metadata.json")
    with open(meta_path, "r") as f:
        metadata = json.load(f)

    metadata["updated_at"] = datetime.now().isoformat()
    metadata["source_files"] = source_files
    metadata["chunk_count"] = len(chunks)

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    with open(os.path.join(model_dir, "chunks.json"), "w") as f:
        json.dump(chunks, f)

    np.save(os.path.join(model_dir, "embeddings.npy"), np.array(embeddings))

    return True


def load_model_data(models_dir, model_id):
    model_dir = os.path.join(models_dir, model_id)
    if not os.path.exists(model_dir):
        return None

    with open(os.path.join(model_dir, "metadata.json"), "r") as f:
        metadata = json.load(f)

    with open(os.path.join(model_dir, "chunks.json"), "r") as f:
        chunks = json.load(f)

    embeddings = np.load(os.path.join(model_dir, "embeddings.npy")).tolist()

    return {"metadata": metadata, "chunks": chunks, "embeddings": embeddings}


def list_models(models_dir):
    if not os.path.exists(models_dir):
        return []

    models = []
    for model_id in os.listdir(models_dir):
        meta_path = os.path.join(models_dir, model_id, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                models.append(json.load(f))

    models.sort(key=lambda m: m.get("updated_at", ""), reverse=True)
    return models


def delete_model(models_dir, model_id):
    model_dir = os.path.join(models_dir, model_id)
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
        return True
    return False
