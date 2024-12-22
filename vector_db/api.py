import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import subprocess
from flask import Flask, request, jsonify
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from vector_database import VectorDatabase, VectorDBConfig
from utils import normalize_vector
import torch

# Flask App
app = Flask(__name__)

# Global Flag to Ensure Pipeline Runs Only Once
pipeline_initialized = False

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DOWNLOADER_SCRIPT = os.path.join(BASE_DIR, "../image_downloader/image_downloader.py")
EMBEDDING_GENERATOR_SCRIPT = os.path.join(BASE_DIR, "../embedding_generator/embedding_generator.py")
EMBEDDINGS_FOLDER = os.path.join(BASE_DIR, "../embedding_generator/embeddings")
INDEX_FOLDER = os.path.join(BASE_DIR, "index")
LOG_FOLDER = os.path.join(BASE_DIR, "logs")

# Placeholder for CLIP Model and Processor
model = None
processor = None

def initialize_model():
    """
    Safely initialize the CLIP model and processor.
    """
    global model, processor
    if model is None or processor is None:
        print("Loading CLIP model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        print("CLIP model loaded successfully.")

def run_script(script_path):
    """
    Run a Python script as a subprocess.
    """
    print(f"Running: {script_path}")
    subprocess.run(["python", script_path], check=True)

def generate_text_embedding(text):
    """
    Generate and normalize an embedding for the given text query using the CLIP model.
    """
    initialize_model()
    inputs = processor(text=text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        embedding = model.get_text_features(**inputs).cpu().numpy().flatten()
    return normalize_vector(embedding)

# Initialize Vector Database
config = VectorDBConfig(dimension=512, index_type="flat")
vector_db = VectorDatabase(EMBEDDINGS_FOLDER, INDEX_FOLDER, LOG_FOLDER, config)
vector_db.load_index()

@app.route('/search', methods=['POST'])
def search():
    """
    Search for similar images using a text query.
    """
    data = request.get_json()
    query_text = data.get("query_text")
    k = data.get("k", 5)

    if not query_text:
        return jsonify({"error": "Query text is required"}), 400

    try:
        # Generate text embedding and search the FAISS index
        query_vector = generate_text_embedding(query_text)
        print("Query vector generated with shape:", query_vector.shape)

        # Perform search
        distances, indices = vector_db.index.search(query_vector.reshape(1, -1), k=k)
        print("Raw distances:", distances)
        print("Raw indices:", indices)

        # Process search results
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx != -1:  # Valid result
                similarity = 1 - float(dist)  # Convert L2 distance to cosine similarity
                id_map_entry = vector_db.id_map.get(str(idx))
                if id_map_entry:
                    results.append({
                        "image_name": id_map_entry["image_name"],
                        "url": id_map_entry["url"],
                        "similarity": similarity
                    })

        print("Processed search results:", results)

        return jsonify({"results": results})
    except Exception as e:
        print(f"Error in search: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/test', methods=['GET'])
def test():
    """
    Simple test endpoint to verify API is running.
    """
    return jsonify({"message": "Pipeline initialized and API is ready!"})

@app.route('/health', methods=['GET'])
def health():
    return "OK", 200


if __name__ == "__main__":
        try:
            app.run(host="0.0.0.0", port=5001, debug=True)
        except Exception as e:
            print(f"Failed to start API: {e}")