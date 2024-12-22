import os
import json
import faiss
import numpy as np
import logging
import sys

# Add the project root directory to sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(BASE_DIR)

from utils import ensure_folder_exists, load_json, save_json, load_npy, normalize_vector

class VectorDBConfig:
    def __init__(self, dimension=512, index_type="flat", batch_size=100):
        self.dimension = dimension
        self.index_type = index_type
        self.batch_size = batch_size


class VectorDatabase:
    def __init__(self, embeddings_folder: str, index_folder: str, log_folder: str, config: VectorDBConfig):
        self.embeddings_folder = embeddings_folder
        self.index_folder = index_folder
        self.log_folder = log_folder
        self.config = config
        self.index = None
        self.id_map = {}
        self.url_map = {}

        ensure_folder_exists(log_folder)
        logging.basicConfig(
            filename=os.path.join(log_folder, "vector_database.log"),
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )
        logging.info("VectorDatabase initialized.")

    def load_url_map(self):
        url_map_path = os.path.join(self.embeddings_folder, "../image_downloader/image_url_map.json")
        if os.path.exists(url_map_path):
            self.url_map = load_json(url_map_path)
            logging.info(f"Loaded URL map from {url_map_path}")
        else:
            logging.warning(f"URL map not found at {url_map_path}")

    def load_embeddings(self):
        embeddings = []
        filenames = []

        for file in os.listdir(self.embeddings_folder):
            if file.endswith(".npy"):
                file_path = os.path.join(self.embeddings_folder, file)
                try:
                    embedding = load_npy(file_path)
                    if embedding.shape[0] == self.config.dimension:
                        embeddings.append(embedding)
                        filenames.append(file)
                    else:
                        logging.warning(f"Skipping {file}: Dimension mismatch {embedding.shape[0]}")
                except Exception as e:
                    logging.error(f"Error loading {file}: {str(e)}")
        return np.vstack(embeddings), filenames

    def build_index(self):
        """
        Build a FAISS index from embeddings in the embeddings folder.
        """
        try:
            # Check if the embeddings folder and files exist
            if not os.path.exists(self.embeddings_folder):
                raise FileNotFoundError(f"Embeddings folder not found: {self.embeddings_folder}")
            if not os.listdir(self.embeddings_folder):
                raise FileNotFoundError(f"No embeddings found in folder: {self.embeddings_folder}")

            logging.info(f"Building FAISS index from embeddings in: {self.embeddings_folder}")

            # Load embeddings from .npy files only
            embeddings = []
            file_names = []
            for file_name in os.listdir(self.embeddings_folder):
                file_path = os.path.join(self.embeddings_folder, file_name)
                logging.debug(f"Processing file: {file_path}")

                if file_name.endswith(".npy") and os.path.isfile(file_path):
                    try:
                        # Debug: Log before loading
                        logging.debug(f"Loading embedding: {file_path}")
                        embedding = np.load(file_path, allow_pickle=False)  # Ensure safety
                        embeddings.append(embedding)
                        file_names.append(file_path)
                    except Exception as e:
                        logging.warning(f"Failed to load {file_name}: {e}")

            # Check if any embeddings were loaded
            if not embeddings:
                raise ValueError("No valid embeddings (.npy) files found in the folder.")

            # Stack all embeddings into a matrix
            embeddings_matrix = np.vstack(embeddings)
            self.index = faiss.IndexFlatL2(self.config.dimension)
            self.index.add(embeddings_matrix)

            # Save the index
            ensure_folder_exists(self.index_folder)
            index_path = os.path.join(self.index_folder, "vector_index.faiss")
            faiss.write_index(self.index, index_path)

            # Save ID map
            id_map = {}
            for i, file_name in enumerate(os.listdir(self.embeddings_folder)):
                if file_name.endswith(".npy"):
                    # Derive image name and URL from the .npy file
                    image_name = file_name.replace(".npy", ".jpg")  # Assuming original images are .jpg
                    image_url = f"http://example.com/images/{image_name}"  # Replace with your actual image URL base
                    id_map[i] = {"image_name": image_name, "url": image_url}

            id_map_path = os.path.join(self.index_folder, "id_map.json")
            save_json(id_map, id_map_path)

            logging.info(f"FAISS index built and saved to: {index_path}")
            logging.info(f"ID map saved to: {id_map_path}")
        except Exception as e:
            logging.error(f"Failed to build index: {str(e)}")
            raise





    def load_index(self):
        """
        Load the FAISS index and ID map from the specified folder.
        """
        try:
            # Define paths to the FAISS index and ID map
            index_path = os.path.join(self.index_folder, "vector_index.faiss")
            id_map_path = os.path.join(self.index_folder, "id_map.json")

            # Check for missing files
            if not os.path.exists(index_path):
                raise FileNotFoundError(f"FAISS index file not found: {index_path}")
            if not os.path.exists(id_map_path):
                raise FileNotFoundError(f"ID map file not found: {id_map_path}")

            # Load the FAISS index and ID map
            self.index = faiss.read_index(index_path)
            self.id_map = load_json(id_map_path)

            # Log success and details
            logging.info(f"FAISS index loaded successfully from: {index_path}")
            logging.info(f"ID map loaded successfully from: {id_map_path}")
            logging.info(f"Number of embeddings in index: {self.index.ntotal}")

        except FileNotFoundError as fnf_error:
            logging.error(f"FileNotFoundError: {fnf_error}")
            raise
        except Exception as e:
            logging.error(f"An error occurred while loading the FAISS index: {str(e)}")
            raise


    def search(self, query_vector: np.ndarray, top_k=5, threshold=0.5):
        if self.index is None:
            raise ValueError("Index not loaded. Call load_index() first.")

        query_vector = normalize_vector(query_vector)
        distances, indices = self.index.search(query_vector.reshape(1, -1), top_k)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx == -1 or dist > threshold:
                continue

            filename = self.id_map.get(idx, "Unknown")
            url = self.url_map.get(filename.replace(".npy", ""), "URL not found")
            results.append({
                "filename": filename,
                "similarity": dist,
                "url": url
            })
        return results


def initialize_database():
    logging.basicConfig(level=logging.INFO)
    try:
        # Example configuration for running the script
        INDEX_FOLDER = os.path.join(BASE_DIR, "index")
        EMBEDDINGS_FOLDER = os.path.join(BASE_DIR,"embeddings")
        print(f"Looking for embeddings in: {EMBEDDINGS_FOLDER}")
        LOG_FOLDER = os.path.join(BASE_DIR, "logs")

        # Initialize vector database
        config = VectorDBConfig(dimension=512, index_type="flat")
        vector_db = VectorDatabase(embeddings_folder=EMBEDDINGS_FOLDER,
                                   index_folder=INDEX_FOLDER,
                                   log_folder=LOG_FOLDER,
                                   config=config)

        # Build and load the FAISS index
        vector_db.build_index()
        vector_db.load_index()

        print("FAISS index built and loaded successfully!")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
