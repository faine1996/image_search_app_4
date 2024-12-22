from embedding_generator import run_embedder
from vector_database import initialize_database


if __name__ == "__main__":
    try:

        # Step 4: Generate Embeddings
        print("Starting Embedding Generator...")
        run_embedder()
        print("Embedding Generator finished.")

        # Step 5: Build Vector Database
        print("Building Vector Database...")
        initialize_database()
        print("Vector Database built.")

        print("All services are initialized.")
    except Exception as e:
        print(f"Pipeline initialization failed: {e}")
