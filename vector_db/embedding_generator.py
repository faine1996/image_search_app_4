import os
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import logging
import numpy as np
from typing import List

class EmbeddingGenerator:
    def __init__(self, image_folder: str, output_folder: str, log_folder: str):
        self.image_folder = image_folder
        self.output_folder = output_folder
        self.log_folder = log_folder
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        # Setup Logging
        os.makedirs(self.output_folder, exist_ok=True)
        os.makedirs(self.log_folder, exist_ok=True)
        logging.basicConfig(
            filename=os.path.join(self.log_folder, "embedding_generator.log"),
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )
        console_handler = logging.StreamHandler()
        logging.getLogger().addHandler(console_handler)

        # Add this line to confirm where embeddings will be saved
        print(f"Embeddings will be saved to: {self.output_folder}")

    def generate_embedding(self, image_path: str) -> np.ndarray:
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)

            with torch.no_grad():
                embedding = self.model.get_image_features(**inputs)
                embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
            return embedding.cpu().numpy().flatten()
        except Exception as e:
            logging.error(f"Error processing {image_path}: {e}")
            return None

    def process_images(self):
        image_files = [
            f for f in os.listdir(self.image_folder)
            if f.endswith((".jpg", ".jpeg", ".png"))
        ]
        logging.info(f"Found {len(image_files)} images to process.")

        for image_file in image_files:
            image_path = os.path.join(self.image_folder, image_file)
            output_file = os.path.join(self.output_folder, f"{os.path.splitext(image_file)[0]}.npy")

            if os.path.exists(output_file):
                logging.info(f"Skipping already processed image: {image_file}")
                continue

            embedding = self.generate_embedding(image_path)
            if embedding is not None:
                np.save(output_file, embedding)
                logging.info(f"Processed and saved embedding: {image_file}")
            else:
                logging.error(f"Failed to process image: {image_file}")

            # Free GPU memory after processing
            torch.cuda.empty_cache()
        print("Total embeddings generated:", len(os.listdir(self.output_folder)))
def run_embedder():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    IMAGE_FOLDER = os.path.join(BASE_DIR, "./images")
    OUTPUT_FOLDER = os.path.join(BASE_DIR, "./embeddings")
    LOG_FOLDER = os.path.join(BASE_DIR, "logs")

    print("Image Folder Path:", IMAGE_FOLDER)  # Debug the image path

    generator = EmbeddingGenerator(IMAGE_FOLDER, OUTPUT_FOLDER, LOG_FOLDER)
    generator.process_images()
