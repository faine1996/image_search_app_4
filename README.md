# Scalable Image Search Application

## Overview
This application is a Dockerized image search engine that uses CLIP for generating embeddings, FAISS for vector-based search, and Flask for the API layer. It supports three main functionalities:

1. *Image Downloader*: Downloads images asynchronously from URLs.
2. *Embedding Generator*: Processes downloaded images, generates embeddings using a pre-trained CLIP model, stores embeddings as .npy files, and inserts them into a FAISS-based vector database for efficient querying.
3. *Search Interface*: A Flask-based API to query the FAISS index for similar images based on text input.
4. *Web Application*: A Streamlit-based interface for users to interact with the image search system in a user-friendly way.

## Current Architecture

The application is structured into three Docker services, each responsible for specific tasks:

1. *Image Downloader*:
   - Downloads images using aiohttp.
   - Saves the images locally and generates a URL mapping.

2. *Embedding Generator, Vector Database and Search API*:
   - Processes images to generate embeddings using CLIP.
   - Stores embeddings as .npy files in the shared volume.
   - Inserts embeddings into a FAISS-based vector database for fast similarity searches.
   - Flask-based API to serve search requests.
   - Uses FAISS for vector similarity search.

3. *Web Application*:
   - Streamlit-based front-end interface for searching images.
   - Communicates with the Flask API for search queries.

### Communication
- Docker Compose manages the services.
- Shared volumes facilitate data exchange (e.g., embeddings and index files).
- Internal Docker networking enables communication between services.

## Scaling for Production

To handle millions of images and high traffic, the following architecture enhancements are proposed:

1. *Scalable Infrastructure*: Deploy services in an auto-scaling group to handle varying traffic.

2. *Load Balancing*: Use a load balancer to distribute incoming API traffic among multiple service instances.

3. *Database Optimization*: Replace the local FAISS index with a distributed solution like Pinecone or AWS OpenSearch for faster querying and higher scalability.

4. *Improved Storage*: Store images and embeddings on cloud storage (e.g., AWS S3) for scalability and durability.

5. *Caching*: Add a caching layer for frequently accessed queries to reduce response times.

## Development Notes

### Running Locally
The entire application is managed through Docker Compose, so no additional manual steps are required for setup. To start all services, run:
```bash
docker-compose build
docker-compose up
```


This will:
- Start the backend services (Image Downloader, Embedding Generator, and Search API).
- Launch the Streamlit web application accessible at http://localhost:8501 by default.

### API Endpoints
- /search (POST): Query for similar images.
- /health (GET): Health check endpoint.

### Example Query
json
```bash
curl --location 'http://localhost:5001/search' \
--header 'Content-Type: application/json' \
--data '{"query_text": "a sunset over the mountains", "k": 5}'
```
