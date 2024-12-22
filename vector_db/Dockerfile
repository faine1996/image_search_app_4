# Base image with Python
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the files
COPY api.py /app/api.py
COPY embedding_generator.py /app/embedding_generator.py
COPY main.py /app/main.py
COPY vector_database.py /app/vector_database.py
COPY utils.py /app/utils.py

# Create output directory
RUN mkdir -p /app/images
RUN mkdir -p /app/embeddings
RUN mkdir -p /app/index

# Command to run the downloader script
CMD ["bash", "-c", "python main.py && python api.py"]
