# Base image with Python
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the files
COPY streamlit_app.py /app/streamlit_app.py

# Create output directory
RUN mkdir -p /app/images

# Command to run the downloader script
CMD ["bash", "-c", "streamlit run streamlit_app.py"]
