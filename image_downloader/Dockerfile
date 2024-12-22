# Base image with Python
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy the script and requirements
COPY image_downloader.py /app/image_downloader.py
COPY image_urls.txt /app/image_urls.txt
COPY requirements.txt /app/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
# Create output directory
RUN mkdir -p /app/images

# Command to run the downloader script
CMD ["python", "image_downloader.py"]