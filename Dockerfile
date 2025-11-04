# MNIST CNN Docker Container
# Base image: Python 3.11 slim for smaller image size
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY train_cnn.py .
COPY predict_cnn.py .

# Create directories for data and outputs
RUN mkdir -p /app/data /app/outputs

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default command: train the model
CMD ["python", "train_cnn.py"]
