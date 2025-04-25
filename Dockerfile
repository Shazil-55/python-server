FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY server.py .
COPY config.env .env

# Create volume mount point for model files
RUN mkdir -p /app/models
VOLUME /app/models

# Expose the API port
EXPOSE 8000

# Run the server
CMD ["python", "server.py"] 