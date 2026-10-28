FROM python:3.11-slim

LABEL maintainer="AI-in-Production Capstone"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create directories
RUN mkdir -p data models logs eda_outputs

# Generate data + run EDA + train models at build time
RUN python data_ingestion.py
RUN python eda.py
RUN python model.py

# Run tests to verify the build
RUN python run_tests.py

# Expose the API port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/')" || exit 1

# Start the API server
CMD ["python", "app.py"]
