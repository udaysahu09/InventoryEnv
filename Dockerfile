FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files
COPY environment.py .
COPY app.py .
COPY openenv.yaml .

# Expose port 7860 (Hugging Face Spaces default)
EXPOSE 7860

# Health check for Hugging Face Spaces
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:7860/', timeout=5).raise_for_status()" || exit 1

# Run the FastAPI application
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
