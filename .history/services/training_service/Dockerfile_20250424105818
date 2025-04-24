FROM pytorch/pytorch:2.6.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the service code
COPY src/ ./src/
COPY models/ ./models/

# Create necessary directories
RUN mkdir -p /app/models /app/models/trained

# Set environment variables
ENV PYTHONPATH=/app
ENV MODEL_SAVE_PATH=/app/models/trained

# Expose port for the service
EXPOSE 8000

# Start the FastAPI application
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"] 