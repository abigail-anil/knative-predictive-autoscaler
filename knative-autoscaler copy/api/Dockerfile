FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for Prophet and TensorFlow
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY api/requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY api/app.py /app/

# Create models directory (will be populated during build or at runtime)
RUN mkdir -p /app/models/prophet /app/models/lstm /app/models/hybrid

# Copy pre-trained models from models directory
COPY models/ /app/models/

# Expose port
EXPOSE 8080

# Set environment variables
ENV MODEL_DIR=/app/models
ENV PORT=8080
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8080/health', timeout=5)"

# Run with gunicorn for production
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "2", "--timeout", "120", "--preload", "app:app"]