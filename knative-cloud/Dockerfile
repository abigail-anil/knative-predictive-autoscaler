FROM tensorflow/tensorflow:2.19.0

ENV PYTHONUNBUFFERED=1
ENV PROPHET_NO_COMPILE=1
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Only the minimal system packages
RUN apt-get update && apt-get install -y \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for cache
COPY api/requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy API
COPY api/ /app/api/

# Copy trained models
COPY models/ /app/models/

# Optional: sanity check
RUN ls -lh /app/models/* || true

EXPOSE 8080

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080", "--timeout-keep-alive", "75", "--workers", "1"]
