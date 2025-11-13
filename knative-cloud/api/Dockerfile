FROM python:3.11-slim

WORKDIR /app

COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get install -y gcc g++ libgomp1

# Copy the API source code
COPY api /app

# Copy models and data folders
COPY models /app/models
COPY data /app/data

EXPOSE 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
