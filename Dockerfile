# Use official Python 3.10 runtime as parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Set working directory inside the container
WORKDIR /app

# Install basic build tools (useful for python packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy dataset, source modules, and trained model files
COPY spam_ham_dataset.csv /app/
COPY outputs/ /app/outputs/
COPY src/ /app/src/

# Expose the default FastAPI port
EXPOSE 8000

# Container healthcheck using python standard library
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\", 8000)}/health')" || exit 1

# Run the app server on container start
CMD ["python", "src/app.py"]

