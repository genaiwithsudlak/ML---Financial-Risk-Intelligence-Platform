# Base Image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies (if any needed for lightgbm/xgboost)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ src/
# Copy data (Note: In production, data usually comes from a mounted volume or S3)
# COPY data/ data/  <-- REMOVED: Data is not in git. Use volumes in prod.
# Copy mlruns (for MVP model loading)
COPY mlruns/ mlruns/

# Expose API port
EXPOSE 8000

# Run the API
# Reload is for dev; remove in prod
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
