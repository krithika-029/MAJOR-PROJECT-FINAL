# Multi-stage Docker build for Ki-67 Medical Diagnostic System
# Stage 1: Build React frontend
FROM node:20-alpine AS frontend-builder

WORKDIR /app/frontend

# Copy frontend package files
COPY frontend-react/package*.json ./
RUN npm ci --production=false

# Copy frontend source and build
COPY frontend-react/ ./
RUN npm run build

# Stage 2: Python backend with model
FROM python:3.11-slim

    # Install system dependencies for OpenCV
    RUN apt-get update && apt-get install -y \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgomp1 \
        && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY backend/ ./backend/

# Copy model checkpoint
COPY models/ ./models/

# Copy frontend build from stage 1
COPY --from=frontend-builder /app/frontend/dist ./frontend-react/dist

# Create necessary directories
RUN mkdir -p uploads results data/BCData

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=backend/app.py
ENV PORT=5001

# Expose port
EXPOSE 5001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5001/api/health')"

# Run the application
CMD ["python", "backend/app.py"]
