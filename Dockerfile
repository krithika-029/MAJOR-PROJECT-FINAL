# Multi-stage Docker build for Ki-67 Medical Diagnostic System
# Optimized for small image size (<4GB for Railway free tier)

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

# Install system dependencies for OpenCV (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /app

# Copy requirements and install CPU-only PyTorch first (much smaller)
COPY requirements.txt .
RUN pip install --no-cache-dir \
    torch==2.0.0+cpu \
    torchvision==0.15.1+cpu \
    -f https://download.pytorch.org/whl/torch_stable.html \
    && pip cache purge

# Install remaining dependencies without PyTorch (already installed)
RUN pip install --no-cache-dir \
    pytorch-lightning==1.9.5 \
    opencv-python-headless>=4.8.0 \
    Pillow>=10.0.0 \
    scikit-image>=0.21.0 \
    scipy>=1.11.0 \
    numpy>=1.24.0 \
    h5py>=3.9.0 \
    segmentation-models-pytorch>=0.3.3 \
    albumentations>=2.0.0 \
    Flask>=3.0.0 \
    Flask-CORS>=4.0.0 \
    Werkzeug>=3.0.0 \
    python-dotenv>=1.0.0 \
    SQLAlchemy>=2.0.0 \
    reportlab>=4.0.0 \
    && pip cache purge

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

# Health check (simplified, no requests library needed)
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5001/api/health')"

# Run the application
CMD ["python", "-u", "backend/app.py"]
