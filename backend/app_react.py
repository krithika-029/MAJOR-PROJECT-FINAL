"""
Ki-67 Medical Diagnostic System - Flask Backend with React Frontend
==================================================================
Integrated Flask API backend serving React frontend
"""

import os
import io
import csv
import base64
import uuid
import random
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

import cv2
import numpy as np
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pytorch_lightning as pl
from skimage.feature import peak_local_max
from PIL import Image, ImageDraw, ImageFont
import h5py
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
import json

# ============================================================================
# MODEL CLASSES (from training script)
# ============================================================================

class ImprovedPointHeatmapGenerator:
    def __init__(self, sigma=8.0):
        self.sigma = sigma

    def generate_heatmap(self, points, image_shape=(640, 640)):
        if len(points) == 0:
            return np.zeros(image_shape, dtype=np.float32)

        heatmap = np.zeros(image_shape, dtype=np.float32)
        kernel_size = int(6 * self.sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1

        x, y = np.meshgrid(np.arange(kernel_size), np.arange(kernel_size))
        center = kernel_size // 2
        kernel = np.exp(-((x - center) ** 2 + (y - center) ** 2) / (2 * self.sigma ** 2))

        for point in points:
            x, y = int(point[0]), int(point[1])
            if 0 <= x < image_shape[1] and 0 <= y < image_shape[0]:
                x_start, x_end = max(0, x - center), min(image_shape[1], x + center + 1)
                y_start, y_end = max(0, y - center), min(image_shape[0], y + center + 1)
                
                k_x_start = max(0, center - x)
                k_x_end = k_x_start + (x_end - x_start)
                k_y_start = max(0, center - y)
                k_y_end = k_y_start + (y_end - y_start)
                
                heatmap[y_start:y_end, x_start:x_end] = np.maximum(
                    heatmap[y_start:y_end, x_start:x_end],
                    kernel[k_y_start:k_y_end, k_x_start:k_x_end]
                )

        return heatmap

class ImprovedKi67PointDetectionModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = smp.Unet(
            encoder_name='efficientnet-b3',
            encoder_weights='imagenet',
            in_channels=3,
            classes=2,
            activation=None
        )
    
    def forward(self, x):
        return self.model(x)

# ============================================================================
# FLASK APP SETUP
# ============================================================================

app = Flask(__name__)

# Enable CORS for React frontend
CORS(app, origins=['http://localhost:3000', 'http://localhost:5173', 'http://127.0.0.1:3000', 'http://127.0.0.1:5173'])

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'

# Create directories
for folder in [app.config['UPLOAD_FOLDER'], app.config['RESULTS_FOLDER']]:
    os.makedirs(folder, exist_ok=True)

# ============================================================================
# GLOBAL VARIABLES
# ============================================================================

model = None
model_loaded = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model():
    global model, model_loaded
    try:
        print("🔄 Loading model...")
        model = ImprovedKi67PointDetectionModel.load_from_checkpoint(
            'ki67-point-epoch=68-val_peak_f1_avg=0.8503.ckpt',
            map_location=device
        )
        model.eval()
        model_loaded = True
        print(f"✅ Model loaded on {device}")
        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        model_loaded = False
        return False

# ============================================================================
# IMAGE PROCESSING FUNCTIONS
# ============================================================================

def preprocess_image(image_path, target_size=(640, 640)):
    """Preprocess image for model inference."""
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_shape = image.shape[:2]
        
        # Resize to target size
        image_resized = cv2.resize(image, target_size)
        
        # Normalize
        transform = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        transformed = transform(image=image_resized)
        image_tensor = transformed['image'].unsqueeze(0)
        
        return image_tensor, original_shape, image_resized
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        raise

def detect_ki67_cells(image_path, threshold=0.3, min_distance=10):
    """Detect Ki-67 positive cells using the loaded model."""
    try:
        if not model_loaded:
            raise ValueError("Model not loaded")
        
        image_tensor, original_shape, image_resized = preprocess_image(image_path)
        
        with torch.no_grad():
            outputs = model(image_tensor.to(device))
            positive_heatmap = torch.sigmoid(outputs[0, 0]).cpu().numpy()
            negative_heatmap = torch.sigmoid(outputs[0, 1]).cpu().numpy()
        
        # Find peaks for positive cells
        positive_peaks = peak_local_max(
            positive_heatmap,
            min_distance=min_distance,
            threshold_abs=threshold
        )
        
        # Find peaks for negative cells  
        negative_peaks = peak_local_max(
            negative_heatmap,
            min_distance=min_distance,
            threshold_abs=threshold
        )
        
        # Convert coordinates
        positive_coords = list(zip(positive_peaks[1], positive_peaks[0]))
        negative_coords = list(zip(negative_peaks[1], negative_peaks[0]))
        
        # Scale coordinates to original image size
        scale_x = original_shape[1] / 640
        scale_y = original_shape[0] / 640
        
        positive_coords_scaled = [(int(x * scale_x), int(y * scale_y)) for x, y in positive_coords]
        negative_coords_scaled = [(int(x * scale_x), int(y * scale_y)) for x, y in negative_coords]
        
        return {
            'positive_cells': len(positive_coords_scaled),
            'negative_cells': len(negative_coords_scaled),
            'total_cells': len(positive_coords_scaled) + len(negative_coords_scaled),
            'ki67_index': (len(positive_coords_scaled) / max(1, len(positive_coords_scaled) + len(negative_coords_scaled))) * 100,
            'positive_coords': positive_coords_scaled,
            'negative_coords': negative_coords_scaled,
            'heatmaps': {
                'positive': positive_heatmap.tolist(),
                'negative': negative_heatmap.tolist()
            }
        }
    except Exception as e:
        print(f"Error in cell detection: {e}")
        raise

def classify_tumor(ki67_index):
    """Classify tumor based on Ki-67 index."""
    if ki67_index < 14:
        return {
            'status': 'Low Grade',
            'risk': 'Low Risk',
            'description': 'Low proliferative activity',
            'recommendation': 'Standard follow-up recommended'
        }
    elif ki67_index < 30:
        return {
            'status': 'Intermediate Grade',
            'risk': 'Moderate Risk', 
            'description': 'Moderate proliferative activity',
            'recommendation': 'Close monitoring and targeted therapy consideration'
        }
    else:
        return {
            'status': 'High Grade',
            'risk': 'High Risk',
            'description': 'High proliferative activity',
            'recommendation': 'Aggressive treatment protocol recommended'
        }

# ============================================================================
# API ROUTES
# ============================================================================

@app.route('/')
def serve_react():
    """Serve the React frontend."""
    return send_from_directory('../frontend-react/dist', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    """Serve static files from React build."""
    return send_from_directory('../frontend-react/dist', path)

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded,
        'device': str(device)
    })

@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    """Analyze single image for Ki-67 cells."""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400
        
        # Get form data
        patient_id = request.form.get('patientId', f'PT-{uuid.uuid4().hex[:8]}')
        manual_positive = request.form.get('manualPositive', type=int)
        manual_negative = request.form.get('manualNegative', type=int)
        notes = request.form.get('notes', '')
        
        # Save file
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4().hex}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)
        
        # Analyze image
        results = detect_ki67_cells(file_path)
        
        # Add classification
        classification = classify_tumor(results['ki67_index'])
        
        # Prepare response
        analysis_result = {
            'id': str(uuid.uuid4()),
            'timestamp': datetime.now().isoformat(),
            'patientId': patient_id,
            'imageName': filename,
            'ki67Index': round(results['ki67_index'], 2),
            'totalCells': results['total_cells'],
            'positiveCells': results['positive_cells'],
            'negativeCells': results['negative_cells'],
            'classification': classification,
            'manualPositive': manual_positive,
            'manualNegative': manual_negative,
            'notes': notes,
            'coordinates': {
                'positive': results['positive_coords'],
                'negative': results['negative_coords']
            }
        }
        
        return jsonify(analysis_result)
        
    except Exception as e:
        print(f"Error in analyze_image: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/batch-analyze', methods=['POST'])
def batch_analyze():
    """Analyze multiple images."""
    try:
        if 'images' not in request.files:
            return jsonify({'error': 'No images provided'}), 400
        
        files = request.files.getlist('images')
        patient_prefix = request.form.get('patientPrefix', 'PT-BATCH')
        
        results = []
        
        for i, file in enumerate(files):
            if file.filename == '':
                continue
                
            # Save file
            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4().hex}_{filename}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(file_path)
            
            # Analyze
            cell_results = detect_ki67_cells(file_path)
            classification = classify_tumor(cell_results['ki67_index'])
            
            # Create result
            result = {
                'id': str(uuid.uuid4()),
                'timestamp': datetime.now().isoformat(),
                'patientId': f"{patient_prefix}-{i+1:03d}",
                'imageName': filename,
                'ki67Index': round(cell_results['ki67_index'], 2),
                'totalCells': cell_results['total_cells'],
                'positiveCells': cell_results['positive_cells'],
                'negativeCells': cell_results['negative_cells'],
                'classification': classification,
                'coordinates': {
                    'positive': cell_results['positive_coords'],
                    'negative': cell_results['negative_coords']
                }
            }
            
            results.append(result)
        
        return jsonify({'results': results})
        
    except Exception as e:
        print(f"Error in batch_analyze: {e}")
        return jsonify({'error': str(e)}), 500

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("=" * 80)
    print("🔬 Ki-67 Analysis System - React + Flask Backend")
    print("=" * 80)
    
    # Load model
    if not load_model():
        print("❌ Failed to load model. Some features may not work.")
    
    print("\n✅ Server starting...")
    print("📱 React Frontend: Build and serve from frontend-react/dist")
    print("🚀 API Endpoints available at: http://localhost:5000/api/")
    print("⚡ Use Ctrl+C to stop")
    print("=" * 80)
    
    app.run(debug=True, host='0.0.0.0', port=5000)