"""
Ki-67 Medical Diagnostic System - Simplified Flask Backend
==========================================================
Flask API backend serving React frontend with simulated analysis
"""

import os
import uuid
import random
from datetime import datetime
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import base64

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
# UTILITY FUNCTIONS
# ============================================================================

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

def simulate_analysis():
    """Simulate Ki-67 analysis results."""
    positive_cells = random.randint(50, 200)
    negative_cells = random.randint(100, 400)
    total_cells = positive_cells + negative_cells
    ki67_index = (positive_cells / total_cells) * 100
    
    return {
        'positive_cells': positive_cells,
        'negative_cells': negative_cells,
        'total_cells': total_cells,
        'ki67_index': round(ki67_index, 2),
        'coordinates': {
            'positive': [[random.randint(0, 640), random.randint(0, 640)] for _ in range(positive_cells)],
            'negative': [[random.randint(0, 640), random.randint(0, 640)] for _ in range(negative_cells)]
        }
    }

def create_visualization(image_path, positive_coords, negative_coords):
    """Create visualization with cell markers."""
    try:
        print(f"\n[VIZ] Creating visualization for: {image_path}")
        print(f"[VIZ] Positive coords: {len(positive_coords)}, Negative coords: {len(negative_coords)}")
        
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"[VIZ ERROR] Could not read image from {image_path}")
            return None
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        vis_image = image.copy()
        
        # Get image dimensions
        height, width = image.shape[:2]
        print(f"[VIZ] Image dimensions: {width}x{height}")
        
        # Scale coordinates if needed (assuming 640x640 detection, scale to actual size)
        scale_x = width / 640
        scale_y = height / 640
        
        # Draw positive cells (green circles)
        for coord in positive_coords:
            x = int(coord[0] * scale_x)
            y = int(coord[1] * scale_y)
            if 0 <= x < width and 0 <= y < height:
                cv2.circle(vis_image, (x, y), 8, (0, 255, 0), 2)  # Green circle
                cv2.circle(vis_image, (x, y), 2, (0, 255, 0), -1)  # Green center dot
        
        # Draw negative cells (red circles)
        for coord in negative_coords:
            x = int(coord[0] * scale_x)
            y = int(coord[1] * scale_y)
            if 0 <= x < width and 0 <= y < height:
                cv2.circle(vis_image, (x, y), 8, (255, 0, 0), 2)  # Red circle
                cv2.circle(vis_image, (x, y), 2, (255, 0, 0), -1)  # Red center dot
        
        print(f"[VIZ] Successfully created visualization")
        return vis_image
    except Exception as e:
        print(f"[VIZ ERROR] Exception: {e}")
        import traceback
        traceback.print_exc()
        return None

def image_to_base64(image_array):
    """Convert numpy image array to base64 string."""
    try:
        print(f"[B64] Converting image to base64, shape: {image_array.shape}")
        # Convert RGB to BGR for cv2
        image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        # Encode to PNG
        success, buffer = cv2.imencode('.png', image_bgr)
        if not success:
            print("[B64 ERROR] Failed to encode image")
            return None
        
        # Convert to base64
        png_base64 = base64.b64encode(buffer).decode('utf-8')
        result = f"data:image/png;base64,{png_base64}"
        print(f"[B64] Successfully converted, length: {len(result)}")
        return result
    except Exception as e:
        print(f"[B64 ERROR] Exception: {e}")
        import traceback
        traceback.print_exc()
        return None

# ============================================================================
# ROUTES
# ============================================================================

@app.route('/')
def serve_react():
    """Serve the React frontend."""
    return send_from_directory('../frontend-react/dist', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    """Serve static files from React build."""
    try:
        return send_from_directory('../frontend-react/dist', path)
    except:
        # If file not found, serve index.html for client-side routing
        return send_from_directory('../frontend-react/dist', 'index.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': True,  # Simulated
        'message': 'Simulated analysis mode - for demonstration'
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
        
        # Simulate analysis
        results = simulate_analysis()
        classification = classify_tumor(results['ki67_index'])
        
        # Create visualization with cell markers
        print(f"\n[API] Starting visualization for {filename}")
        vis_image = create_visualization(
            file_path,
            results['coordinates']['positive'],
            results['coordinates']['negative']
        )
        
        # Convert images to base64
        print(f"[API] Reading original image")
        original_image = cv2.imread(file_path)
        if original_image is not None:
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        
        print(f"[API] Converting images to base64")
        images = {
            'original': image_to_base64(original_image) if original_image is not None else None,
            'analyzed': image_to_base64(vis_image) if vis_image is not None else None
        }
        print(f"[API] Images created - original: {images['original'] is not None}, analyzed: {images['analyzed'] is not None}")
        
        # Prepare response
        analysis_result = {
            'id': str(uuid.uuid4()),
            'timestamp': datetime.now().isoformat(),
            'patientId': patient_id,
            'imageName': filename,
            'ki67Index': results['ki67_index'],
            'totalCells': results['total_cells'],
            'positiveCells': results['positive_cells'],
            'negativeCells': results['negative_cells'],
            'classification': classification,
            'manualPositive': manual_positive,
            'manualNegative': manual_negative,
            'notes': notes,
            'coordinates': results['coordinates'],
            'images': images
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
            
            # Simulate analysis
            sim_results = simulate_analysis()
            classification = classify_tumor(sim_results['ki67_index'])
            
            # Create visualization with cell markers
            vis_image = create_visualization(
                file_path,
                sim_results['coordinates']['positive'],
                sim_results['coordinates']['negative']
            )
            
            # Convert images to base64
            original_image = cv2.imread(file_path)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            
            images = {
                'original': image_to_base64(original_image) if original_image is not None else None,
                'analyzed': image_to_base64(vis_image) if vis_image is not None else None
            }
            
            # Create result
            result = {
                'id': str(uuid.uuid4()),
                'timestamp': datetime.now().isoformat(),
                'patientId': f"{patient_prefix}-{i+1:03d}",
                'imageName': filename,
                'ki67Index': sim_results['ki67_index'],
                'totalCells': sim_results['total_cells'],
                'positiveCells': sim_results['positive_cells'],
                'negativeCells': sim_results['negative_cells'],
                'classification': classification,
                'coordinates': sim_results['coordinates'],
                'images': images
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
    print("🔬 Ki-67 Analysis System - Simplified Demo Mode")
    print("=" * 80)
    print("📝 Note: Using simulated analysis for demonstration")
    print("📱 React Frontend: http://localhost:5000")
    print("🚀 API Endpoints: http://localhost:5000/api/")
    print("⚡ Use Ctrl+C to stop")
    print("=" * 80)
    
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)