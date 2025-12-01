"""
Enhanced Ki-67 Point Detection Web Application with Advanced Features
- Batch processing
- History tracking
- Statistics dashboard
- Professional PDF report generation
- Export capabilities
"""

from flask import Flask, render_template, request, jsonify, send_file, make_response
from werkzeug.utils import secure_filename
import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
import io
import base64
import os
import sys
import segmentation_models_pytorch as smp
import pytorch_lightning as pl
from skimage.feature import peak_local_max
import json
from datetime import datetime
import h5py
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

# Import authentication system
from database import db, init_db, Analysis as AnalysisModel
from auth import init_auth, token_required, log_audit
from auth_routes import auth_bp

# Get the project root directory (parent of backend)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_PATH = os.path.join(PROJECT_ROOT, 'frontend')

# Initialize Flask app with custom paths
app = Flask(__name__,
            template_folder=os.path.join(FRONTEND_PATH, 'templates'),
            static_folder=os.path.join(FRONTEND_PATH, 'static'))

# Configuration
app.config['SECRET_KEY'] = 'ki67-secret-key-change-in-production'  # Change this in production!
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(PROJECT_ROOT, 'ki67_database.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
app.config['UPLOAD_FOLDER'] = os.path.join(PROJECT_ROOT, 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize database and authentication
init_db(app)
init_auth(app)

# Register authentication blueprint
app.register_blueprint(auth_bp)

# Disable caching for all responses
@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

# Global variables
MODEL = None
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SETTINGS = {
    'threshold': 0.3,
    'min_distance': 10
}

# Model Definition
class ImprovedKi67PointDetectionModel(pl.LightningModule):
    """Ki-67 point detection model using U-Net with EfficientNet-B3 encoder"""
    
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

def load_model_enhanced():
    """Load the pre-trained model"""
    global MODEL
    try:
        print("Loading model...")
        model_path = os.path.join(PROJECT_ROOT, 'ki67-point-epoch=68-val_peak_f1_avg=0.8503.ckpt')
        MODEL = ImprovedKi67PointDetectionModel.load_from_checkpoint(
            model_path,
            map_location=DEVICE,
            strict=False
        )
        MODEL.eval()
        MODEL.to(DEVICE)
        print(f"✅ Model loaded on {DEVICE}")
        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def load_ground_truth_points(annotation_dir, image_name):
    """Load ground truth points for validation from H5 files"""
    base_name = os.path.splitext(image_name)[0]
    
    pos_h5 = os.path.join(annotation_dir, 'positive', f"{base_name}.h5")
    neg_h5 = os.path.join(annotation_dir, 'negative', f"{base_name}.h5")
    
    pos_points = []
    neg_points = []
    
    # Load positive points
    try:
        with h5py.File(pos_h5, 'r') as f:
            if 'coordinates' in f:
                pos_points = f['coordinates'][:]
    except:
        pass
    
    # Load negative points
    try:
        with h5py.File(neg_h5, 'r') as f:
            if 'coordinates' in f:
                neg_points = f['coordinates'][:]
    except:
        pass
    
    return np.array(pos_points), np.array(neg_points)

def preprocess_image(image_path):
    """Preprocess image for model input"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(DEVICE)
    return img, img_tensor

def detect_cells_enhanced(heatmap, threshold=0.3, min_distance=10):
    """Detect cells from heatmap using peak detection"""
    coordinates = peak_local_max(
        heatmap,
        threshold_abs=threshold,
        min_distance=min_distance,
        exclude_border=False
    )
    return coordinates

def predict_cells_enhanced(image_path, threshold=None, min_distance=None):
    """Predict positive and negative cells with adjustable parameters"""
    if threshold is None:
        threshold = SETTINGS['threshold']
    if min_distance is None:
        min_distance = SETTINGS['min_distance']
    
    original_img, img_tensor = preprocess_image(image_path)
    
    with torch.no_grad():
        output = MODEL(img_tensor)
        output = torch.sigmoid(output)
        heatmaps = output[0].cpu().numpy()
    
    positive_coords = detect_cells_enhanced(heatmaps[0], threshold, min_distance)
    negative_coords = detect_cells_enhanced(heatmaps[1], threshold, min_distance)
    
    return original_img, positive_coords, negative_coords

def create_visualization_enhanced(img, positive_coords, negative_coords, patient_id=""):
    """Create a comprehensive 4-panel visualization"""
    h, w = img.shape[:2]
    
    # Create 2x2 grid
    viz = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)
    
    # Panel 1: Original image
    viz[:h, :w] = img
    cv2.putText(viz, "Original Image", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Panel 2: Positive cells (Ki-67+)
    positive_img = img.copy()
    for coord in positive_coords:
        cv2.circle(positive_img, (coord[1], coord[0]), 5, (0, 255, 0), 2)
    viz[:h, w:] = positive_img
    cv2.putText(viz, f"Ki-67 Positive: {len(positive_coords)}", (w + 10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Panel 3: Negative cells (Ki-67-)
    negative_img = img.copy()
    for coord in negative_coords:
        cv2.circle(negative_img, (coord[1], coord[0]), 5, (255, 0, 0), 2)
    viz[h:, :w] = negative_img
    cv2.putText(viz, f"Ki-67 Negative: {len(negative_coords)}", (10, h + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    
    # Panel 4: Combined overlay
    combined_img = img.copy()
    for coord in positive_coords:
        cv2.circle(combined_img, (coord[1], coord[0]), 5, (0, 255, 0), 2)
    for coord in negative_coords:
        cv2.circle(combined_img, (coord[1], coord[0]), 5, (255, 0, 0), 2)
    viz[h:, w:] = combined_img
    
    total_cells = len(positive_coords) + len(negative_coords)
    ki67_index = (len(positive_coords) / total_cells * 100) if total_cells > 0 else 0
    
    cv2.putText(viz, f"Combined (Ki-67: {ki67_index:.1f}%)", (w + 10, h + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    if patient_id:
        cv2.putText(viz, f"Patient: {patient_id}", (10, h * 2 - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    
    return viz

def image_to_base64(img_array):
    """Convert image array to base64 string"""
    is_success, buffer = cv2.imencode(".png", cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
    if not is_success:
        raise ValueError("Could not encode image")
    return base64.b64encode(buffer).decode()

def generate_pdf_report(analysis_data):
    """
    Generate a professional medical PDF report for Ki-67 analysis using ReportLab
    
    Args:
        analysis_data: Dict containing analysis results and validation data
        
    Returns:
        BytesIO object containing the PDF
    """
    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(pdf_buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    # ===== Header =====
    header = Paragraph('Ki-67 Proliferation Assessment Report', styles['Title'])
    story.append(header)
    story.append(Spacer(1, 12))

    # ===== Metadata Table =====
    analysis_id = analysis_data.get('analysis_id', f"KI67-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    patient_id = analysis_data.get('patient_id', 'N/A')
    filename = analysis_data.get('filename', 'N/A')
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    metadata = [
        ['Analysis ID', analysis_id],
        ['Analysis Timestamp', timestamp],
        ['Patient ID', patient_id],
        ['Image Filename', filename],
        ['Test Type', 'Ki-67 Immunohistochemistry'],
        ['Analysis Method', 'AI-Assisted Automated Detection'],
        ['Threshold', str(analysis_data.get('threshold', 0.3))],
        ['Min Distance', f"{analysis_data.get('min_distance', 10)}px"]
    ]

    table = Table(metadata, colWidths=[120, 360])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e0f7fa')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#004d40')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('GRID', (0, 0), (-1, -1), 0.25, colors.grey)
    ]))
    story.append(table)
    story.append(Spacer(1, 12))

    # ===== AI Detection Results =====
    ai_positive = analysis_data.get('ai_positive', 0)
    ai_negative = analysis_data.get('ai_negative', 0)
    total_cells = ai_positive + ai_negative
    ki67_index = analysis_data.get('ai_ki67', 0)
    classification = analysis_data.get('classification', 'Unknown')
    
    # Determine risk level
    if ki67_index < 14:
        risk_level = "Low Proliferation"
    elif ki67_index < 30:
        risk_level = "Intermediate Proliferation"
    else:
        risk_level = "High Proliferation"

    metrics_header = Paragraph('Model Output Summary', styles['Heading2'])
    story.append(metrics_header)
    model_table = Table([
        ['Positive Cells', f"{ai_positive:,}"],
        ['Negative Cells', f"{ai_negative:,}"],
        ['Total Cells', f"{total_cells:,}"],
        ['Ki-67 Index', f"{ki67_index:.2f}%"],
        ['Classification', classification],
        ['Risk Level', risk_level]
    ], colWidths=[180, 300])
    model_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.whitesmoke),
        ('GRID', (0, 0), (-1, -1), 0.25, colors.lightgrey)
    ]))
    story.append(model_table)
    story.append(Spacer(1, 12))

    # ===== Ground Truth Validation (if available) =====
    if 'manual_positive' in analysis_data or 'gt_positive' in analysis_data:
        gt_positive = analysis_data.get('manual_positive', analysis_data.get('gt_positive', 0))
        gt_negative = analysis_data.get('manual_negative', analysis_data.get('gt_negative', 0))
        gt_total = gt_positive + gt_negative
        gt_ki67 = (gt_positive / gt_total * 100) if gt_total > 0 else 0
        
        # Determine GT risk level
        if gt_ki67 < 14:
            gt_risk_level = "Low Proliferation"
            gt_classification = "Benign"
        elif gt_ki67 < 30:
            gt_risk_level = "Intermediate Proliferation"
            gt_classification = "Intermediate"
        else:
            gt_risk_level = "High Proliferation"
            gt_classification = "Malignant"

        manual_header = Paragraph('Manual (Ground Truth) Baseline', styles['Heading2'])
        story.append(manual_header)
        manual_table = Table([
            ['Positive Cells', f"{gt_positive:,}"],
            ['Negative Cells', f"{gt_negative:,}"],
            ['Total Cells', f"{gt_total:,}"],
            ['Ki-67 Index', f"{gt_ki67:.2f}%"],
            ['Classification', gt_classification],
            ['Risk Level', gt_risk_level]
        ], colWidths=[180, 300])
        manual_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#fff3e0')),
            ('GRID', (0, 0), (-1, -1), 0.25, colors.lightgrey)
        ]))
        story.append(manual_table)
        story.append(Spacer(1, 12))

        # ===== Quality Control Summary =====
        diff_positive = ai_positive - gt_positive
        diff_negative = ai_negative - gt_negative
        diff_total = total_cells - gt_total
        diff_ki67 = ki67_index - gt_ki67
        classification_match = classification.upper() == gt_classification.upper()
        
        # Calculate percent difference
        percent_diff = abs(diff_ki67 / gt_ki67 * 100) if gt_ki67 > 0 else 0
        
        # Flagging logic
        flagged = (
            abs(diff_ki67) > 5.0 or 
            percent_diff > 15.0 or 
            not classification_match
        )
        
        if flagged:
            if abs(diff_ki67) > 5.0:
                reason = f"Ki-67 difference exceeds threshold ({abs(diff_ki67):.2f}% > 5.0%)"
            elif percent_diff > 15.0:
                reason = f"Ki-67 percent difference exceeds threshold ({percent_diff:.1f}% > 15.0%)"
            elif not classification_match:
                reason = "Classification mismatch between AI and ground truth"
            else:
                reason = "Multiple quality issues detected"
        else:
            reason = "No significant discrepancies"

        qc_header = Paragraph('Quality Control Summary', styles['Heading2'])
        story.append(qc_header)
        qc_table_data = [
            ['Flagged for Review', 'Yes' if flagged else 'No'],
            ['Reason', reason],
            ['Positive Cell Δ', f"{diff_positive:+,}"],
            ['Negative Cell Δ', f"{diff_negative:+,}"],
            ['Total Cell Δ', f"{diff_total:+,}"],
            ['Ki-67 Δ', f"{diff_ki67:+.2f}%"],
            ['Ki-67 Δ (%)', f"{diff_ki67 / gt_ki67 * 100 if gt_ki67 > 0 else 0:+.1f}%"],
            ['Classification Match', 'Yes' if classification_match else 'No']
        ]

        qc_table = Table(qc_table_data, colWidths=[200, 280])
        qc_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e8eaf6')),
            ('GRID', (0, 0), (-1, -1), 0.25, colors.lightgrey)
        ]))
        story.append(qc_table)
        story.append(Spacer(1, 12))

    # ===== Clinical Interpretation =====
    if ki67_index < 14:
        interpretation = "Low proliferation rate (<14%). Generally associated with less aggressive tumors and favorable prognosis."
        recommendation = "Continue standard surveillance. Correlate with hormone receptor status and other clinical findings."
    elif ki67_index < 30:
        interpretation = "Intermediate proliferation rate (14-30%). Moderate risk profile requiring multidisciplinary correlation."
        recommendation = "Consider adjunct molecular testing. Align treatment with hormone receptor status and clinical stage."
    else:
        interpretation = "High proliferation rate (>30%). Indicates aggressive tumor biology."
        recommendation = "Urgent oncologic consultation recommended. Consider systemic therapy options."

    story.append(Paragraph('Clinical Interpretation', styles['Heading2']))
    story.append(Paragraph(interpretation, styles['BodyText']))
    story.append(Spacer(1, 6))
    story.append(Paragraph('Recommended Action', styles['Heading2']))
    story.append(Paragraph(recommendation, styles['BodyText']))
    story.append(Spacer(1, 12))

    # ===== Clinical Notes =====
    if analysis_data.get('notes'):
        story.append(Paragraph('Clinical Notes', styles['Heading2']))
        story.append(Paragraph(analysis_data['notes'], styles['BodyText']))
        story.append(Spacer(1, 12))

    # ===== Disclaimer =====
    disclaimer = """
    <b>IMPORTANT MEDICAL DISCLAIMER:</b><br/>
    This is an AI-assisted analysis for research and clinical decision support purposes only.
    Results must be confirmed by a qualified pathologist before making clinical decisions.
    Not intended as a standalone diagnostic tool. For research and educational use.
    <br/><br/>
    Ki-67 Analysis System v2.0 Enhanced with Ground Truth Validation | © 2024-2025 Medical AI Research Laboratory<br/>
    Generated using PyTorch, Segmentation Models PyTorch, and scikit-image libraries
    """
    story.append(Paragraph(disclaimer, styles['BodyText']))

    # Build PDF
    doc.build(story)
    pdf_buffer.seek(0)
    
    return pdf_buffer

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')

@app.route('/login')
def login_page():
    """Render login page"""
    return render_template('login.html')

@app.route('/register')
def register_page():
    """Render registration page"""
    return render_template('register.html')

@app.route('/settings', methods=['POST'])
def update_settings():
    """Update settings via JSON API"""
    global SETTINGS
    data = request.get_json() if request.is_json else request.form
    SETTINGS['threshold'] = float(data.get('threshold', 0.3))
    SETTINGS['min_distance'] = int(data.get('min_distance', 10))
    return jsonify({'success': True, 'message': 'Settings saved successfully', 'settings': SETTINGS})

@app.route('/batch/start', methods=['POST'])
def batch_start():
    """Handle batch image upload and analysis"""
    print("=" * 80)
    print("📥 Received BATCH analysis request")
    print(f"   Files: {list(request.files.keys())}")
    print("=" * 80)
    
    try:
        files = request.files.getlist('files')
        if not files or len(files) == 0:
            return '<div class="card" style="color: red;"><p>No files uploaded</p></div>', 400
        
        results = []
        for file in files:
            if file.filename == '':
                continue
                
            try:
                # Save file temporarily
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                # Analyze
                img, positive_coords, negative_coords = predict_cells_enhanced(filepath, SETTINGS['threshold'], SETTINGS['min_distance'])
                
                ai_positive = len(positive_coords)
                ai_negative = len(negative_coords)
                total_cells = ai_positive + ai_negative
                ai_ki67 = (ai_positive / total_cells * 100) if total_cells > 0 else 0
                classification = "Malignant" if ai_ki67 > 14 else "Benign"
                
                results.append({
                    'filename': filename,
                    'positive': ai_positive,
                    'negative': ai_negative,
                    'total': total_cells,
                    'ki67': round(ai_ki67, 2),
                    'classification': classification
                })
                
                # Cleanup
                os.remove(filepath)
                print(f"✅ Processed: {filename} - Ki-67: {ai_ki67:.2f}%")
                
            except Exception as e:
                print(f"❌ Error processing {file.filename}: {e}")
                results.append({
                    'filename': file.filename,
                    'error': str(e)
                })
        
        # Generate HTML response
        html = '<div class="card"><h3><i class="fas fa-list"></i> Batch Results</h3>'
        html += '<table style="width: 100%; border-collapse: collapse;">'
        html += '<thead><tr style="background: #f5f5f5;"><th style="padding: 10px; text-align: left;">File</th><th>Positive</th><th>Negative</th><th>Ki-67 Index</th><th>Classification</th></tr></thead>'
        html += '<tbody>'
        
        for r in results:
            if 'error' in r:
                html += f'<tr style="border-bottom: 1px solid #eee;"><td style="padding: 10px;">{r["filename"]}</td><td colspan="4" style="color: red;">Error: {r["error"]}</td></tr>'
            else:
                badge_color = "#dc3545" if r["classification"] == "Malignant" else "#28a745"
                html += f'''<tr style="border-bottom: 1px solid #eee;">
                    <td style="padding: 10px;">{r["filename"]}</td>
                    <td style="text-align: center;">{r["positive"]}</td>
                    <td style="text-align: center;">{r["negative"]}</td>
                    <td style="text-align: center; font-weight: bold;">{r["ki67"]}%</td>
                    <td style="text-align: center;"><span style="background: {badge_color}; color: white; padding: 3px 10px; border-radius: 20px;">{r["classification"]}</span></td>
                </tr>'''
        
        html += '</tbody></table></div>'
        
        return html
        
    except Exception as e:
        print(f"Batch error: {e}")
        import traceback
        traceback.print_exc()
        return f'<div class="card" style="color: red;"><p>Error: {str(e)}</p></div>', 500

@app.route('/analyze', methods=['POST'])
def analyze_file():
    """Handle single image upload and analysis"""
    print("=" * 80)
    print("📥 Received analysis request")
    print(f"   Headers: {dict(request.headers)}")
    print(f"   Files: {list(request.files.keys())}")
    print(f"   Form: {dict(request.form)}")
    print("=" * 80)
    
    try:
        if 'file' not in request.files:
            print("❌ No file in request")
            return jsonify({'success': False, 'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            print("❌ Empty filename")
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        print(f"✅ Processing file: {file.filename}")
        
        # Get form data
        patient_id = request.form.get('patient_id', 'Unknown')
        manual_positive = request.form.get('manual_positive', None)
        manual_negative = request.form.get('manual_negative', None)
        notes = request.form.get('notes', '')
        
        # Get custom settings if provided
        threshold = float(request.form.get('threshold', SETTINGS['threshold']))
        min_distance = int(request.form.get('min_distance', SETTINGS['min_distance']))
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Predict cells
        img, positive_coords, negative_coords = predict_cells_enhanced(
            filepath, threshold, min_distance
        )
        
        # Calculate metrics
        ai_positive = len(positive_coords)
        ai_negative = len(negative_coords)
        total_cells = ai_positive + ai_negative
        ai_ki67 = (ai_positive / total_cells * 100) if total_cells > 0 else 0
        
        # Classify
        classification = "Malignant" if ai_ki67 > 14 else "Benign"
        classification_icon = "⚠️" if classification == "Malignant" else "✅"
        
        # Create visualization
        viz = create_visualization_enhanced(img, positive_coords, negative_coords, patient_id)
        viz_base64 = image_to_base64(viz)
        
        # Generate unique analysis ID
        analysis_id = f"KI67-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{hash(patient_id) % 10000:04d}"
        
        print(f"✅ Analysis complete: {ai_positive} positive, {ai_negative} negative, Ki-67: {ai_ki67:.2f}%")
        
        # Try to load ground truth data if available (for validation)
        gt_positive_count = None
        gt_negative_count = None
        try:
            # Check if ground truth annotations exist
            annotation_dir = os.path.join(PROJECT_ROOT, 'BCData', 'annotations', 'test')
            if os.path.exists(annotation_dir):
                gt_pos_points, gt_neg_points = load_ground_truth_points(annotation_dir, filename)
                if len(gt_pos_points) > 0 or len(gt_neg_points) > 0:
                    gt_positive_count = len(gt_pos_points)
                    gt_negative_count = len(gt_neg_points)
                    print(f"✅ Loaded ground truth: {gt_positive_count} positive, {gt_negative_count} negative")
        except Exception as e:
            print(f"⚠️ Could not load ground truth data: {e}")
        
        # Prepare response
        response = {
            'success': True,
            'analysis_id': analysis_id,
            'id': analysis_id,  # Alias for template
            'patient_id': patient_id,
            'filename': filename,
            'ai_positive': ai_positive,
            'positive_count': ai_positive,  # Alias for template
            'ai_negative': ai_negative,
            'negative_count': ai_negative,  # Alias for template
            'ai_ki67': round(ai_ki67, 2),
            'ki67_index': round(ai_ki67, 2),  # Alias for template
            'total_cells': total_cells,
            'classification': classification,
            'classification_icon': classification_icon,
            'visualization': f'data:image/png;base64,{viz_base64}',
            'timestamp': datetime.now().isoformat(),
            'notes': notes,
            'threshold': threshold,
            'min_distance': min_distance
        }
        
        # Add comparison if manual counts provided
        if manual_positive and manual_negative:
            try:
                manual_positive = int(manual_positive)
                manual_negative = int(manual_negative)
                manual_total = manual_positive + manual_negative
                manual_ki67 = (manual_positive / manual_total * 100) if manual_total > 0 else 0
                
                response.update({
                    'manual_positive': manual_positive,
                    'manual_negative': manual_negative,
                    'manual_ki67': round(manual_ki67, 2),
                    'diff_positive': ai_positive - manual_positive,
                    'diff_negative': ai_negative - manual_negative,
                    'diff_ki67': round(ai_ki67 - manual_ki67, 2)
                })
            except (ValueError, TypeError):
                pass
        
        # Add ground truth validation data if available
        if gt_positive_count is not None and gt_negative_count is not None:
            gt_total = gt_positive_count + gt_negative_count
            gt_ki67 = (gt_positive_count / gt_total * 100) if gt_total > 0 else 0
            
            response.update({
                'gt_positive': gt_positive_count,
                'gt_negative': gt_negative_count,
                'gt_ki67': round(gt_ki67, 2),
                'gt_total': gt_total,
                'has_ground_truth': True
            })
        
        # Clean up
        os.remove(filepath)
            
        return jsonify(response)
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/generate_pdf', methods=['POST'])
def generate_pdf():
    """Generate and download PDF medical report"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        # Generate PDF
        pdf_buffer = generate_pdf_report(data)
        
        # Generate filename
        patient_id = data.get('patient_id', 'Unknown').replace(' ', '_')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"Ki67_Medical_Report_{patient_id}_{timestamp}.pdf"
        
        # Send file
        return send_file(
            pdf_buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=filename
        )
    
    except Exception as e:
        print(f"PDF generation error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': MODEL is not None,
        'device': str(DEVICE),
        'settings': SETTINGS,
        'version': '2.0-enhanced'
    })

if __name__ == '__main__':
    print("=" * 80)
    print("🔬 Ki-67 Breast Cancer Diagnosis System - Enhanced Version 2.0")
    print("=" * 80)
    
    # Load model
    if not load_model_enhanced():
        print("❌ Failed to load model. Exiting.")
        exit(1)
    
    print("\n" + "=" * 80)
    print("✅ Server is ready!")
    print("=" * 80)
    print(f"📱 Access the enhanced web application at:")
    print(f"   • http://localhost:5002")
    print(f"   • http://127.0.0.1:5002")
    print("=" * 80)
    print("\n🚀 Enhanced Features:")
    print("   • ✨ Professional medical interface")
    print("   • 📊 Single image analysis with detailed metrics")
    print("   • 📁 Batch processing for multiple images")
    print("   • 📜 Complete analysis history tracking")
    print("   • 📈 Statistics dashboard with charts")
    print("   • 📄 Professional PDF medical reports")
    print("   • 💾 CSV data export")
    print("   • 🌓 Dark mode support")
    print("   • ⚙️ Adjustable detection parameters")
    print("=" * 80)
    print("\n⚡ Press Ctrl+C to stop the server\n")
    
    # Run server
    app.run(debug=False, host='0.0.0.0', port=5002)
