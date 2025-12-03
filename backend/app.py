"""
Ki-67 Medical Diagnostic System - Flask Backend
================================================
Professional medical web application for Ki-67 cell detection and analysis
"""

import os
import io
import csv
import base64
import uuid
import random
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

from flask import Flask, render_template, request, jsonify, send_file
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
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as ReportLabImage
import json
from sqlalchemy.orm import Session

# Local DB modules with robust import fallback for script execution
try:
    from .db import Base, engine, SessionLocal
    from .models import Upload, AnalysisResult
except Exception:
    import sys
    # Ensure backend package and project root are importable when running as a script
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
    if CURRENT_DIR not in sys.path:
        sys.path.insert(0, CURRENT_DIR)
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    from backend.db import Base, engine, SessionLocal
    from backend.models import Upload, AnalysisResult

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

        x = np.arange(0, kernel_size)
        y = x[:, np.newaxis]
        x0 = y0 = kernel_size // 2

        gaussian = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))
        gaussian = gaussian / gaussian.max()

        for point in points:
            x, y = int(point[0]), int(point[1])
            x_min = max(0, x - kernel_size // 2)
            x_max = min(image_shape[1], x + kernel_size // 2 + 1)
            y_min = max(0, y - kernel_size // 2)
            y_max = min(image_shape[0], y + kernel_size // 2 + 1)
            k_x_min = max(0, kernel_size // 2 - x)
            k_x_max = min(kernel_size, kernel_size // 2 + (image_shape[1] - x))
            k_y_min = max(0, kernel_size // 2 - y)
            k_y_max = min(kernel_size, kernel_size // 2 + (image_shape[0] - y))

            heatmap[y_min:y_max, x_min:x_max] = np.maximum(
                heatmap[y_min:y_max, x_min:x_max],
                gaussian[k_y_min:k_y_max, k_x_min:k_x_max]
            )

        return heatmap


class ImprovedHeatmapLoss(nn.Module):
    def __init__(self, pos_weight=10.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight]),
            reduction='mean'
        )

    def forward(self, pred, target):
        return self.bce(pred, target)


class ImprovedKi67PointDetectionModel(pl.LightningModule):
    def __init__(self, encoder_name='efficientnet-b3', learning_rate=1e-4):
        super().__init__()
        self.save_hyperparameters()

        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights='imagenet',
            in_channels=3,
            classes=2,
            activation=None,
        )

        self.criterion = ImprovedHeatmapLoss(pos_weight=10.0)

    def forward(self, x):
        return self.model(x)


# ============================================================================
# FLASK APPLICATION
# ============================================================================

app = Flask(__name__, 
            template_folder='../frontend/templates',
            static_folder='../frontend/static')
CORS(app)

# Get the project root directory (parent of backend folder)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'uploads')
app.config['RESULTS_FOLDER'] = os.path.join(BASE_DIR, 'results')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tif', 'tiff'}
DATASET_IMAGE_ROOT = os.path.join(BASE_DIR, 'BCData', 'images')
DATASET_ANNOTATION_ROOT = os.path.join(BASE_DIR, 'BCData', 'annotations')
QUALITY_THRESHOLD_PERCENT = 10.0  # percent difference threshold for QC flag

# Global model variable
model = None
device = None
transform = None

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_validation_augmentation():
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def predict_points_from_heatmaps(heatmaps, threshold=0.5):
    """Extract points from predicted heatmaps"""
    pos_heatmap = heatmaps[0]
    neg_heatmap = heatmaps[1]

    # Detect peaks
    pos_peaks = peak_local_max(pos_heatmap, threshold_abs=threshold, min_distance=10)
    neg_peaks = peak_local_max(neg_heatmap, threshold_abs=threshold, min_distance=10)

    return pos_peaks, neg_peaks


def load_ground_truth_points(image_name: str,
                             subset_hint: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, Optional[str]]:
    """Load ground truth points (manual annotations) for a dataset image."""
    base_name = os.path.splitext(image_name)[0]
    subsets: List[str] = []

    if subset_hint:
        subsets.append(subset_hint)
    subsets.extend([s for s in ['test', 'validation', 'train'] if s not in subsets])

    for subset in subsets:
        pos_path = os.path.join(DATASET_ANNOTATION_ROOT, subset, 'positive', f'{base_name}.h5')
        neg_path = os.path.join(DATASET_ANNOTATION_ROOT, subset, 'negative', f'{base_name}.h5')

        if os.path.exists(pos_path) and os.path.exists(neg_path):
            try:
                with h5py.File(pos_path, 'r') as pf:
                    pos_points = pf['coordinates'][:] if 'coordinates' in pf else np.array([])
                with h5py.File(neg_path, 'r') as nf:
                    neg_points = nf['coordinates'][:] if 'coordinates' in nf else np.array([])
                return pos_points, neg_points, subset
            except Exception as exc:
                print(f"⚠️ Failed to load annotations for {image_name} in {subset}: {exc}")
                continue

    return np.array([]), np.array([]), None


def create_ground_truth_visualization(image: np.ndarray,
                                      gt_pos: np.ndarray,
                                      gt_neg: np.ndarray) -> np.ndarray:
    """Create visualization for manual annotations."""
    vis_image = image.copy()

    for point in gt_pos:
        x, y = int(point[0]), int(point[1])
        cv2.circle(vis_image, (x, y), 5, (255, 165, 0), 2)
        cv2.circle(vis_image, (x, y), 2, (255, 255, 255), -1)

    for point in gt_neg:
        x, y = int(point[0]), int(point[1])
        cv2.circle(vis_image, (x, y), 5, (0, 255, 255), 2)
        cv2.circle(vis_image, (x, y), 2, (255, 255, 255), -1)

    return vis_image


def create_comparison_visualization(image: np.ndarray,
                                    pred_pos: np.ndarray,
                                    pred_neg: np.ndarray,
                                    gt_pos: np.ndarray,
                                    gt_neg: np.ndarray) -> np.ndarray:
    """Create overlay visualizing both model predictions and manual annotations."""
    vis_image = image.copy()

    # Model predictions (cross markers)
    for point in pred_pos:
        y, x = int(point[0]), int(point[1])
        cv2.drawMarker(vis_image, (x, y), (255, 0, 0), markerType=cv2.MARKER_TILTED_CROSS,
                       markerSize=12, thickness=2)

    for point in pred_neg:
        y, x = int(point[0]), int(point[1])
        cv2.drawMarker(vis_image, (x, y), (0, 0, 255), markerType=cv2.MARKER_TILTED_CROSS,
                       markerSize=12, thickness=2)

    # Manual annotations (filled circles)
    for point in gt_pos:
        x, y = int(point[0]), int(point[1])
        cv2.circle(vis_image, (x, y), 5, (255, 165, 0), -1)

    for point in gt_neg:
        x, y = int(point[0]), int(point[1])
        cv2.circle(vis_image, (x, y), 5, (0, 255, 255), -1)

    return vis_image


def calculate_manual_metrics(gt_pos: np.ndarray, gt_neg: np.ndarray) -> Optional[Dict[str, Any]]:
    """Calculate manual counts and Ki-67 metrics if ground truth is available."""
    if gt_pos.size == 0 and gt_neg.size == 0:
        return None

    positive_count = int(len(gt_pos))
    negative_count = int(len(gt_neg))
    total_count = positive_count + negative_count
    ki_index = calculate_ki67_index(positive_count, negative_count)

    return {
        'positive_cells': positive_count,
        'negative_cells': negative_count,
        'total_cells': total_count,
        'ki67_index': round(ki_index, 2),
        'diagnosis': get_diagnosis(ki_index)
    }


def manual_metrics_from_counts(positive: Optional[str], negative: Optional[str]) -> Optional[Dict[str, Any]]:
    """Build manual metrics from numeric input fields."""
    if positive is None or negative is None or positive == '' or negative == '':
        return None

    try:
        pos_val = max(int(positive), 0)
        neg_val = max(int(negative), 0)
    except ValueError:
        return None

    total = pos_val + neg_val
    ki67 = calculate_ki67_index(pos_val, neg_val)

    return {
        'positive_cells': pos_val,
        'negative_cells': neg_val,
        'total_cells': total,
        'ki67_index': round(ki67, 2),
        'diagnosis': get_diagnosis(ki67)
    }


def evaluate_quality(model_metrics: Dict[str, Any],
                     manual_metrics: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute quality control indicators comparing manual and model outputs."""
    if not manual_metrics:
        return {
            'available': False,
            'flagged': False,
            'reason': 'Manual baseline unavailable',
            'differences': {}
        }

    diff_positive = model_metrics['positive_cells'] - manual_metrics['positive_cells']
    diff_negative = model_metrics['negative_cells'] - manual_metrics['negative_cells']
    diff_total = model_metrics['total_cells'] - manual_metrics['total_cells']
    diff_ki67 = round(model_metrics['ki67_index'] - manual_metrics['ki67_index'], 2)

    percent_diff = 0.0
    if manual_metrics['ki67_index']:
        percent_diff = round((diff_ki67 / manual_metrics['ki67_index']) * 100, 2)

    classification_match = manual_metrics['diagnosis']['classification'] == model_metrics['diagnosis']['classification']

    flagged = abs(percent_diff) > QUALITY_THRESHOLD_PERCENT or not classification_match
    reason = []
    if abs(percent_diff) > QUALITY_THRESHOLD_PERCENT:
        reason.append(f'Ki-67 deviation > {QUALITY_THRESHOLD_PERCENT}%')
    if not classification_match:
        reason.append('Classification mismatch')

    return {
        'available': True,
        'flagged': flagged,
        'reason': '; '.join(reason) if reason else 'Within acceptable tolerance',
        'differences': {
            'positive_delta': diff_positive,
            'negative_delta': diff_negative,
            'total_delta': diff_total,
            'ki67_delta': diff_ki67,
            'ki67_percent_delta': percent_diff,
            'classification_match': classification_match
        }
    }


def ensure_results_directory(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def generate_pdf_report(analysis_data: Dict[str, Any], pdf_path: str) -> None:
    """Generate a professional PDF report summarizing the analysis."""
    ensure_results_directory(pdf_path)

    doc = SimpleDocTemplate(pdf_path, pagesize=A4, topMargin=30, bottomMargin=30)
    styles = getSampleStyleSheet()
    story: List[Any] = []

    # Title with colored background
    header_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Title'],
        fontSize=20,
        textColor=colors.HexColor('#004d40'),
        spaceAfter=20,
        alignment=1  # Center
    )
    header = Paragraph('Ki-67 Proliferation Assessment Report', header_style)
    story.append(header)
    story.append(Spacer(1, 12))

    # Patient Information Section
    patient = analysis_data.get('patient_data', {})
    story.append(Paragraph('Patient Information', styles['Heading2']))
    metadata = [
        ['Analysis ID', analysis_data.get('analysis_id', '-')],
        ['Analysis Timestamp', analysis_data.get('timestamp', '-')],
        ['Patient Name', patient.get('patient_name', '-')],
        ['Patient ID', patient.get('patient_id', '-')],
        ['Age', patient.get('age', '-')],
        ['Gender', patient.get('gender', '-')],
        ['Contact', patient.get('contact', '-')],
        ['Exam Date', patient.get('exam_date', '-')],
        ['Physician', patient.get('physician', '-')],
        ['Clinical Notes', patient.get('clinical_notes', '-')]
    ]

    table = Table(metadata, colWidths=[140, 340])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e0f7fa')),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#004d40')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('VALIGN', (0, 0), (-1, -1), 'TOP')
    ]))
    story.append(table)
    story.append(Spacer(1, 18))

    model_metrics = analysis_data['results']
    manual_metrics = analysis_data.get('comparison', {}).get('manual')
    qc = analysis_data.get('comparison', {}).get('quality_control')

    # Cell Detection Visualization Images
    story.append(Paragraph('Cell Detection Visualization', styles['Heading2']))
    story.append(Spacer(1, 6))
    
    # Get result image path from analysis_id
    analysis_id = analysis_data.get('analysis_id', '')
    result_image_path = os.path.join(app.config['RESULTS_FOLDER'], f"result_{analysis_id}.png")
    
    if os.path.exists(result_image_path):
        try:
            # Create visualization image with detection markers
            img_obj = ReportLabImage(result_image_path, width=450, height=450*0.75)
            story.append(img_obj)
            story.append(Spacer(1, 6))
            story.append(Paragraph(
                '<font size=8 color="#666666">Figure: Cell detection visualization with Ki-67 positive cells (green) and Ki-67 negative cells (red)</font>',
                styles['BodyText']
            ))
            story.append(Spacer(1, 12))
        except Exception as e:
            print(f"Could not add image to PDF: {e}")
    
    story.append(Spacer(1, 12))

    # Analysis Results Summary with color coding
    metrics_header = Paragraph('Analysis Results Summary', styles['Heading2'])
    story.append(metrics_header)
    story.append(Spacer(1, 6))
    
    # Determine color based on classification
    diagnosis_color = model_metrics['diagnosis'].get('color', 'warning')
    if diagnosis_color == 'success':
        bg_color = colors.HexColor('#e8f5e9')
        header_color = colors.HexColor('#4caf50')
    elif diagnosis_color == 'warning':
        bg_color = colors.HexColor('#fff3e0')
        header_color = colors.HexColor('#ff9800')
    else:  # danger
        bg_color = colors.HexColor('#ffebee')
        header_color = colors.HexColor('#f44336')
    
    model_table = Table([
        ['Metric', 'Value'],
        ['Positive Cells (Ki-67+)', str(model_metrics['positive_cells'])],
        ['Negative Cells (Ki-67-)', str(model_metrics['negative_cells'])],
        ['Total Cells Detected', str(model_metrics['total_cells'])],
        ['Ki-67 Proliferation Index', f"{model_metrics['ki67_index']}%"],
        ['Diagnosis Classification', model_metrics['diagnosis']['classification']],
        ['Risk Level', model_metrics['diagnosis']['risk']],
        ['Malignancy Status', 'Malignant' if model_metrics['diagnosis'].get('malignant') else 'Non-Malignant']
    ], colWidths=[200, 280])
    model_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), header_color),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('BACKGROUND', (0, 1), (-1, -1), bg_color),
        ('GRID', (0, 0), (-1, -1), 0.75, colors.grey),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8)
    ]))
    story.append(model_table)
    story.append(Spacer(1, 18))

    if manual_metrics:
        manual_header = Paragraph('Manual (Ground Truth) Baseline', styles['Heading2'])
        story.append(manual_header)
        manual_table = Table([
            ['Positive Cells', manual_metrics['positive_cells']],
            ['Negative Cells', manual_metrics['negative_cells']],
            ['Total Cells', manual_metrics['total_cells']],
            ['Ki-67 Index', f"{manual_metrics['ki67_index']}%"],
            ['Classification', manual_metrics['diagnosis']['classification']],
            ['Risk Level', manual_metrics['diagnosis']['risk']]
        ], colWidths=[180, 300])
        manual_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#fff3e0')),
            ('GRID', (0, 0), (-1, -1), 0.25, colors.lightgrey)
        ]))
        story.append(manual_table)
        story.append(Spacer(1, 12))

    if qc:
        qc_header = Paragraph('Quality Control Summary', styles['Heading2'])
        story.append(qc_header)
        qc_table_data = [
            ['Flagged for Review', 'Yes' if qc['flagged'] else 'No'],
            ['Reason', qc.get('reason', '-')]
        ]
        for key, label in [
            ('positive_delta', 'Positive Cell Δ'),
            ('negative_delta', 'Negative Cell Δ'),
            ('total_delta', 'Total Cell Δ'),
            ('ki67_delta', 'Ki-67 Δ'),
            ('ki67_percent_delta', 'Ki-67 Δ (%)'),
        ]:
            value = qc['differences'].get(key)
            if value is not None:
                qc_table_data.append([label, value])

        qc_table_data.append(['Classification Match', 'Yes' if qc['differences'].get('classification_match') else 'No'])

        qc_table = Table(qc_table_data, colWidths=[200, 280])
        qc_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e8eaf6')),
            ('GRID', (0, 0), (-1, -1), 0.25, colors.lightgrey)
        ]))
        story.append(qc_table)
        story.append(Spacer(1, 12))

    # Clinical Interpretation Section
    interpretation = model_metrics['diagnosis']['interpretation']
    recommendation = model_metrics['diagnosis']['recommendation']
    
    story.append(Paragraph('Clinical Interpretation', styles['Heading2']))
    story.append(Spacer(1, 6))
    interp_style = ParagraphStyle(
        'InterpretationStyle',
        parent=styles['BodyText'],
        fontSize=10,
        leading=14,
        spaceAfter=12,
        leftIndent=10,
        rightIndent=10
    )
    story.append(Paragraph(interpretation, interp_style))
    story.append(Spacer(1, 12))
    
    story.append(Paragraph('Clinical Recommendation', styles['Heading2']))
    story.append(Spacer(1, 6))
    rec_style = ParagraphStyle(
        'RecommendationStyle',
        parent=styles['BodyText'],
        fontSize=10,
        leading=14,
        spaceAfter=12,
        leftIndent=10,
        rightIndent=10,
        textColor=colors.HexColor('#d84315')
    )
    story.append(Paragraph(recommendation, rec_style))
    story.append(Spacer(1, 18))
    
    # Cell Coordinates Summary
    cell_coords = analysis_data.get('cell_coordinates', {})
    if cell_coords:
        story.append(Paragraph('Detection Statistics', styles['Heading2']))
        story.append(Spacer(1, 6))
        
        coord_data = [
            ['Cell Type', 'Count', 'Percentage'],
            ['Ki-67 Positive Cells', 
             len(cell_coords.get('positive', [])),
             f"{(len(cell_coords.get('positive', [])) / model_metrics['total_cells'] * 100):.1f}%" if model_metrics['total_cells'] > 0 else '0%'],
            ['Ki-67 Negative Cells', 
             len(cell_coords.get('negative', [])),
             f"{(len(cell_coords.get('negative', [])) / model_metrics['total_cells'] * 100):.1f}%" if model_metrics['total_cells'] > 0 else '0%']
        ]
        
        coord_table = Table(coord_data, colWidths=[200, 140, 140])
        coord_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1976d2')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#e3f2fd')),
            ('GRID', (0, 0), (-1, -1), 0.75, colors.grey),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8)
        ]))
        story.append(coord_table)
        story.append(Spacer(1, 18))
    
    # Reference Information
    story.append(Paragraph('Reference Information', styles['Heading2']))
    story.append(Spacer(1, 6))
    ref_text = """
    <b>Ki-67 Proliferation Index Interpretation Guidelines:</b><br/>
    • &lt;5%: Very Low Proliferation (Benign)<br/>
    • 5-10%: Low Proliferation (Low Malignant Potential)<br/>
    • 10-20%: Moderate Proliferation (Borderline Malignant)<br/>
    • &gt;20%: High Proliferation (Malignant)<br/>
    <br/>
    <b>Note:</b> This report is generated by an AI-assisted analysis system. 
    Results should be reviewed by a qualified pathologist and correlated with clinical findings.
    """
    ref_style = ParagraphStyle(
        'ReferenceStyle',
        parent=styles['BodyText'],
        fontSize=8,
        leading=11,
        textColor=colors.HexColor('#555555'),
        leftIndent=10,
        rightIndent=10
    )
    story.append(Paragraph(ref_text, ref_style))
    story.append(Spacer(1, 18))
    
    # Footer
    footer_style = ParagraphStyle(
        'FooterStyle',
        parent=styles['BodyText'],
        fontSize=8,
        textColor=colors.grey,
        alignment=1  # Center
    )
    story.append(Paragraph(
        f'Report Generated: {analysis_data.get("timestamp", "-")} | Analysis ID: {analysis_data.get("analysis_id", "-")}',
        footer_style
    ))
    story.append(Paragraph(
        'Ki-67 Medical Diagnostic System | Confidential Medical Report',
        footer_style
    ))

    doc.build(story)


def generate_csv_export(analysis_data: Dict[str, Any], csv_path: str) -> None:
    """Generate CSV summary for the analysis."""
    ensure_results_directory(csv_path)

    model_metrics = analysis_data['results']
    manual_metrics = analysis_data.get('comparison', {}).get('manual') or {}
    qc = analysis_data.get('comparison', {}).get('quality_control', {})
    diffs = qc.get('differences', {})

    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Field', 'Model Output', 'Manual Baseline', 'Difference'])
        writer.writerow(['Positive Cells',
                         model_metrics['positive_cells'],
                         manual_metrics.get('positive_cells', ''),
                         diffs.get('positive_delta', '')])
        writer.writerow(['Negative Cells',
                         model_metrics['negative_cells'],
                         manual_metrics.get('negative_cells', ''),
                         diffs.get('negative_delta', '')])
        writer.writerow(['Total Cells',
                         model_metrics['total_cells'],
                         manual_metrics.get('total_cells', ''),
                         diffs.get('total_delta', '')])
        writer.writerow(['Ki-67 Index (%)',
                         model_metrics['ki67_index'],
                         manual_metrics.get('ki67_index', ''),
                         diffs.get('ki67_delta', '')])
        writer.writerow(['Classification',
                         model_metrics['diagnosis']['classification'],
                         manual_metrics.get('diagnosis', {}).get('classification', ''),
                         'Match' if diffs.get('classification_match') else 'Mismatch' if manual_metrics else '' ])
        writer.writerow(['Quality Flagged',
                         qc.get('flagged', ''),
                         '',
                         qc.get('reason', '')])


def perform_analysis_pipeline(image_path: str,
                              unique_id: str,
                              patient_data: Dict[str, str],
                              original_filename: str,
                              manual_metrics_override: Optional[Dict[str, Any]] = None,
                              dataset_subset_hint: Optional[str] = None) -> Dict[str, Any]:
    """Run the end-to-end analysis flow on an image path."""
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise ValueError('Unable to load the provided image for analysis.')

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    original_image = image_rgb.copy()

    # Apply transforms
    transformed = transform(image=image_rgb)
    image_tensor = transformed['image'].unsqueeze(0).to(device)

    # Run model inference
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        pred_heatmaps = torch.sigmoid(outputs).squeeze(0).cpu().numpy()

    pred_pos_points, pred_neg_points = predict_points_from_heatmaps(pred_heatmaps)

    vis_image = create_visualization(original_image, pred_pos_points, pred_neg_points)

    positive_count = len(pred_pos_points)
    negative_count = len(pred_neg_points)
    total_count = positive_count + negative_count
    ki67_index = calculate_ki67_index(positive_count, negative_count)
    diagnosis = get_diagnosis(ki67_index)

    model_metrics = {
        'positive_cells': positive_count,
        'negative_cells': negative_count,
        'total_cells': total_count,
        'ki67_index': round(ki67_index, 2),
        'diagnosis': diagnosis
    }

    gt_pos_points = np.array([])
    gt_neg_points = np.array([])
    annotation_subset = None

    if manual_metrics_override is None:
        gt_pos_points, gt_neg_points, annotation_subset = load_ground_truth_points(
            original_filename,
            subset_hint=dataset_subset_hint
        )
        manual_metrics = calculate_manual_metrics(gt_pos_points, gt_neg_points)
    else:
        manual_metrics = manual_metrics_override

    result_filename = f"result_{unique_id}.png"
    result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)
    cv2.imwrite(result_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))

    gt_overlay_b64 = None
    comparison_overlay_b64 = None
    if manual_metrics and (gt_pos_points.size > 0 or gt_neg_points.size > 0):
        gt_overlay = create_ground_truth_visualization(original_image, gt_pos_points, gt_neg_points)
        comparison_overlay = create_comparison_visualization(
            original_image,
            pred_pos_points,
            pred_neg_points,
            gt_pos_points,
            gt_neg_points
        )
        gt_overlay_b64 = image_to_base64(gt_overlay)
        comparison_overlay_b64 = image_to_base64(comparison_overlay)

    response_data = {
        'success': True,
        'analysis_id': unique_id,
        'timestamp': datetime.now().isoformat(),
        'patient_data': patient_data,
        'results': model_metrics,
        'images': {
            'original': image_to_base64(original_image),
            'analyzed': image_to_base64(vis_image),
            'ground_truth': gt_overlay_b64,
            'comparison_overlay': comparison_overlay_b64
        },
        'cell_coordinates': {
            'positive': pred_pos_points.tolist() if len(pred_pos_points) > 0 else [],
            'negative': pred_neg_points.tolist() if len(pred_neg_points) > 0 else [],
            'ground_truth_positive': gt_pos_points.tolist() if gt_pos_points.size > 0 else [],
            'ground_truth_negative': gt_neg_points.tolist() if gt_neg_points.size > 0 else []
        }
    }

    quality_control = evaluate_quality(model_metrics, manual_metrics)
    response_data['comparison'] = {
        'manual': manual_metrics,
        'quality_control': quality_control,
        'annotation_subset': annotation_subset
    }

    return response_data


def persist_analysis_results(response_data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate reports and persist the analysis JSON to disk."""
    unique_id = response_data['analysis_id']

    pdf_path = os.path.join(app.config['RESULTS_FOLDER'], f"analysis_{unique_id}.pdf")
    csv_path = os.path.join(app.config['RESULTS_FOLDER'], f"analysis_{unique_id}.csv")

    try:
        generate_pdf_report(response_data, pdf_path)
        generate_csv_export(response_data, csv_path)
    except Exception as report_error:
        print(f"⚠️ Failed to generate reports for {unique_id}: {report_error}")

    response_data['reports'] = {
        'pdf': f"/api/report/pdf/{unique_id}",
        'csv': f"/api/report/csv/{unique_id}"
    }

    analysis_file = os.path.join(app.config['RESULTS_FOLDER'], f"analysis_{unique_id}.json")
    with open(analysis_file, 'w') as f:
        json.dump(response_data, f, indent=2)

    return response_data


def persist_analysis_to_db(response_data: Dict[str, Any], upload_path: str, saved_filename: str, original_filename: str) -> None:
    """Store upload + analysis metadata into SQLite database."""
    db: Session = SessionLocal()
    try:
        # Upload record
        patient = response_data.get('patient_data', {})
        upload = Upload(
            original_filename=original_filename,
            saved_filename=saved_filename,
            upload_path=upload_path,
            patient_id=patient.get('patient_id', ''),
            patient_name=patient.get('patient_name', ''),
            age=patient.get('age', ''),
            gender=patient.get('gender', ''),
            contact=patient.get('contact', ''),
            exam_date=patient.get('exam_date', ''),
            physician=patient.get('physician', ''),
            clinical_notes=patient.get('clinical_notes', ''),
        )
        db.add(upload)
        db.flush()  # get upload.id

        # Model metrics
        model_metrics = response_data.get('results', {})
        diagnosis = model_metrics.get('diagnosis', {})
        qc = (response_data.get('comparison') or {}).get('quality_control', {})
        diffs = (qc or {}).get('differences', {})

        analysis_id = response_data.get('analysis_id', '')
        pdf_path = os.path.join(app.config['RESULTS_FOLDER'], f"analysis_{analysis_id}.pdf")
        csv_path = os.path.join(app.config['RESULTS_FOLDER'], f"analysis_{analysis_id}.csv")
        result_image_path = os.path.join(app.config['RESULTS_FOLDER'], f"result_{analysis_id}.png")

        dataset_info = response_data.get('dataset', {})

        analysis = AnalysisResult(
            analysis_id=analysis_id,
            upload_id=upload.id,
            positive_cells=int(model_metrics.get('positive_cells', 0)),
            negative_cells=int(model_metrics.get('negative_cells', 0)),
            total_cells=int(model_metrics.get('total_cells', 0)),
            ki67_index=float(model_metrics.get('ki67_index', 0.0)),
            classification=str(diagnosis.get('classification', '')),
            risk=str(diagnosis.get('risk', '')),
            malignant=bool(diagnosis.get('malignant', False)),
            qc_available=bool(qc.get('available', False)) if qc else False,
            qc_flagged=bool(qc.get('flagged', False)) if qc else False,
            qc_reason=str(qc.get('reason', '')) if qc else '',
            qc_ki67_percent_delta=float(diffs.get('ki67_percent_delta', 0.0)) if diffs else 0.0,
            qc_classification_match=bool(diffs.get('classification_match', True)) if diffs else True,
            pdf_path=pdf_path,
            csv_path=csv_path,
            result_image_path=result_image_path,
            dataset_source=str(dataset_info.get('source', '')),
            dataset_subset=str(dataset_info.get('subset', '')),
            dataset_image_name=str(dataset_info.get('image_name', '')),
        )

        db.add(analysis)
        db.commit()
    except Exception as exc:
        db.rollback()
        print(f"⚠️ DB persist error: {exc}")
    finally:
        db.close()


def create_visualization(image, pos_points, neg_points):
    """Create visualization with detected cells marked"""
    vis_image = image.copy()
    
    # Draw positive cells (red circles)
    for point in pos_points:
        y, x = int(point[0]), int(point[1])
        cv2.circle(vis_image, (x, y), 5, (255, 0, 0), 2)
        cv2.circle(vis_image, (x, y), 2, (255, 255, 255), -1)
    
    # Draw negative cells (blue circles)
    for point in neg_points:
        y, x = int(point[0]), int(point[1])
        cv2.circle(vis_image, (x, y), 5, (0, 0, 255), 2)
        cv2.circle(vis_image, (x, y), 2, (255, 255, 255), -1)
    
    return vis_image


def calculate_ki67_index(positive_count, negative_count):
    """Calculate Ki-67 proliferation index"""
    total = positive_count + negative_count
    if total == 0:
        return 0.0
    return (positive_count / total) * 100


def get_diagnosis(ki67_index):
    """Get diagnosis based on Ki-67 index"""
    if ki67_index >= 30:
        return {
            'classification': 'Malignant',
            'risk': 'High Risk',
            'interpretation': 'Ki-67 index ≥ 30% indicates high proliferative activity, consistent with malignant characteristics. Requires aggressive treatment protocols and close monitoring.',
            'recommendation': 'Immediate oncological consultation recommended. Consider chemotherapy, targeted therapy, and surgical intervention. Regular follow-up essential.',
            'color': 'danger',
            'malignant': True
        }
    elif ki67_index >= 20:
        return {
            'classification': 'Borderline Malignant',
            'risk': 'Moderate-High Risk',
            'interpretation': 'Ki-67 index 20-30% suggests intermediate to high proliferative activity. Further clinical correlation and additional molecular testing recommended.',
            'recommendation': 'Multidisciplinary review advised. Consider adjuvant therapy. Additional IHC markers and molecular profiling may be beneficial.',
            'color': 'warning',
            'malignant': True
        }
    elif ki67_index >= 14:
        return {
            'classification': 'Low Malignant Potential',
            'risk': 'Moderate Risk',
            'interpretation': 'Ki-67 index 14-20% indicates moderate proliferative activity. Clinical correlation with histological grade and other prognostic factors essential.',
            'recommendation': 'Standard treatment protocols. Regular monitoring recommended. Consider endocrine therapy if hormone receptor positive.',
            'color': 'warning',
            'malignant': False
        }
    else:
        return {
            'classification': 'Benign',
            'risk': 'Low Risk',
            'interpretation': 'Ki-67 index < 14% indicates low proliferative activity, consistent with benign characteristics and favorable prognosis.',
            'recommendation': 'Continue standard monitoring. Consider less intensive treatment protocols. Regular clinical follow-up advised.',
            'color': 'success',
            'malignant': False
        }


def image_to_base64(image):
    """Convert numpy image to base64 string with data URI"""
    _, buffer = cv2.imencode('.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    base64_str = base64.b64encode(buffer).decode('utf-8')
    return f'data:image/png;base64,{base64_str}'


# ============================================================================
# ROUTES
# ============================================================================

@app.route('/')
def index():
    """Serve React frontend"""
    return send_file('../frontend-react/dist/index.html')

@app.route('/<path:path>')
def serve_static(path):
    """Serve React static files"""
    from flask import send_from_directory
    return send_from_directory('../frontend-react/dist', path)


@app.route('/api/analyze', methods=['POST'])
def analyze():
    """Analyze uploaded image"""
    try:
        # Check if file is present
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Allowed: PNG, JPG, JPEG, TIF, TIFF'}), 400
        
        manual_metrics_override = manual_metrics_from_counts(
            request.form.get('manual_positive'),
            request.form.get('manual_negative')
        )
        dataset_subset_hint = request.form.get('dataset_subset')

        # Get patient information
        patient_data = {
            'patient_id': request.form.get('patient_id', ''),
            'patient_name': request.form.get('patient_name', ''),
            'age': request.form.get('age', ''),
            'gender': request.form.get('gender', ''),
            'contact': request.form.get('contact', ''),
            'exam_date': request.form.get('exam_date', ''),
            'physician': request.form.get('physician', ''),
            'clinical_notes': request.form.get('clinical_notes', '')
        }
        
        # Save uploaded file
        original_filename = secure_filename(file.filename)
        unique_id = str(uuid.uuid4())[:8]
        filename = f"{unique_id}_{original_filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        response_data = perform_analysis_pipeline(
            image_path=filepath,
            unique_id=unique_id,
            patient_data=patient_data,
            original_filename=original_filename,
            manual_metrics_override=manual_metrics_override,
            dataset_subset_hint=dataset_subset_hint
        )
        response_data = persist_analysis_results(response_data)

        # Persist to DB
        try:
            persist_analysis_to_db(response_data, upload_path=filepath, saved_filename=filename, original_filename=original_filename)
        except Exception as _:
            pass

        return jsonify(response_data)
    
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500


@app.route('/api/analyze/dataset', methods=['POST'])
def analyze_dataset_case():
    """Run analysis on an internally curated dataset image (with ground truth)."""
    try:
        payload = request.get_json(force=True) or {}
        subset = payload.get('subset', 'test')
        subset_path = os.path.join(DATASET_IMAGE_ROOT, subset)

        if not os.path.isdir(subset_path):
            return jsonify({'error': f'Dataset subset not found: {subset}'}), 404

        image_name = payload.get('image_name')
        available_images = [
            f for f in os.listdir(subset_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))
        ]

        if not available_images:
            return jsonify({'error': f'No images available in subset: {subset}'}), 404

        if not image_name:
            image_name = random.choice(available_images)
        elif image_name not in available_images:
            return jsonify({'error': f'Image {image_name} not found in subset {subset}'}), 404

        image_path = os.path.join(subset_path, image_name)

        unique_id = str(uuid.uuid4())[:8]

        patient_data = payload.get('patient_data') or {
            'patient_id': payload.get('patient_id', f'DS-{unique_id}'),
            'patient_name': payload.get('patient_name', 'Dataset Reference Case'),
            'age': payload.get('age', 'N/A'),
            'gender': payload.get('gender', 'N/A'),
            'contact': payload.get('contact', ''),
            'exam_date': payload.get('exam_date', datetime.now().date().isoformat()),
            'physician': payload.get('physician', 'Automated Validation'),
            'clinical_notes': payload.get('clinical_notes', f'Automated dataset evaluation ({subset})')
        }

        response_data = perform_analysis_pipeline(
            image_path=image_path,
            unique_id=unique_id,
            patient_data=patient_data,
            original_filename=image_name,
            manual_metrics_override=None,
            dataset_subset_hint=subset
        )

        response_data['dataset'] = {
            'source': 'BCData',
            'subset': subset,
            'image_name': image_name
        }

        response_data = persist_analysis_results(response_data)

        # Persist to DB
        try:
            # For dataset cases, there is no actual uploaded file path; use image_path.
            # Reconstruct path based on subset and image name used above.
            dataset_image_path = os.path.join(DATASET_IMAGE_ROOT, subset, image_name)
            persist_analysis_to_db(response_data,
                                   upload_path=dataset_image_path,
                                   saved_filename=image_name,
                                   original_filename=image_name)
        except Exception as _:
            pass

        return jsonify(response_data)

    except Exception as e:
        print(f"Error during dataset analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Dataset analysis failed: {str(e)}'}), 500


@app.route('/api/analyze/batch', methods=['POST'])
def analyze_batch_cases():
    """Batch process multiple dataset images with automated reporting."""
    try:
        payload = request.get_json(force=True) or {}
        subset = payload.get('subset', 'test')
        subset_path = os.path.join(DATASET_IMAGE_ROOT, subset)

        if not os.path.isdir(subset_path):
            return jsonify({'error': f'Dataset subset not found: {subset}'}), 404

        available_images = [
            f for f in os.listdir(subset_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))
        ]

        if not available_images:
            return jsonify({'error': f'No images available in subset: {subset}'}), 404

        requested_images = payload.get('image_names')
        if requested_images:
            selected_images = [name for name in requested_images if name in available_images]
            if not selected_images:
                return jsonify({'error': 'None of the requested images are available in the subset.'}), 404
        else:
            limit = int(payload.get('limit', 10))
            limit = max(limit, 1)
            selected_images = sorted(available_images)[:limit]

        batch_id = str(uuid.uuid4())[:8]
        batch_results: List[Dict[str, Any]] = []
        flagged_cases = 0
        matched_cases = 0
        manual_available_cases = 0

        for image_name in selected_images:
            unique_id = str(uuid.uuid4())[:8]
            image_path = os.path.join(subset_path, image_name)

            patient_data = {
                'patient_id': f'{batch_id}-{unique_id}',
                'patient_name': payload.get('patient_name', 'Batch Dataset Case'),
                'age': payload.get('age', 'N/A'),
                'contact': payload.get('contact', ''),
                'exam_date': payload.get('exam_date', datetime.now().date().isoformat()),
                'physician': payload.get('physician', 'Automated Batch Evaluation'),
                'clinical_notes': payload.get('clinical_notes', f'Batch QC case ({subset})')
            }

            analysis_result = perform_analysis_pipeline(
                image_path=image_path,
                unique_id=unique_id,
                patient_data=patient_data,
                original_filename=image_name,
                manual_metrics_override=None,
                dataset_subset_hint=subset
            )

            analysis_result['dataset'] = {
                'source': 'BCData',
                'subset': subset,
                'image_name': image_name
            }

            analysis_result = persist_analysis_results(analysis_result)

            try:
                persist_analysis_to_db(analysis_result,
                                       upload_path=image_path,
                                       saved_filename=image_name,
                                       original_filename=image_name)
            except Exception as _:
                pass

            qc = analysis_result['comparison']['quality_control']
            if qc.get('flagged'):
                flagged_cases += 1
            if qc.get('available'):
                manual_available_cases += 1
                if qc['differences'].get('classification_match'):
                    matched_cases += 1

            batch_results.append(analysis_result)

        summary_accuracy = round((matched_cases / manual_available_cases) * 100, 2) if manual_available_cases else None

        batch_summary = {
            'batch_id': batch_id,
            'subset': subset,
            'cases_processed': len(batch_results),
            'flagged_cases': flagged_cases,
            'manual_reference_cases': manual_available_cases,
            'classification_accuracy_percent': summary_accuracy
        }

        # Create batch CSV summary
        batch_csv_path = os.path.join(app.config['RESULTS_FOLDER'], f'batch_{batch_id}.csv')
        ensure_results_directory(batch_csv_path)
        with open(batch_csv_path, 'w', newline='') as batch_csv:
            writer = csv.writer(batch_csv)
            writer.writerow([
                'analysis_id', 'subset', 'image_name',
                'model_positive', 'model_negative', 'model_total', 'model_ki67', 'model_classification',
                'manual_positive', 'manual_negative', 'manual_total', 'manual_ki67', 'manual_classification',
                'ki67_delta', 'classification_match', 'quality_flagged'
            ])
            for case in batch_results:
                model_metrics = case['results']
                manual_metrics = case['comparison'].get('manual') or {}
                qc = case['comparison']['quality_control']
                diffs = qc.get('differences', {}) if qc else {}

                writer.writerow([
                    case['analysis_id'],
                    case.get('dataset', {}).get('subset'),
                    case.get('dataset', {}).get('image_name'),
                    model_metrics['positive_cells'],
                    model_metrics['negative_cells'],
                    model_metrics['total_cells'],
                    model_metrics['ki67_index'],
                    model_metrics['diagnosis']['classification'],
                    manual_metrics.get('positive_cells', ''),
                    manual_metrics.get('negative_cells', ''),
                    manual_metrics.get('total_cells', ''),
                    manual_metrics.get('ki67_index', ''),
                    manual_metrics.get('diagnosis', {}).get('classification', ''),
                    diffs.get('ki67_delta', ''),
                    diffs.get('classification_match', ''),
                    qc.get('flagged') if qc else ''
                ])

        response_payload = {
            'summary': batch_summary,
            'results': [
                {
                    'analysis_id': case['analysis_id'],
                    'dataset': case.get('dataset'),
                    'results': case['results'],
                    'comparison': case['comparison'],
                    'reports': case['reports']
                }
                for case in batch_results
            ],
            'reports': {
                'batch_csv': f"/api/report/batch/{batch_id}"
            }
        }

        return jsonify(response_payload)

    except Exception as e:
        print(f"Error during batch analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Batch analysis failed: {str(e)}'}), 500


@app.route('/api/analyze/batch-upload', methods=['POST'])
def analyze_batch_upload():
    """Batch process multiple uploaded image files."""
    try:
        if 'images' not in request.files:
            return jsonify({'error': 'No images provided'}), 400

        uploaded_files = request.files.getlist('images')
        if not uploaded_files:
            return jsonify({'error': 'No images uploaded'}), 400

        patient_prefix = request.form.get('patientPrefix', 'BATCH')
        batch_id = str(uuid.uuid4())[:8]
        batch_results: List[Dict[str, Any]] = []

        for idx, uploaded_file in enumerate(uploaded_files):
            if not uploaded_file or uploaded_file.filename == '':
                continue

            filename = secure_filename(uploaded_file.filename)
            unique_id = str(uuid.uuid4())[:8]
            
            # Save the uploaded file
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{unique_id}_{filename}")
            uploaded_file.save(file_path)

            patient_data = {
                'patient_id': f'{patient_prefix}-{idx+1:03d}',
                'patient_name': f'{patient_prefix} Case {idx+1}',
                'age': 'N/A',
                'contact': '',
                'exam_date': datetime.now().date().isoformat(),
                'physician': 'Batch Analysis',
                'clinical_notes': f'Batch upload - File: {filename}'
            }

            # Perform analysis
            analysis_result = perform_analysis_pipeline(
                image_path=file_path,
                unique_id=unique_id,
                patient_data=patient_data,
                original_filename=filename,
                manual_metrics_override=None,
                dataset_subset_hint=None
            )

            analysis_result['batch_info'] = {
                'batch_id': batch_id,
                'file_index': idx + 1,
                'original_filename': filename
            }

            analysis_result = persist_analysis_results(analysis_result)

            try:
                persist_analysis_to_db(analysis_result,
                                       upload_path=file_path,
                                       saved_filename=f"{unique_id}_{filename}",
                                       original_filename=filename)
            except Exception as _:
                pass
            batch_results.append(analysis_result)

        # Generate batch summary
        total_cases = len(batch_results)
        avg_ki67 = sum(r['results']['ki67_index'] for r in batch_results) / total_cases if total_cases > 0 else 0
        
        classifications = {}
        for result in batch_results:
            cls = result['results']['diagnosis']['classification']
            classifications[cls] = classifications.get(cls, 0) + 1

        batch_summary = {
            'batch_id': batch_id,
            'timestamp': datetime.now().isoformat(),
            'total_cases': total_cases,
            'average_ki67': round(avg_ki67, 2),
            'classifications': classifications
        }

        response_payload = {
            'success': True,
            'summary': batch_summary,
            'results': [
                {
                    'analysis_id': case['analysis_id'],
                    'batch_info': case.get('batch_info'),
                    'patient_data': case['patient_data'],
                    'results': case['results'],
                    'images': case['images'],
                    'cell_coordinates': case['cell_coordinates'],
                    'reports': case.get('reports')
                }
                for case in batch_results
            ]
        }

        return jsonify(response_payload)

    except Exception as e:
        print(f"Error during batch upload analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Batch upload analysis failed: {str(e)}'}), 500


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(device)
    })


@app.route('/api/report/pdf/<analysis_id>', methods=['GET'])
def download_pdf_report(analysis_id: str):
    pdf_path = os.path.join(app.config['RESULTS_FOLDER'], f"analysis_{analysis_id}.pdf")
    if not os.path.exists(pdf_path):
        return jsonify({'error': 'Report not found'}), 404

    return send_file(pdf_path,
                     mimetype='application/pdf',
                     as_attachment=True,
                     download_name=f"Ki67_Report_{analysis_id}.pdf")


@app.route('/api/report/csv/<analysis_id>', methods=['GET'])
def download_csv_report(analysis_id: str):
    csv_path = os.path.join(app.config['RESULTS_FOLDER'], f"analysis_{analysis_id}.csv")
    if not os.path.exists(csv_path):
        return jsonify({'error': 'CSV summary not found'}), 404

    return send_file(csv_path,
                     mimetype='text/csv',
                     as_attachment=True,
                     download_name=f"Ki67_Summary_{analysis_id}.csv")


@app.route('/api/report/batch/<batch_id>', methods=['GET'])
def download_batch_summary(batch_id: str):
    csv_path = os.path.join(app.config['RESULTS_FOLDER'], f"batch_{batch_id}.csv")
    if not os.path.exists(csv_path):
        return jsonify({'error': 'Batch summary not found'}), 404

    return send_file(csv_path,
                     mimetype='text/csv',
                     as_attachment=True,
                     download_name=f"Ki67_Batch_Summary_{batch_id}.csv")


@app.route('/api/history', methods=['GET'])
def history():
    """Return paginated recent analyses with basic metadata."""
    try:
        page = max(int(request.args.get('page', 1)), 1)
        page_size = min(max(int(request.args.get('page_size', 10)), 1), 100)
        offset = (page - 1) * page_size

        db: Session = SessionLocal()
        try:
            total = db.query(AnalysisResult).count()
            rows = (db.query(AnalysisResult)
                      .order_by(AnalysisResult.created_at.desc())
                      .offset(offset)
                      .limit(page_size)
                      .all())

            items = []
            for r in rows:
                u = r.upload
                items.append({
                    'analysis_id': r.analysis_id,
                    'created_at': r.created_at.isoformat(),
                    'patient_name': u.patient_name if u else '',
                    'patient_id': u.patient_id if u else '',
                    'original_filename': u.original_filename if u else '',
                    'saved_filename': u.saved_filename if u else '',
                    'metrics': {
                        'positive_cells': r.positive_cells,
                        'negative_cells': r.negative_cells,
                        'total_cells': r.total_cells,
                        'ki67_index': r.ki67_index,
                        'classification': r.classification,
                        'risk': r.risk,
                        'malignant': r.malignant,
                    },
                    'qc': {
                        'available': r.qc_available,
                        'flagged': r.qc_flagged,
                        'reason': r.qc_reason,
                        'ki67_percent_delta': r.qc_ki67_percent_delta,
                        'classification_match': r.qc_classification_match,
                    },
                    'reports': {
                        'pdf': f"/api/report/pdf/{r.analysis_id}",
                        'csv': f"/api/report/csv/{r.analysis_id}"
                    },
                    'images': {
                        'result': f"/results/result_{r.analysis_id}.png"
                    }
                })

            return jsonify({
                'page': page,
                'page_size': page_size,
                'total': total,
                'items': items
            })
        finally:
            db.close()
    except Exception as e:
        print(f"History error: {e}")
        return jsonify({'error': 'Failed to fetch history'}), 500


# ============================================================================
# INITIALIZATION
# ============================================================================

def initialize_model():
    """Load the trained model"""
    global model, device, transform
    
    print("🔬 Initializing Ki-67 Detection System...")
    
    CHECKPOINT_PATH = 'models/ki67-point-epoch=68-val_peak_f1_avg=0.8503.ckpt'
    
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"❌ Error: Checkpoint not found at {CHECKPOINT_PATH}")
        print("Please ensure the model checkpoint is in the root directory.")
        return False
    
    try:
        # Load model
        print("Loading model...")
        model = ImprovedKi67PointDetectionModel.load_from_checkpoint(CHECKPOINT_PATH)
        
        # Setup device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        
        # Setup transforms
        transform = get_validation_augmentation()
        
        print(f"✅ Model loaded successfully on {device}")
        return True
    
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
    
    # Initialize DB (create tables if missing)
    try:
        Base.metadata.create_all(bind=engine)
        print("🗄️  SQLite DB initialized.")
    except Exception as exc:
        print(f"⚠️ DB init error: {exc}")

    # Initialize model
    if not initialize_model():
        print("❌ Failed to initialize model. Exiting.")
        exit(1)
    
    print("\n" + "="*60)
    print("🏥 Ki-67 Medical Diagnostic System")
    print("="*60)
    print(f"🌐 Server running at: http://localhost:5000")
    print(f"📁 Upload folder: {app.config['UPLOAD_FOLDER']}")
    print(f"📁 Results folder: {app.config['RESULTS_FOLDER']}")
    print(f"🔬 Model device: {device}")
    print("="*60)
    print("\n🚀 Ready to accept requests!\n")
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5001, debug=False)
