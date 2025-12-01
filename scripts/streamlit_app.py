"""
Ki-67 Malignancy Classification System - Streamlit Interface
============================================================
Professional medical dashboard for breast cancer tissue analysis
"""

import streamlit as st
import requests
import io
import base64
from PIL import Image
import pandas as pd
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import time
import json
import os

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Ki-67 Analysis System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS STYLING
# ============================================================================

def load_custom_css():
    """Apply custom CSS for professional medical dashboard aesthetic"""
    st.markdown("""
    <style>
    /* Global Styles */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }
    
    /* Card Styles */
    .upload-card, .results-card {
        background: white;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    
    /* Header Styles */
    .section-header {
        display: flex;
        align-items: center;
        gap: 12px;
        color: #3B82F6;
        font-size: 20px;
        font-weight: bold;
        margin-bottom: 20px;
    }
    
    /* Info Box */
    .info-box {
        background: #FEF3C7;
        border-left: 4px solid #F59E0B;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 20px;
    }
    
    .info-box-title {
        font-weight: 600;
        color: #92400E;
        margin-bottom: 8px;
    }
    
    .info-box-content {
        color: #78350F;
        font-size: 14px;
        line-height: 1.6;
    }
    
    /* Upload Zone */
    .upload-zone {
        border: 3px dashed #CBD5E1;
        border-radius: 12px;
        padding: 40px;
        text-align: center;
        background: #F8FAFC;
        transition: all 0.3s ease;
        margin: 20px 0;
    }
    
    .upload-zone:hover {
        border-color: #7C3AED;
        background: #FAF5FF;
    }
    
    /* Metric Card */
    .metric-card {
        background: linear-gradient(135deg, #7C3AED 0%, #5B21B6 100%);
        border-radius: 16px;
        padding: 24px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 12px rgba(124, 58, 237, 0.3);
        transition: transform 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 16px rgba(124, 58, 237, 0.4);
    }
    
    .metric-label {
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 1px;
        opacity: 0.9;
        margin-bottom: 8px;
    }
    
    .metric-value {
        font-size: 48px;
        font-weight: bold;
        line-height: 1;
        margin: 12px 0;
    }
    
    .metric-sublabel {
        font-size: 14px;
        opacity: 0.8;
    }
    
    /* Diagnosis Badge */
    .diagnosis-badge {
        display: inline-block;
        padding: 16px 40px;
        border-radius: 50px;
        font-size: 24px;
        font-weight: bold;
        margin: 20px 0;
        text-align: center;
    }
    
    .diagnosis-benign {
        background: #10B981;
        color: white;
    }
    
    .diagnosis-malignant {
        background: #EF4444;
        color: white;
    }
    
    /* Button Styles */
    .stButton > button {
        width: 100%;
        background: #7C3AED;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: #6D28D9;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(124, 58, 237, 0.4);
    }
    
    /* History Card */
    .history-card {
        background: white;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        transition: all 0.2s ease;
    }
    
    .history-card:hover {
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
        transform: translateY(-2px);
    }
    
    .history-card-benign {
        border-left: 4px solid #10B981;
    }
    
    .history-card-malignant {
        border-left: 4px solid #EF4444;
    }
    
    /* Print Styles */
    @media print {
        .main {
            background: white !important;
        }
        .stButton, .stDownloadButton {
            display: none !important;
        }
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .metric-value {
            font-size: 36px;
        }
        .diagnosis-badge {
            font-size: 18px;
            padding: 12px 24px;
        }
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def init_session_state():
    """Initialize session state variables"""
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []
    
    if 'current_results' not in st.session_state:
        st.session_state.current_results = None
    
    if 'batch_files' not in st.session_state:
        st.session_state.batch_files = []
    
    if 'batch_results' not in st.session_state:
        st.session_state.batch_results = []

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def validate_image(uploaded_file):
    """Validate uploaded image file"""
    if uploaded_file is None:
        return False, "No file uploaded"
    
    # Check file type
    if uploaded_file.type not in ['image/png', 'image/jpeg', 'image/jpg']:
        return False, "❌ Invalid file type. Please upload PNG or JPG."
    
    # Check file size (max 16MB)
    if uploaded_file.size > 16 * 1024 * 1024:
        return False, "❌ File too large. Maximum size is 16MB."
    
    return True, "Valid"

def format_file_size(size_bytes):
    """Format file size in human-readable format"""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"

def analyze_image(image, patient_id="", manual_pos=None, manual_neg=None, notes=""):
    """Send image to Flask backend for analysis"""
    try:
        # Convert image to bytes
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        # Create form data
        files = {'image': ('image.png', img_bytes, 'image/png')}
        data = {
            'patient_id': patient_id if patient_id else 'Unknown',
            'patient_name': 'Streamlit User',
            'age': '0',
            'gender': 'other',
            'exam_date': datetime.now().strftime('%Y-%m-%d'),
            'contact': '',
            'physician': '',
            'clinical_notes': notes
        }
        
        # POST to Flask backend
        response = requests.post('http://localhost:5001/api/analyze', files=files, data=data, timeout=60)
        
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, f"Server error: {response.status_code}"
    
    except requests.exceptions.ConnectionError:
        return False, "❌ Unable to connect to server. Please ensure the backend is running on port 5001."
    except requests.exceptions.Timeout:
        return False, "❌ Analysis timeout. Please try again."
    except Exception as e:
        return False, f"❌ Error: {str(e)}"

def save_to_history(results, patient_id, filename):
    """Save analysis results to history"""
    history_entry = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'patient_id': patient_id,
        'filename': filename,
        'results': results,
        'analysis_id': results.get('analysis_id', 'N/A')
    }
    st.session_state.analysis_history.insert(0, history_entry)

# ============================================================================
# TAB 1: SINGLE ANALYSIS
# ============================================================================

def render_single_analysis():
    """Render single image analysis tab"""
    
    # Create two-column layout
    left_col, right_col = st.columns([4, 6], gap="large")
    
    # ========================================================================
    # LEFT COLUMN: Upload Section
    # ========================================================================
    with left_col:
        # Upload Card Container
        st.markdown('<div class="upload-card">', unsafe_allow_html=True)
        
        # Section 1: Header
        st.markdown("""
        <div class="section-header">
            <span style="font-size: 32px;">🔬</span>
            <span>Upload Patch</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Section 2: Instructions Box
        st.markdown("""
        <div class="info-box">
            <div class="info-box-title">ℹ️ Instructions</div>
            <div class="info-box-content">
                • Upload a breast tissue patch image (PNG/JPG)<br>
                • Optionally enter patient ID<br>
                • For validation: enter manual cell counts<br>
                • Click "Analyze Patch" to get AI predictions
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Section 3: Patient ID Input
        st.markdown("### 👤 Patient ID (Optional)")
        patient_id = st.text_input(
            "Patient ID",
            placeholder="e.g., P-2024-001",
            label_visibility="collapsed"
        )
        
        # Section 4: Image Upload Area
        st.markdown("### 📁 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a histology image (PNG, JPG, or JPEG, max 16MB)",
            label_visibility="collapsed"
        )
        
        # Section 5: Image Preview
        if uploaded_file is not None:
            is_valid, message = validate_image(uploaded_file)
            if is_valid:
                image = Image.open(uploaded_file)
                st.image(image, caption=f"Selected: {uploaded_file.name}", use_container_width=True)
                st.info(f"📊 Size: {format_file_size(uploaded_file.size)}")
            else:
                st.error(message)
                uploaded_file = None
        
        # Section 6: Manual Counts (Optional)
        st.markdown("### 🔢 Manual Counts (Optional - for validation)")
        col1, col2 = st.columns(2)
        with col1:
            manual_pos = st.number_input("Positive cells", min_value=0, value=0, step=1)
        with col2:
            manual_neg = st.number_input("Negative cells", min_value=0, value=0, step=1)
        
        # Section 7: Notes Section
        st.markdown("### 📝 Notes (Optional)")
        notes = st.text_area(
            "Clinical notes",
            placeholder="Add any clinical notes...",
            height=100,
            label_visibility="collapsed"
        )
        
        # Section 8: Analyze Button
        st.markdown("<br>", unsafe_allow_html=True)
        analyze_btn = st.button(
            "🔍 Analyze Patch",
            disabled=(uploaded_file is None),
            use_container_width=True,
            type="primary"
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Handle Analysis
        if analyze_btn and uploaded_file is not None:
            with st.spinner("🔄 Analyzing tissue sample... This may take a few seconds."):
                image = Image.open(uploaded_file)
                success, result = analyze_image(
                    image,
                    patient_id,
                    manual_pos if manual_pos > 0 else None,
                    manual_neg if manual_neg > 0 else None,
                    notes
                )
                
                if success:
                    st.session_state.current_results = result
                    save_to_history(result, patient_id, uploaded_file.name)
                    st.success("✅ Analysis complete!")
                    st.rerun()
                else:
                    st.error(result)
    
    # ========================================================================
    # RIGHT COLUMN: Results Section
    # ========================================================================
    with right_col:
        st.markdown('<div class="results-card">', unsafe_allow_html=True)
        
        # Section 1: Results Header
        st.markdown("""
        <div class="section-header">
            <span style="font-size: 32px;">📊</span>
            <span>Analysis Results</span>
        </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.current_results is None:
            # Empty state
            st.markdown("""
            <div style="text-align: center; padding: 60px 20px; color: #9CA3AF;">
                <div style="font-size: 64px; margin-bottom: 16px;">📋</div>
                <h3>Upload an image to see results</h3>
                <p>Your analysis results will appear here</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            results = st.session_state.current_results['results']
            
            # Extract data
            pos_count = results['positive_cells']
            neg_count = results['negative_cells']
            total_count = results['total_cells']
            ki67_index = results['ki67_index']
            diagnosis = results['diagnosis']
            is_malignant = diagnosis.get('malignant', False)
            classification = diagnosis.get('classification', 'Unknown').title()
            
            # Section 2: Metrics Grid
            st.markdown("### 📈 Key Metrics")
            metric_col1, metric_col2 = st.columns(2)
            
            with metric_col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">⭐ POSITIVE CELLS</div>
                    <div class="metric-value">{pos_count:,}</div>
                    <div class="metric-sublabel">Ki-67 positive</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">📊 KI-67 INDEX</div>
                    <div class="metric-value">{ki67_index:.1f}%</div>
                    <div class="metric-sublabel">proliferation index</div>
                </div>
                """, unsafe_allow_html=True)
            
            with metric_col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">⚫ NEGATIVE CELLS</div>
                    <div class="metric-value">{neg_count:,}</div>
                    <div class="metric-sublabel">Ki-67 negative</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">🔢 TOTAL CELLS</div>
                    <div class="metric-value">{total_count:,}</div>
                    <div class="metric-sublabel">detected</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Section 3: Diagnosis Badge
            st.markdown("<br><br>", unsafe_allow_html=True)
            badge_class = "diagnosis-malignant" if is_malignant else "diagnosis-benign"
            badge_icon = "⚠️" if is_malignant else "✅"
            st.markdown(f"""
            <div style="text-align: center;">
                <div class="diagnosis-badge {badge_class}">
                    {badge_icon} {classification}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Section 4: Detection Visualization
            with st.expander("🔬 Detection Visualization", expanded=False):
                if 'images' in st.session_state.current_results:
                    images = st.session_state.current_results['images']
                    
                    viz_col1, viz_col2 = st.columns(2)
                    
                    with viz_col1:
                        if 'original' in images:
                            original_img = base64.b64decode(images['original'])
                            st.image(original_img, caption="Original Patch", use_container_width=True)
                    
                    with viz_col2:
                        if 'analyzed' in images:
                            analyzed_img = base64.b64decode(images['analyzed'])
                            st.image(analyzed_img, caption=f"Combined Detection ({total_count} cells)", use_container_width=True)
            
            # Section 5: Action Buttons
            st.markdown("<br>", unsafe_allow_html=True)
            action_col1, action_col2, action_col3 = st.columns(3)
            
            with action_col1:
                if st.button("⬇️ Download", use_container_width=True):
                    st.info("Download functionality - connect to backend")
            
            with action_col2:
                if st.button("🔍 Zoom", use_container_width=True):
                    st.info("Zoom functionality - implement modal")
            
            with action_col3:
                if st.button("📤 Share", use_container_width=True):
                    st.info("Share functionality - generate link")
            
            # Section 6: Export Options & Report Generation
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### 📄 Export & Reports")
            
            export_col1, export_col2, export_col3 = st.columns(3)
            
            with export_col1:
                if st.button("📄 Generate PDF Report", use_container_width=True, type="primary"):
                    analysis_id = st.session_state.current_results.get('analysis_id')
                    if analysis_id:
                        with st.spinner("Generating PDF report..."):
                            try:
                                response = requests.get(f'http://localhost:5001/api/report/pdf/{analysis_id}', timeout=30)
                                if response.status_code == 200:
                                    st.download_button(
                                        "⬇️ Download PDF Report",
                                        data=response.content,
                                        file_name=f"Ki67_Medical_Report_{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                        mime="application/pdf",
                                        use_container_width=True
                                    )
                                    st.success("✅ PDF report generated successfully!")
                                else:
                                    st.error("❌ Failed to generate PDF report")
                            except Exception as e:
                                st.error(f"❌ Error generating PDF: {str(e)}")
            
            with export_col2:
                csv_data = pd.DataFrame([{
                    'Patient ID': patient_id,
                    'Positive Cells': pos_count,
                    'Negative Cells': neg_count,
                    'Total Cells': total_count,
                    'Ki-67 Index': f"{ki67_index:.2f}%",
                    'Diagnosis': classification,
                    'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }])
                st.download_button(
                    "📊 Export CSV Data",
                    data=csv_data.to_csv(index=False),
                    file_name=f"Ki67_Data_{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with export_col3:
                if st.button("🖨️ Print Report", use_container_width=True):
                    st.info("💡 Use browser's print function (Ctrl+P) or download PDF for printing")
            
            # Section 7: Detailed Medical Report Preview
            st.markdown("<br><br>", unsafe_allow_html=True)
            with st.expander("📋 View Detailed Medical Report", expanded=False):
                render_medical_report_preview(
                    st.session_state.current_results,
                    patient_id,
                    pos_count,
                    neg_count,
                    total_count,
                    ki67_index,
                    classification,
                    is_malignant
                )
        
        st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# TAB 2: BATCH PROCESSING
# ============================================================================

def render_batch_processing():
    """Render batch processing tab"""
    
    st.markdown('<div class="upload-card">', unsafe_allow_html=True)
    
    # Section 1: Header
    st.markdown("""
    <div class="section-header">
        <span style="font-size: 32px;">📚</span>
        <span>Batch Upload</span>
    </div>
    <p style="color: #6B7280; margin-bottom: 24px;">Upload multiple patches for analysis</p>
    """, unsafe_allow_html=True)
    
    # Section 2: Multi-File Upload
    st.markdown("### ☁️ Upload Multiple Images")
    uploaded_files = st.file_uploader(
        "Select multiple images",
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True,
        help="Upload multiple histology images",
        label_visibility="collapsed"
    )
    
    if uploaded_files:
        st.info(f"📁 Selected Files: {len(uploaded_files)}")
        
        # Section 3: File List
        st.markdown("### 📄 File List")
        for idx, file in enumerate(uploaded_files):
            col1, col2, col3 = st.columns([3, 2, 1])
            with col1:
                st.text(f"📄 {file.name}")
            with col2:
                st.text(f"Size: {format_file_size(file.size)}")
            with col3:
                if st.button("❌", key=f"remove_{idx}"):
                    uploaded_files.pop(idx)
                    st.rerun()
        
        # Section 6: Process Button
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("⚙️ Process All Images", use_container_width=True, type="primary"):
            # Section 4: Progress Tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            batch_results = []
            
            for idx, file in enumerate(uploaded_files):
                status_text.text(f"Processing {idx + 1}/{len(uploaded_files)}: {file.name}")
                progress_bar.progress((idx + 1) / len(uploaded_files))
                
                image = Image.open(file)
                success, result = analyze_image(image, f"Batch_{idx+1}", notes=f"Batch processing - {file.name}")
                
                if success:
                    batch_results.append({
                        'filename': file.name,
                        'result': result,
                        'status': 'Success'
                    })
                else:
                    batch_results.append({
                        'filename': file.name,
                        'result': None,
                        'status': f'Failed: {result}'
                    })
                
                time.sleep(0.5)  # Small delay for UX
            
            st.session_state.batch_results = batch_results
            status_text.text("✅ Processing complete!")
            st.success(f"Processed {len(batch_results)} images")
            st.rerun()
    
    # Section 5: Batch Statistics
    if st.session_state.batch_results:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("### 📊 Batch Statistics")
        
        # Calculate statistics
        total_processed = len(st.session_state.batch_results)
        benign_count = sum(1 for r in st.session_state.batch_results 
                          if r['result'] and not r['result']['results']['diagnosis'].get('malignant', False))
        malignant_count = sum(1 for r in st.session_state.batch_results 
                             if r['result'] and r['result']['results']['diagnosis'].get('malignant', False))
        avg_ki67 = sum(r['result']['results']['ki67_index'] for r in st.session_state.batch_results 
                      if r['result']) / max(total_processed, 1)
        
        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
        
        with stat_col1:
            st.metric("Total Processed", total_processed, delta=None)
        with stat_col2:
            st.metric("Benign", benign_count, delta=None, delta_color="off")
        with stat_col3:
            st.metric("Malignant", malignant_count, delta=None, delta_color="inverse")
        with stat_col4:
            st.metric("Avg Ki-67", f"{avg_ki67:.1f}%", delta=None)
        
        # Section 7: Individual Results Table
        st.markdown("### 📋 Individual Results")
        
        results_data = []
        for idx, batch_item in enumerate(st.session_state.batch_results):
            if batch_item['result']:
                results_dict = batch_item['result']['results']
                diagnosis = results_dict['diagnosis']
                results_data.append({
                    'Batch': f"Batch_{idx+1}",
                    'Filename': batch_item['filename'],
                    'Classification': diagnosis.get('classification', 'Unknown').title(),
                    'Ki-67 Index': f"{results_dict['ki67_index']:.2f}%",
                    'Positive': results_dict['positive_cells'],
                    'Negative': results_dict['negative_cells'],
                    'Total': results_dict['total_cells']
                })
        
        if results_data:
            df = pd.DataFrame(results_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            # Section 8: Download Batch Report
            st.markdown("<br>", unsafe_allow_html=True)
            csv_export = df.to_csv(index=False)
            st.download_button(
                "📥 Download Batch Report (CSV)",
                data=csv_export,
                file_name=f"Batch_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
                type="primary"
            )
    
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# TAB 3: HISTORY
# ============================================================================

def render_history():
    """Render analysis history tab"""
    
    st.markdown('<div class="upload-card">', unsafe_allow_html=True)
    
    # Section 1: Header
    st.markdown("""
    <div class="section-header">
        <span style="font-size: 32px;">🕒</span>
        <span>Analysis History</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Section 3: Filter/Sort Options
    col1, col2, col3 = st.columns(3)
    with col1:
        filter_classification = st.selectbox(
            "Classification",
            ["All", "Benign", "Malignant"]
        )
    with col2:
        sort_by = st.selectbox(
            "Sort by",
            ["Date (Newest)", "Date (Oldest)", "Ki-67 Index", "Patient ID"]
        )
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()
    
    # Section 2: History List
    if not st.session_state.analysis_history:
        # Section 4: Empty State
        st.markdown("""
        <div style="text-align: center; padding: 60px 20px; color: #9CA3AF;">
            <div style="font-size: 64px; margin-bottom: 16px;">📋</div>
            <h3>No analysis history yet</h3>
            <p>Your completed analyses will appear here</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Filter and sort history
        filtered_history = st.session_state.analysis_history.copy()
        
        if filter_classification != "All":
            filtered_history = [
                h for h in filtered_history
                if h['results']['results']['diagnosis'].get('classification', '').lower() == filter_classification.lower()
            ]
        
        if sort_by == "Date (Oldest)":
            filtered_history.reverse()
        elif sort_by == "Ki-67 Index":
            filtered_history.sort(key=lambda x: x['results']['results']['ki67_index'], reverse=True)
        elif sort_by == "Patient ID":
            filtered_history.sort(key=lambda x: x['patient_id'])
        
        st.markdown(f"### 📊 {len(filtered_history)} Results")
        
        for entry in filtered_history:
            results = entry['results']['results']
            diagnosis = results['diagnosis']
            is_malignant = diagnosis.get('malignant', False)
            classification = diagnosis.get('classification', 'Unknown').title()
            ki67_index = results['ki67_index']
            pos_count = results['positive_cells']
            neg_count = results['negative_cells']
            
            # History Card
            card_class = "history-card-malignant" if is_malignant else "history-card-benign"
            
            with st.container():
                st.markdown(f"""
                <div class="history-card {card_class}">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <strong style="font-size: 16px;">{entry['patient_id']}</strong>
                        <span style="color: #6B7280; font-size: 14px;">{entry['timestamp']}</span>
                    </div>
                    <div style="color: #6B7280; margin-bottom: 8px;">
                        📄 {entry['filename']}
                    </div>
                    <div style="display: flex; gap: 16px; flex-wrap: wrap;">
                        <span style="background: {'#EF4444' if is_malignant else '#10B981'}; color: white; padding: 4px 12px; border-radius: 12px; font-size: 12px; font-weight: 600;">
                            {classification}
                        </span>
                        <span style="color: #374151;">Ki-67: {ki67_index:.1f}%</span>
                        <span style="color: #374151;">Positive: {pos_count:,} | Negative: {neg_count:,}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # View details button
                if st.button(f"👁️ View Details", key=f"view_{entry['analysis_id']}", use_container_width=False):
                    st.session_state.current_results = entry['results']
                    st.info("Switched to Single Analysis tab to view details")
    
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# TAB 4: STATISTICS
# ============================================================================

def render_statistics():
    """Render performance statistics tab"""
    
    st.markdown('<div class="upload-card">', unsafe_allow_html=True)
    
    # Section 1: Header
    st.markdown("""
    <div class="section-header">
        <span style="font-size: 32px;">📈</span>
        <span>Performance Statistics</span>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.analysis_history:
        st.markdown("""
        <div style="text-align: center; padding: 60px 20px; color: #9CA3AF;">
            <div style="font-size: 64px; margin-bottom: 16px;">📊</div>
            <h3>No data available</h3>
            <p>Complete some analyses to see statistics</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Calculate statistics
        total_analyses = len(st.session_state.analysis_history)
        benign_cases = sum(1 for h in st.session_state.analysis_history 
                          if not h['results']['results']['diagnosis'].get('malignant', False))
        malignant_cases = total_analyses - benign_cases
        avg_ki67 = sum(h['results']['results']['ki67_index'] 
                      for h in st.session_state.analysis_history) / total_analyses
        
        # Section 2: Overview Metrics
        st.markdown("### 📊 Overview")
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">🔬 TOTAL ANALYSES</div>
                <div class="metric-value">{total_analyses}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with metric_col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">✅ BENIGN CASES</div>
                <div class="metric-value">{benign_cases}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with metric_col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">⚠️ MALIGNANT CASES</div>
                <div class="metric-value">{malignant_cases}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with metric_col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">📊 AVG KI-67 INDEX</div>
                <div class="metric-value">{avg_ki67:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Section 3: Recent Performance
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("### 🕒 Recent Performance")
        recent_count = min(5, len(st.session_state.analysis_history))
        st.write(f"Last {recent_count} analyses:")
        
        for entry in st.session_state.analysis_history[:recent_count]:
            results = entry['results']['results']
            diagnosis = results['diagnosis']
            classification = diagnosis.get('classification', 'Unknown').title()
            is_malignant = diagnosis.get('malignant', False)
            
            col1, col2, col3 = st.columns([2, 2, 1])
            with col1:
                st.text(f"📄 {entry['filename']}")
            with col2:
                badge_color = "#EF4444" if is_malignant else "#10B981"
                st.markdown(f"""
                <span style="background: {badge_color}; color: white; padding: 4px 12px; border-radius: 12px; font-size: 12px; font-weight: 600;">
                    {classification}
                </span>
                """, unsafe_allow_html=True)
            with col3:
                st.text(f"{results['ki67_index']:.1f}%")
        
        # Section 4: Charts
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("### 📊 Visualizations")
        
        chart_col1, chart_col2 = st.columns(2)
        
        # Chart 1: Distribution Chart
        with chart_col1:
            fig1 = go.Figure(data=[
                go.Bar(
                    x=['Benign', 'Malignant'],
                    y=[benign_cases, malignant_cases],
                    marker_color=['#10B981', '#EF4444']
                )
            ])
            fig1.update_layout(
                title="Classification Distribution",
                xaxis_title="Classification",
                yaxis_title="Count",
                height=300
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        # Chart 2: Ki-67 Index Distribution
        with chart_col2:
            ki67_values = [h['results']['results']['ki67_index'] 
                          for h in st.session_state.analysis_history]
            
            fig2 = go.Figure(data=[
                go.Histogram(
                    x=ki67_values,
                    nbinsx=10,
                    marker_color='#7C3AED'
                )
            ])
            fig2.update_layout(
                title="Ki-67 Index Distribution",
                xaxis_title="Ki-67 Index (%)",
                yaxis_title="Frequency",
                height=300
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        # Chart 3: Timeline Chart
        if len(st.session_state.analysis_history) > 1:
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Prepare timeline data
            timeline_data = {}
            for entry in st.session_state.analysis_history:
                date = entry['timestamp'].split(' ')[0]
                timeline_data[date] = timeline_data.get(date, 0) + 1
            
            dates = sorted(timeline_data.keys())
            counts = [timeline_data[d] for d in dates]
            
            fig3 = go.Figure(data=[
                go.Scatter(
                    x=dates,
                    y=counts,
                    mode='lines+markers',
                    marker_color='#3B82F6',
                    line=dict(color='#3B82F6', width=2)
                )
            ])
            fig3.update_layout(
                title="Analyses Over Time",
                xaxis_title="Date",
                yaxis_title="Number of Analyses",
                height=300
            )
            st.plotly_chart(fig3, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# MEDICAL REPORT GENERATION
# ============================================================================

def render_medical_report_preview(results_data, patient_id, pos_count, neg_count, total_count, ki67_index, classification, is_malignant):
    """Render a detailed medical report preview"""
    
    # Extract data
    patient_data = results_data.get('patient_data', {})
    timestamp = results_data.get('timestamp', datetime.now().isoformat())
    analysis_id = results_data.get('analysis_id', 'N/A')
    
    # Clinical interpretation
    if ki67_index < 14:
        risk_level = "Low Proliferation"
        interpretation = "Low proliferation rate. Generally associated with less aggressive tumors and favorable prognosis."
        recommendation = "Continue standard surveillance and correlate with additional clinical findings."
    elif ki67_index < 30:
        risk_level = "Intermediate Proliferation"
        interpretation = "Intermediate proliferation rate. Moderate risk profile requiring multidisciplinary correlation."
        recommendation = "Consider adjunct molecular testing and align with hormone receptor status."
    else:
        risk_level = "High Proliferation"
        interpretation = "High proliferation rate. Indicates aggressive tumor biology requiring comprehensive treatment planning."
        recommendation = "Urgent oncologic consultation recommended with consideration for systemic therapy."
    
    # Render professional medical report
    st.markdown("""
    <style>
    .medical-report {
        background: white;
        padding: 40px;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        font-family: 'Arial', sans-serif;
        color: #333;
    }
    .report-header {
        text-align: center;
        border-bottom: 3px solid #7C3AED;
        padding-bottom: 20px;
        margin-bottom: 30px;
    }
    .report-header h1 {
        color: #7C3AED;
        font-size: 28px;
        margin: 0;
    }
    .report-header h2 {
        color: #666;
        font-size: 18px;
        margin: 10px 0 5px 0;
        font-weight: normal;
    }
    .report-header h3 {
        color: #888;
        font-size: 14px;
        margin: 5px 0;
        font-weight: normal;
    }
    .report-section {
        margin-bottom: 30px;
    }
    .report-section h4 {
        color: #7C3AED;
        font-size: 16px;
        border-bottom: 2px solid #E5E7EB;
        padding-bottom: 8px;
        margin-bottom: 15px;
    }
    .report-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 12px;
        margin-bottom: 20px;
    }
    .report-field {
        display: flex;
        justify-content: space-between;
        padding: 8px;
        background: #F9FAFB;
        border-radius: 4px;
    }
    .report-field strong {
        color: #4B5563;
        font-weight: 600;
    }
    .report-highlight {
        background: #F3F4F6;
        border-left: 4px solid #7C3AED;
        padding: 20px;
        margin: 20px 0;
        border-radius: 4px;
    }
    .report-diagnosis {
        text-align: center;
        padding: 20px;
        margin: 20px 0;
        border-radius: 8px;
        font-size: 24px;
        font-weight: bold;
    }
    .diagnosis-benign {
        background: #D1FAE5;
        color: #065F46;
        border: 2px solid #10B981;
    }
    .diagnosis-malignant {
        background: #FEE2E2;
        color: #991B1B;
        border: 2px solid #EF4444;
    }
    .report-footer {
        margin-top: 40px;
        padding-top: 20px;
        border-top: 2px solid #E5E7EB;
        text-align: center;
        font-size: 12px;
        color: #6B7280;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Report content
    diagnosis_class = "diagnosis-benign" if not is_malignant else "diagnosis-malignant"
    diagnosis_icon = "✅" if not is_malignant else "⚠️"
    
    report_html = f"""
    <div class="medical-report">
        <div class="report-header">
            <h1>🏥 MEDICAL DIAGNOSTIC REPORT</h1>
            <h2>Ki-67 Breast Cancer Immunohistochemistry Analysis</h2>
            <h3>Automated AI-Assisted Analysis System v1.0</h3>
            <p style="margin-top: 15px; color: #888;">
                <strong>Report ID:</strong> {analysis_id} | 
                <strong>Generated:</strong> {datetime.now().strftime('%B %d, %Y at %I:%M %p')}
            </p>
        </div>
        
        <div class="report-section">
            <h4>📋 Patient Information</h4>
            <div class="report-grid">
                <div class="report-field">
                    <strong>Patient ID:</strong>
                    <span>{patient_id or 'N/A'}</span>
                </div>
                <div class="report-field">
                    <strong>Analysis Date:</strong>
                    <span>{datetime.now().strftime('%Y-%m-%d')}</span>
                </div>
                <div class="report-field">
                    <strong>Test Type:</strong>
                    <span>Ki-67 Immunohistochemistry</span>
                </div>
                <div class="report-field">
                    <strong>Analysis Method:</strong>
                    <span>AI-Assisted Automated</span>
                </div>
            </div>
        </div>
        
        <div class="report-section">
            <h4>🔬 Analysis Results</h4>
            <div class="report-highlight">
                <div class="report-grid">
                    <div class="report-field">
                        <strong>Ki-67 Positive Cells:</strong>
                        <span style="color: #7C3AED; font-weight: bold; font-size: 18px;">{pos_count:,}</span>
                    </div>
                    <div class="report-field">
                        <strong>Ki-67 Negative Cells:</strong>
                        <span style="color: #6B7280; font-weight: bold; font-size: 18px;">{neg_count:,}</span>
                    </div>
                    <div class="report-field">
                        <strong>Total Cells Detected:</strong>
                        <span style="font-weight: bold; font-size: 18px;">{total_count:,}</span>
                    </div>
                    <div class="report-field">
                        <strong>Ki-67 Proliferation Index:</strong>
                        <span style="color: #7C3AED; font-weight: bold; font-size: 18px;">{ki67_index:.2f}%</span>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="report-section">
            <h4>🩺 Diagnosis</h4>
            <div class="report-diagnosis {diagnosis_class}">
                {diagnosis_icon} {classification.upper()}
            </div>
            <div style="background: #F9FAFB; padding: 15px; border-radius: 4px; margin-top: 15px;">
                <p style="margin: 0 0 10px 0;"><strong>Risk Level:</strong> {risk_level}</p>
                <p style="margin: 0;"><strong>Classification Basis:</strong> {'Positive cells > Negative cells' if is_malignant else 'Negative cells ≥ Positive cells'}</p>
            </div>
        </div>
        
        <div class="report-section">
            <h4>📊 Clinical Interpretation</h4>
            <div style="background: #F9FAFB; padding: 20px; border-radius: 4px; line-height: 1.8;">
                <p style="margin-bottom: 15px;"><strong>Analysis:</strong> {interpretation}</p>
                <p style="margin: 0;"><strong>Recommendation:</strong> {recommendation}</p>
            </div>
        </div>
        
        <div class="report-section">
            <h4>📝 Summary</h4>
            <p style="background: #F9FAFB; padding: 20px; border-radius: 4px; line-height: 1.8;">
                The tissue specimen demonstrates a Ki-67 proliferation index of <strong>{ki67_index:.2f}%</strong>, 
                with <strong>{pos_count:,}</strong> positive cells and <strong>{neg_count:,}</strong> negative cells 
                detected, totaling <strong>{total_count:,}</strong> cells analyzed. Based on the automated analysis, 
                the specimen is classified as <strong>{classification}</strong>. {interpretation}
            </p>
        </div>
        
        <div class="report-section">
            <h4>⚙️ Methodology</h4>
            <div style="background: #F9FAFB; padding: 15px; border-radius: 4px;">
                <ul style="margin: 0; padding-left: 20px; line-height: 1.8;">
                    <li><strong>Detection Model:</strong> U-Net with EfficientNet-B3 backbone</li>
                    <li><strong>Input Resolution:</strong> 640×640 pixels</li>
                    <li><strong>Analysis Method:</strong> Heatmap regression + Peak detection</li>
                    <li><strong>Classification Rule:</strong> Positive count vs. Negative count comparison</li>
                </ul>
            </div>
        </div>
        
        <div class="report-footer">
            <p><strong>⚠️ IMPORTANT DISCLAIMER</strong></p>
            <p>This is an AI-assisted analysis for research and clinical decision support purposes only.</p>
            <p>Results must be confirmed by a qualified pathologist before clinical use.</p>
            <p>Not intended as a standalone diagnostic tool. For research use only.</p>
            <p style="margin-top: 15px;">
                <strong>Ki-67 Analysis System v1.0</strong> | 
                © 2024 Medical AI Research Lab
            </p>
        </div>
    </div>
    """
    
    st.markdown(report_html, unsafe_allow_html=True)
    
    # Additional export options within the report
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.info("💡 **Tip:** Download the PDF version above for a professionally formatted report suitable for medical records.")
    with col2:
        st.info("📋 **Note:** This preview shows the report content. Use 'Generate PDF Report' button above to create the official document.")

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application entry point"""
    
    # Initialize
    init_session_state()
    load_custom_css()
    
    # Header
    st.markdown("""
    <div style="text-align: center; padding: 20px 0; background: white; border-radius: 16px; margin-bottom: 30px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
        <h1 style="color: #7C3AED; margin: 0;">🔬 Ki-67 Malignancy Classification System</h1>
        <p style="color: #6B7280; margin: 8px 0 0 0;">AI-Powered Breast Cancer Tissue Analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        
        st.markdown("**Backend Status**")
        try:
            response = requests.get('http://localhost:5001/api/health', timeout=2)
            if response.status_code == 200:
                st.success("✅ Connected")
            else:
                st.error("❌ Backend error")
        except:
            st.error("❌ Backend offline")
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Quick Stats**")
        st.metric("Total Analyses", len(st.session_state.analysis_history))
        st.metric("Session Results", 1 if st.session_state.current_results else 0)
        
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.analysis_history = []
            st.session_state.current_results = None
            st.session_state.batch_results = []
            st.success("History cleared!")
            st.rerun()
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #6B7280; font-size: 12px;">
            <p><strong>Ki-67 Analysis System v1.0</strong></p>
            <p>For research use only</p>
            <p>© 2024 Medical AI Lab</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Main Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔬 Single Analysis",
        "📚 Batch Processing",
        "🕒 History",
        "📈 Statistics"
    ])
    
    with tab1:
        render_single_analysis()
    
    with tab2:
        render_batch_processing()
    
    with tab3:
        render_history()
    
    with tab4:
        render_statistics()

# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == "__main__":
    main()
