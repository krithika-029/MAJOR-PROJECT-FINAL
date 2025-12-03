from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship

from .db import Base


class Upload(Base):
    __tablename__ = 'uploads'

    id = Column(Integer, primary_key=True, index=True)
    original_filename = Column(String(512), nullable=False)
    saved_filename = Column(String(512), nullable=False)
    upload_path = Column(Text, nullable=False)
    uploaded_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Patient metadata
    patient_id = Column(String(128), default='')
    patient_name = Column(String(256), default='')
    age = Column(String(32), default='')
    gender = Column(String(32), default='')
    contact = Column(String(128), default='')
    exam_date = Column(String(64), default='')
    physician = Column(String(256), default='')
    clinical_notes = Column(Text, default='')

    analyses = relationship('AnalysisResult', back_populates='upload', cascade='all, delete-orphan')


class AnalysisResult(Base):
    __tablename__ = 'analysis_results'

    id = Column(Integer, primary_key=True, index=True)
    analysis_id = Column(String(64), unique=True, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    upload_id = Column(Integer, ForeignKey('uploads.id', ondelete='CASCADE'), nullable=False)
    upload = relationship('Upload', back_populates='analyses')

    # Model metrics
    positive_cells = Column(Integer, default=0)
    negative_cells = Column(Integer, default=0)
    total_cells = Column(Integer, default=0)
    ki67_index = Column(Float, default=0.0)
    classification = Column(String(64), default='')
    risk = Column(String(64), default='')
    malignant = Column(Boolean, default=False)

    # QC summary
    qc_available = Column(Boolean, default=False)
    qc_flagged = Column(Boolean, default=False)
    qc_reason = Column(Text, default='')
    qc_ki67_percent_delta = Column(Float, default=0.0)
    qc_classification_match = Column(Boolean, default=True)

    # Outputs
    pdf_path = Column(Text, default='')
    csv_path = Column(Text, default='')
    result_image_path = Column(Text, default='')

    # Dataset/source info
    dataset_source = Column(String(64), default='')
    dataset_subset = Column(String(64), default='')
    dataset_image_name = Column(String(256), default='')
