"""
Database models and initialization for Ki-67 Analysis System
Supports user authentication, roles, and audit logging
"""

from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import secrets

db = SQLAlchemy()

class User(UserMixin, db.Model):
    """User model with role-based access control"""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    
    # User Information
    full_name = db.Column(db.String(150))
    role = db.Column(db.String(20), nullable=False, default='technician')  # admin, pathologist, technician
    license_number = db.Column(db.String(50))  # Medical license for pathologists
    department = db.Column(db.String(100))
    phone = db.Column(db.String(20))
    
    # Account Status
    is_active = db.Column(db.Boolean, default=True)
    is_verified = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    
    # Security
    reset_token = db.Column(db.String(100), unique=True)
    reset_token_expiry = db.Column(db.DateTime)
    
    # Relationships
    analyses = db.relationship('Analysis', foreign_keys='Analysis.user_id', backref='user', lazy=True)
    audit_logs = db.relationship('AuditLog', backref='user', lazy=True)
    
    def set_password(self, password):
        """Hash and set password"""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Verify password"""
        return check_password_hash(self.password_hash, password)
    
    def generate_reset_token(self):
        """Generate password reset token"""
        self.reset_token = secrets.token_urlsafe(32)
        return self.reset_token
    
    @property
    def is_admin(self):
        return self.role == 'admin'
    
    @property
    def is_pathologist(self):
        return self.role == 'pathologist'
    
    @property
    def is_technician(self):
        return self.role == 'technician'
    
    def to_dict(self):
        """Convert user to dictionary (excluding sensitive data)"""
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'full_name': self.full_name,
            'role': self.role,
            'department': self.department,
            'license_number': self.license_number,
            'phone': self.phone,
            'is_active': self.is_active,
            'is_verified': self.is_verified,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_login': self.last_login.isoformat() if self.last_login else None
        }
    
    def __repr__(self):
        return f'<User {self.username} ({self.role})>'


class Analysis(db.Model):
    """Analysis records with user tracking"""
    __tablename__ = 'analyses'
    
    id = db.Column(db.Integer, primary_key=True)
    analysis_id = db.Column(db.String(50), unique=True, nullable=False, index=True)
    
    # User who performed the analysis
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    
    # Patient Information
    patient_id = db.Column(db.String(100), index=True)
    patient_name = db.Column(db.String(150))
    
    # Image Information
    filename = db.Column(db.String(255), nullable=False)
    image_path = db.Column(db.String(500))
    
    # Analysis Results
    ai_positive = db.Column(db.Integer)
    ai_negative = db.Column(db.Integer)
    ai_ki67 = db.Column(db.Float)
    classification = db.Column(db.String(20))  # Benign, Malignant
    
    # Manual Validation (if provided)
    manual_positive = db.Column(db.Integer)
    manual_negative = db.Column(db.Integer)
    manual_ki67 = db.Column(db.Float)
    
    # Quality Control
    confidence_score = db.Column(db.Float)
    flagged = db.Column(db.Boolean, default=False)
    flag_reason = db.Column(db.Text)
    
    # Review Status
    status = db.Column(db.String(20), default='pending')  # pending, reviewed, approved
    reviewed_by = db.Column(db.Integer, db.ForeignKey('users.id'))
    reviewed_at = db.Column(db.DateTime)
    
    # Clinical Notes
    clinical_notes = db.Column(db.Text)
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    reviewer = db.relationship('User', foreign_keys=[reviewed_by], backref='reviewed_analyses')
    
    def to_dict(self):
        """Convert analysis to dictionary"""
        return {
            'id': self.id,
            'analysis_id': self.analysis_id,
            'user_id': self.user_id,
            'username': self.user.username if self.user else None,
            'patient_id': self.patient_id,
            'patient_name': self.patient_name,
            'filename': self.filename,
            'ai_positive': self.ai_positive,
            'ai_negative': self.ai_negative,
            'ai_ki67': self.ai_ki67,
            'classification': self.classification,
            'manual_positive': self.manual_positive,
            'manual_negative': self.manual_negative,
            'manual_ki67': self.manual_ki67,
            'confidence_score': self.confidence_score,
            'flagged': self.flagged,
            'flag_reason': self.flag_reason,
            'status': self.status,
            'reviewed_by': self.reviewed_by,
            'reviewed_at': self.reviewed_at.isoformat() if self.reviewed_at else None,
            'clinical_notes': self.clinical_notes,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }
    
    def __repr__(self):
        return f'<Analysis {self.analysis_id} by {self.user.username}>'


class AuditLog(db.Model):
    """Audit trail for all user actions"""
    __tablename__ = 'audit_logs'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    
    # Action Information
    action = db.Column(db.String(50), nullable=False)  # login, logout, analyze, review, etc.
    resource_type = db.Column(db.String(50))  # analysis, user, settings, etc.
    resource_id = db.Column(db.String(100))
    
    # Details
    description = db.Column(db.Text)
    ip_address = db.Column(db.String(45))
    user_agent = db.Column(db.String(255))
    
    # Extra data (renamed from metadata to avoid SQLAlchemy conflict)
    extra_data = db.Column(db.JSON)  # Additional structured data
    
    # Timestamp
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    
    def to_dict(self):
        """Convert audit log to dictionary"""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'username': self.user.username if self.user else None,
            'action': self.action,
            'resource_type': self.resource_type,
            'resource_id': self.resource_id,
            'description': self.description,
            'ip_address': self.ip_address,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'extra_data': self.extra_data
        }
    
    def __repr__(self):
        return f'<AuditLog {self.action} by {self.user.username} at {self.created_at}>'


def init_db(app):
    """Initialize database with app context"""
    db.init_app(app)
    with app.app_context():
        db.create_all()
        create_default_admin()


def create_default_admin():
    """Create default admin user if no users exist"""
    if User.query.count() == 0:
        admin = User(
            username='admin',
            email='admin@ki67system.com',
            full_name='System Administrator',
            role='admin',
            is_active=True,
            is_verified=True
        )
        admin.set_password('admin123')  # Change this in production!
        
        db.session.add(admin)
        db.session.commit()
        
        print("✅ Default admin user created!")
        print("   Username: admin")
        print("   Password: admin123")
        print("   ⚠️  PLEASE CHANGE THE PASSWORD IMMEDIATELY!")
