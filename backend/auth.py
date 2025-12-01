"""
Authentication and authorization utilities for Ki-67 System
JWT-based authentication with role-based access control
"""

from functools import wraps
from flask import request, jsonify, current_app
from flask_login import LoginManager, current_user
import jwt
from datetime import datetime, timedelta
from database import User, AuditLog, db

login_manager = LoginManager()


@login_manager.user_loader
def load_user(user_id):
    """Load user by ID for Flask-Login"""
    return User.query.get(int(user_id))


def generate_token(user_id, expires_in=24):
    """Generate JWT token for user authentication"""
    payload = {
        'user_id': user_id,
        'exp': datetime.utcnow() + timedelta(hours=expires_in),
        'iat': datetime.utcnow()
    }
    token = jwt.encode(payload, current_app.config['SECRET_KEY'], algorithm='HS256')
    return token


def decode_token(token):
    """Decode and verify JWT token"""
    try:
        payload = jwt.decode(token, current_app.config['SECRET_KEY'], algorithms=['HS256'])
        return payload['user_id']
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


def token_required(f):
    """Decorator to require valid JWT token"""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        
        # Get token from header
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            try:
                token = auth_header.split(' ')[1]  # Bearer <token>
            except IndexError:
                return jsonify({'error': 'Invalid token format'}), 401
        
        if not token:
            return jsonify({'error': 'Authentication token is missing'}), 401
        
        # Verify token
        user_id = decode_token(token)
        if user_id is None:
            return jsonify({'error': 'Token is invalid or expired'}), 401
        
        # Load user
        current_user = User.query.get(user_id)
        if not current_user or not current_user.is_active:
            return jsonify({'error': 'User not found or inactive'}), 401
        
        return f(current_user, *args, **kwargs)
    
    return decorated


def role_required(*roles):
    """Decorator to require specific user role(s)"""
    def decorator(f):
        @wraps(f)
        def decorated(current_user, *args, **kwargs):
            if current_user.role not in roles:
                log_audit(
                    user_id=current_user.id,
                    action='access_denied',
                    description=f'Attempted to access {request.endpoint} without proper role',
                    resource_type='endpoint'
                )
                return jsonify({
                    'error': 'Access denied',
                    'message': f'This action requires one of the following roles: {", ".join(roles)}'
                }), 403
            return f(current_user, *args, **kwargs)
        return decorated
    return decorator


def admin_required(f):
    """Decorator to require admin role"""
    return role_required('admin')(f)


def pathologist_required(f):
    """Decorator to require pathologist or admin role"""
    return role_required('admin', 'pathologist')(f)


def log_audit(user_id, action, description=None, resource_type=None, resource_id=None, extra_data=None):
    """Log user action to audit trail"""
    try:
        audit_log = AuditLog(
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            description=description,
            ip_address=request.remote_addr if request else None,
            user_agent=request.headers.get('User-Agent') if request else None,
            extra_data=extra_data
        )
        db.session.add(audit_log)
        db.session.commit()
    except Exception as e:
        print(f"Error logging audit: {str(e)}")
        db.session.rollback()


def verify_password_strength(password):
    """Verify password meets security requirements"""
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    
    has_upper = any(c.isupper() for c in password)
    has_lower = any(c.islower() for c in password)
    has_digit = any(c.isdigit() for c in password)
    
    if not (has_upper and has_lower and has_digit):
        return False, "Password must contain uppercase, lowercase, and numbers"
    
    return True, "Password is strong"


def can_user_access_analysis(user, analysis):
    """Check if user can access a specific analysis"""
    # Admin can access everything
    if user.is_admin:
        return True
    
    # Pathologists can access analyses in their department
    if user.is_pathologist:
        if analysis.user.department == user.department:
            return True
    
    # Users can access their own analyses
    if analysis.user_id == user.id:
        return True
    
    return False


def can_user_review_analysis(user, analysis):
    """Check if user can review/approve an analysis"""
    # Only pathologists and admins can review
    if not (user.is_pathologist or user.is_admin):
        return False
    
    # Can't review own analysis
    if analysis.user_id == user.id:
        return False
    
    # Pathologists can review in their department
    if user.is_pathologist:
        return analysis.user.department == user.department
    
    # Admins can review everything
    return user.is_admin


def get_user_permissions(user):
    """Get user permissions based on role"""
    base_permissions = {
        'can_analyze': True,
        'can_view_own_analyses': True,
        'can_export_reports': True
    }
    
    if user.is_technician:
        return {
            **base_permissions,
            'can_view_all_analyses': False,
            'can_review_analyses': False,
            'can_manage_users': False,
            'can_view_audit_logs': False
        }
    
    if user.is_pathologist:
        return {
            **base_permissions,
            'can_view_all_analyses': True,  # In their department
            'can_review_analyses': True,
            'can_manage_users': False,
            'can_view_audit_logs': True
        }
    
    if user.is_admin:
        return {
            **base_permissions,
            'can_view_all_analyses': True,
            'can_review_analyses': True,
            'can_manage_users': True,
            'can_view_audit_logs': True,
            'can_manage_system': True
        }
    
    return base_permissions


def init_auth(app):
    """Initialize authentication system"""
    login_manager.init_app(app)
    login_manager.login_view = 'login'
    login_manager.login_message = 'Please log in to access this page.'
