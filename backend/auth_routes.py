"""
Authentication API Routes for Ki-67 System
Handles login, registration, user management, and audit logs
"""

from flask import Blueprint, request, jsonify
from datetime import datetime
from database import db, User, Analysis, AuditLog
from auth import (
    generate_token, token_required, admin_required, 
    log_audit, verify_password_strength, get_user_permissions,
    can_user_access_analysis, can_user_review_analysis
)
from sqlalchemy import desc

auth_bp = Blueprint('auth', __name__, url_prefix='/api/auth')


@auth_bp.route('/register', methods=['POST'])
def register():
    """Register a new user"""
    data = request.get_json()
    
    # Validate required fields
    required_fields = ['username', 'email', 'password', 'full_name', 'role']
    if not all(field in data for field in required_fields):
        return jsonify({'error': 'Missing required fields'}), 400
    
    # Check if username or email already exists
    if User.query.filter_by(username=data['username']).first():
        return jsonify({'error': 'Username already exists'}), 400
    
    if User.query.filter_by(email=data['email']).first():
        return jsonify({'error': 'Email already exists'}), 400
    
    # Verify password strength
    is_strong, message = verify_password_strength(data['password'])
    if not is_strong:
        return jsonify({'error': message}), 400
    
    # Validate role
    valid_roles = ['admin', 'pathologist', 'technician']
    if data['role'] not in valid_roles:
        return jsonify({'error': f'Invalid role. Must be one of: {", ".join(valid_roles)}'}), 400
    
    # Create new user
    user = User(
        username=data['username'],
        email=data['email'],
        full_name=data['full_name'],
        role=data['role'],
        department=data.get('department'),
        license_number=data.get('license_number'),
        phone=data.get('phone'),
        is_active=True,
        is_verified=False  # Requires admin approval
    )
    user.set_password(data['password'])
    
    try:
        db.session.add(user)
        db.session.commit()
        
        log_audit(
            user_id=user.id,
            action='user_registered',
            description=f'New user registered: {user.username} ({user.role})',
            resource_type='user',
            resource_id=str(user.id)
        )
        
        return jsonify({
            'message': 'User registered successfully. Awaiting admin approval.',
            'user': user.to_dict()
        }), 201
    
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Registration failed: {str(e)}'}), 500


@auth_bp.route('/login', methods=['POST'])
def login():
    """User login"""
    data = request.get_json()
    
    if not data.get('username') or not data.get('password'):
        return jsonify({'error': 'Username and password are required'}), 400
    
    # Find user
    user = User.query.filter_by(username=data['username']).first()
    
    if not user or not user.check_password(data['password']):
        log_audit(
            user_id=user.id if user else None,
            action='login_failed',
            description=f'Failed login attempt for username: {data["username"]}',
            resource_type='user'
        )
        return jsonify({'error': 'Invalid username or password'}), 401
    
    if not user.is_active:
        return jsonify({'error': 'Account is inactive. Contact administrator.'}), 403
    
    if not user.is_verified:
        return jsonify({'error': 'Account pending approval. Contact administrator.'}), 403
    
    # Update last login
    user.last_login = datetime.utcnow()
    db.session.commit()
    
    # Generate token
    token = generate_token(user.id)
    
    # Log successful login
    log_audit(
        user_id=user.id,
        action='login',
        description=f'User logged in: {user.username}',
        resource_type='user',
        resource_id=str(user.id)
    )
    
    return jsonify({
        'message': 'Login successful',
        'token': token,
        'user': user.to_dict(),
        'permissions': get_user_permissions(user)
    }), 200


@auth_bp.route('/logout', methods=['POST'])
@token_required
def logout(current_user):
    """User logout"""
    log_audit(
        user_id=current_user.id,
        action='logout',
        description=f'User logged out: {current_user.username}',
        resource_type='user',
        resource_id=str(current_user.id)
    )
    
    return jsonify({'message': 'Logout successful'}), 200


@auth_bp.route('/me', methods=['GET'])
@token_required
def get_current_user(current_user):
    """Get current user information"""
    return jsonify({
        'user': current_user.to_dict(),
        'permissions': get_user_permissions(current_user)
    }), 200


@auth_bp.route('/profile', methods=['PUT'])
@token_required
def update_profile(current_user):
    """Update user profile"""
    data = request.get_json()
    
    # Update allowed fields
    allowed_fields = ['full_name', 'email', 'phone', 'department']
    
    for field in allowed_fields:
        if field in data:
            setattr(current_user, field, data[field])
    
    # Handle password change
    if 'new_password' in data:
        if not data.get('current_password'):
            return jsonify({'error': 'Current password is required'}), 400
        
        if not current_user.check_password(data['current_password']):
            return jsonify({'error': 'Current password is incorrect'}), 401
        
        is_strong, message = verify_password_strength(data['new_password'])
        if not is_strong:
            return jsonify({'error': message}), 400
        
        current_user.set_password(data['new_password'])
        
        log_audit(
            user_id=current_user.id,
            action='password_changed',
            description='User changed their password',
            resource_type='user',
            resource_id=str(current_user.id)
        )
    
    try:
        db.session.commit()
        
        log_audit(
            user_id=current_user.id,
            action='profile_updated',
            description='User updated their profile',
            resource_type='user',
            resource_id=str(current_user.id),
            extra_data={'updated_fields': list(data.keys())}
        )
        
        return jsonify({
            'message': 'Profile updated successfully',
            'user': current_user.to_dict()
        }), 200
    
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Update failed: {str(e)}'}), 500


@auth_bp.route('/users', methods=['GET'])
@token_required
@admin_required
def list_users(current_user):
    """List all users (admin only)"""
    users = User.query.order_by(User.created_at.desc()).all()
    
    return jsonify({
        'users': [user.to_dict() for user in users],
        'total': len(users)
    }), 200


@auth_bp.route('/users/<int:user_id>', methods=['GET'])
@token_required
@admin_required
def get_user(current_user, user_id):
    """Get specific user details (admin only)"""
    user = User.query.get_or_404(user_id)
    
    # Get user's analysis count
    analysis_count = Analysis.query.filter_by(user_id=user.id).count()
    
    # Get recent activity
    recent_logs = AuditLog.query.filter_by(user_id=user.id).order_by(desc(AuditLog.created_at)).limit(10).all()
    
    return jsonify({
        'user': user.to_dict(),
        'analysis_count': analysis_count,
        'recent_activity': [log.to_dict() for log in recent_logs]
    }), 200


@auth_bp.route('/users/<int:user_id>', methods=['PUT'])
@token_required
@admin_required
def update_user(current_user, user_id):
    """Update user (admin only)"""
    user = User.query.get_or_404(user_id)
    data = request.get_json()
    
    # Update allowed fields
    allowed_fields = ['full_name', 'email', 'role', 'department', 'license_number', 'phone', 'is_active', 'is_verified']
    
    for field in allowed_fields:
        if field in data:
            setattr(user, field, data[field])
    
    try:
        db.session.commit()
        
        log_audit(
            user_id=current_user.id,
            action='user_updated',
            description=f'Admin updated user: {user.username}',
            resource_type='user',
            resource_id=str(user.id),
            extra_data={'updated_fields': list(data.keys())}
        )
        
        return jsonify({
            'message': 'User updated successfully',
            'user': user.to_dict()
        }), 200
    
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Update failed: {str(e)}'}), 500


@auth_bp.route('/users/<int:user_id>', methods=['DELETE'])
@token_required
@admin_required
def delete_user(current_user, user_id):
    """Delete user (admin only)"""
    if current_user.id == user_id:
        return jsonify({'error': 'Cannot delete your own account'}), 400
    
    user = User.query.get_or_404(user_id)
    
    try:
        # Log before deletion
        log_audit(
            user_id=current_user.id,
            action='user_deleted',
            description=f'Admin deleted user: {user.username}',
            resource_type='user',
            resource_id=str(user.id)
        )
        
        db.session.delete(user)
        db.session.commit()
        
        return jsonify({'message': 'User deleted successfully'}), 200
    
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Deletion failed: {str(e)}'}), 500


@auth_bp.route('/audit-logs', methods=['GET'])
@token_required
def get_audit_logs(current_user):
    """Get audit logs"""
    # Only admins and pathologists can view audit logs
    if not (current_user.is_admin or current_user.is_pathologist):
        return jsonify({'error': 'Access denied'}), 403
    
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 50, type=int)
    action = request.args.get('action')
    user_id = request.args.get('user_id', type=int)
    
    query = AuditLog.query
    
    # Filter by action
    if action:
        query = query.filter_by(action=action)
    
    # Filter by user (admins can see all, pathologists can see their department)
    if user_id:
        query = query.filter_by(user_id=user_id)
    elif current_user.is_pathologist:
        # Get users in same department
        dept_users = User.query.filter_by(department=current_user.department).all()
        user_ids = [u.id for u in dept_users]
        query = query.filter(AuditLog.user_id.in_(user_ids))
    
    # Paginate
    pagination = query.order_by(desc(AuditLog.created_at)).paginate(
        page=page, per_page=per_page, error_out=False
    )
    
    return jsonify({
        'logs': [log.to_dict() for log in pagination.items],
        'total': pagination.total,
        'page': page,
        'per_page': per_page,
        'pages': pagination.pages
    }), 200


@auth_bp.route('/stats', methods=['GET'])
@token_required
@admin_required
def get_user_stats(current_user):
    """Get user statistics (admin only)"""
    total_users = User.query.count()
    active_users = User.query.filter_by(is_active=True).count()
    pending_users = User.query.filter_by(is_verified=False).count()
    
    # Count by role
    admins = User.query.filter_by(role='admin').count()
    pathologists = User.query.filter_by(role='pathologist').count()
    technicians = User.query.filter_by(role='technician').count()
    
    # Recent registrations
    recent_users = User.query.order_by(desc(User.created_at)).limit(5).all()
    
    return jsonify({
        'total_users': total_users,
        'active_users': active_users,
        'pending_approval': pending_users,
        'by_role': {
            'admin': admins,
            'pathologist': pathologists,
            'technician': technicians
        },
        'recent_registrations': [user.to_dict() for user in recent_users]
    }), 200
