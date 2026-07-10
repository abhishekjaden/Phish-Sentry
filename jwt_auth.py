"""
JWT authentication for PhishGuard's API layer.
Used by programmatic clients (e.g., the browser extension) — separate from the
session-based auth that powers the web UI.
"""
import os
import datetime
from functools import wraps
import jwt
from flask import request, jsonify

JWT_SECRET = os.environ.get('JWT_SECRET_KEY')
if not JWT_SECRET:
    # Fall back to Flask secret if a dedicated JWT secret isn't set
    JWT_SECRET = os.environ.get('FLASK_SECRET_KEY', 'dev-only-change-me')

JWT_ALGORITHM = 'HS256'
ACCESS_TOKEN_EXPIRES = datetime.timedelta(minutes=30)
REFRESH_TOKEN_EXPIRES = datetime.timedelta(days=7)


def generate_tokens(user_id, role):
    """Create an access + refresh token pair for a user."""
    now = datetime.datetime.now(datetime.timezone.utc)

    access_payload = {
        'user_id': user_id,
        'role': role,
        'type': 'access',
        'iat': now,
        'exp': now + ACCESS_TOKEN_EXPIRES
    }
    refresh_payload = {
        'user_id': user_id,
        'type': 'refresh',
        'iat': now,
        'exp': now + REFRESH_TOKEN_EXPIRES
    }

    access_token = jwt.encode(access_payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    refresh_token = jwt.encode(refresh_payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return access_token, refresh_token


def decode_token(token):
    """Decode and validate a token. Returns payload dict or None."""
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        return {'error': 'Token expired'}
    except jwt.InvalidTokenError:
        return {'error': 'Invalid token'}


def jwt_required(f):
    """Decorator: protect an API endpoint with a valid access token."""
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get('Authorization', '')
        if not auth_header.startswith('Bearer '):
            return jsonify({'error': 'Missing or malformed Authorization header'}), 401

        token = auth_header.split(' ', 1)[1].strip()
        payload = decode_token(token)

        if payload is None or 'error' in payload:
            msg = payload.get('error', 'Invalid token') if payload else 'Invalid token'
            return jsonify({'error': msg}), 401

        if payload.get('type') != 'access':
            return jsonify({'error': 'Wrong token type'}), 401

        # Attach identity to the request context for the wrapped view
        request.jwt_user_id = payload['user_id']
        request.jwt_role = payload.get('role', 'user')
        return f(*args, **kwargs)

    return decorated