import os
import secrets
from datetime import datetime, timedelta
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, Text, DateTime, ForeignKey
from sqlalchemy.orm import sessionmaker, declarative_base
from werkzeug.security import generate_password_hash, check_password_hash

Base = declarative_base()


class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True, autoincrement=True)
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    username = Column(String(100), unique=True, nullable=False)
    email = Column(String(200), unique=True, nullable=False)
    password_hash = Column(String(500), nullable=False)
    role = Column(String(20), default='user')
    is_verified = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class Scan(Base):
    __tablename__ = 'scans'

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    scan_type = Column(String(50), nullable=False)
    content = Column(Text, nullable=False)
    result = Column(Text, nullable=False)
    confidence = Column(Float, nullable=False)
    is_phishing = Column(Boolean, nullable=False)
    shap_explanation = Column(Text, nullable=True)
    # 'web'  = scan run from the website UI
    # 'extension' = scan run by the browser extension (live page checks + manual re-checks)
    source = Column(String(20), nullable=False, default='web')
    created_at = Column(DateTime, default=datetime.utcnow)

class AuditLog(Base):
    __tablename__ = 'audit_logs'

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, nullable=True)  # null for anonymous events (e.g. failed login)
    event_type = Column(String(50), nullable=False)  # login, login_failed, predict, api_access
    detail = Column(String(500), nullable=True)
    ip_address = Column(String(64), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow) 
 

class VerificationToken(Base):
    __tablename__ = 'verification_tokens'

    id = Column(Integer, primary_key=True, autoincrement=True)
    token = Column(String(128), unique=True, nullable=False, index=True)
    user_id = Column(Integer, nullable=False)
    purpose = Column(String(20), nullable=False)  # 'verify' or 'reset'
    expires_at = Column(DateTime, nullable=False)
    used = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class Database:
    def __init__(self):
        database_url = os.environ.get('DATABASE_URL', 'sqlite:///phishing.db')
        
        # SQLAlchemy engine with connection pooling
        self.engine = create_engine(
            database_url,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True
        )
        self.Session = sessionmaker(bind=self.engine)
        self.init_database()

    def init_database(self):
        Base.metadata.create_all(self.engine)
        self._create_admin_if_needed()

    def _create_admin_if_needed(self):
        admin_email = os.environ.get('ADMIN_EMAIL')
        admin_password = os.environ.get('ADMIN_PASSWORD')

        if admin_email and admin_password:
            session = self.Session()
            try:
                existing = session.query(User).filter_by(email=admin_email).first()
                if existing is None:
                    admin = User(
                        first_name='Admin',
                        last_name='User',
                        username='admin',
                        email=admin_email,
                        password_hash=generate_password_hash(admin_password),
                        role='admin'
                    )
                    session.add(admin)
                    session.commit()
                    print(f"Admin account created: {admin_email}")
            except Exception:
                session.rollback()
            finally:
                session.close()

    def create_user(self, first_name, last_name, username, email, password):
        session = self.Session()
        try:
            user = User(
                first_name=first_name,
                last_name=last_name,
                username=username,
                email=email,
                password_hash=generate_password_hash(password),
                role='user'
            )
            session.add(user)
            session.commit()
            user_id = user.id
            return user_id
        except Exception:
            session.rollback()
            return None
        finally:
            session.close()

    def authenticate_user(self, email, password, role):
        session = self.Session()
        try:
            user = session.query(User).filter_by(email=email, role=role).first()
            if user and check_password_hash(user.password_hash, password):
                return {
                    'id': user.id,
                    'first_name': user.first_name,
                    'last_name': user.last_name,
                    'username': user.username,
                    'email': user.email,
                    'role': user.role,
                    'is_verified': user.is_verified
                }
            return None
        finally:
            session.close()

    def get_user_by_id(self, user_id):
        session = self.Session()
        try:
            user = session.query(User).filter_by(id=user_id).first()
            if user:
                return {
                    'id': user.id,
                    'first_name': user.first_name,
                    'last_name': user.last_name,
                    'username': user.username,
                    'email': user.email,
                    'role': user.role
                }
            return None
        finally:
            session.close()

    def save_scan(self, user_id, scan_type, content, result, confidence, is_phishing, shap_explanation=None, source='web'):
        session = self.Session()
        try:
            scan = Scan(
                user_id=user_id,
                scan_type=scan_type,
                content=content,
                result=result,
                confidence=confidence,
                is_phishing=is_phishing,
                shap_explanation=shap_explanation,
                source=source
            )
            session.add(scan)
            session.commit()
            return scan.id
        except Exception:
            session.rollback()
            return None
        finally:
            session.close()
    def log_event(self, event_type, user_id=None, detail=None, ip_address=None):
        """Write a security/audit event. Best-effort — never breaks the request."""
        session = self.Session()
        try:
            entry = AuditLog(
                user_id=user_id,
                event_type=event_type,
                detail=detail[:500] if detail else None,
                ip_address=ip_address
            )
            session.add(entry)
            session.commit()
            return entry.id
        except Exception:
            session.rollback()
            return None
        finally:
            session.close()
    def create_verification_token(self, user_id, purpose, ttl_hours):
        """Create a secure token for 'verify' or 'reset'. Returns the token string."""
        session = self.Session()
        try:
            token = secrets.token_urlsafe(32)
            entry = VerificationToken(
                token=token,
                user_id=user_id,
                purpose=purpose,
                expires_at=datetime.utcnow() + timedelta(hours=ttl_hours),
                used=False
            )
            session.add(entry)
            session.commit()
            return token
        except Exception:
            session.rollback()
            return None
        finally:
            session.close()

    def get_valid_token(self, token, purpose):
        """Return token info if valid (right purpose, unused, not expired), else None."""
        session = self.Session()
        try:
            entry = session.query(VerificationToken).filter_by(
                token=token, purpose=purpose, used=False
            ).first()
            if not entry or entry.expires_at < datetime.utcnow():
                return None
            return {'id': entry.id, 'user_id': entry.user_id, 'purpose': entry.purpose}
        finally:
            session.close()

    def mark_token_used(self, token):
        session = self.Session()
        try:
            entry = session.query(VerificationToken).filter_by(token=token).first()
            if entry:
                entry.used = True
                session.commit()
                return True
            return False
        except Exception:
            session.rollback()
            return False
        finally:
            session.close()

    def set_user_verified(self, user_id):
        session = self.Session()
        try:
            user = session.query(User).filter_by(id=user_id).first()
            if user:
                user.is_verified = True
                session.commit()
                return True
            return False
        except Exception:
            session.rollback()
            return False
        finally:
            session.close()

    def get_user_by_email(self, email):
        session = self.Session()
        try:
            user = session.query(User).filter_by(email=email).first()
            if user:
                return {
                    'id': user.id, 'first_name': user.first_name,
                    'last_name': user.last_name, 'email': user.email,
                    'role': user.role, 'is_verified': user.is_verified
                }
            return None
        finally:
            session.close()

    def update_user_password(self, user_id, new_password):
        session = self.Session()
        try:
            user = session.query(User).filter_by(id=user_id).first()
            if user:
                user.password_hash = generate_password_hash(new_password)
                session.commit()
                return True
            return False
        except Exception:
            session.rollback()
            return False
        finally:
            session.close()

    def get_user_scans(self, user_id, limit=50, source=None):
        session = self.Session()
        try:
            q = session.query(Scan).filter_by(user_id=user_id)
            if source in ('web', 'extension'):
                q = q.filter_by(source=source)
            scans = q.order_by(Scan.created_at.desc()).limit(limit).all()
            return [{
                'id': s.id,
                'scan_type': s.scan_type,
                'content': s.content[:100] + '...' if len(s.content) > 100 else s.content,
                'result': s.result,
                'confidence': s.confidence,
                'is_phishing': s.is_phishing,
                'source': s.source or 'web',
                'created_at': s.created_at.isoformat() if s.created_at else None
            } for s in scans]
        finally:
            session.close()

    def get_scan_details(self, scan_id, user_id):
        session = self.Session()
        try:
            scan = session.query(Scan).filter_by(id=scan_id, user_id=user_id).first()
            if scan:
                return {
                    'id': scan.id,
                    'scan_type': scan.scan_type,
                    'content': scan.content,
                    'result': scan.result,
                    'confidence': scan.confidence,
                    'is_phishing': scan.is_phishing,
                    'shap_explanation': scan.shap_explanation,
                    'created_at': scan.created_at.isoformat() if scan.created_at else None
                }
            return None
        finally:
            session.close()

    def get_admin_stats(self):
        session = self.Session()
        try:
            total_users = session.query(User).filter_by(role='user').count()
            total_scans = session.query(Scan).count()
            phishing_detected = session.query(Scan).filter_by(is_phishing=True).count()
            legitimate_detected = session.query(Scan).filter_by(is_phishing=False).count()

            from sqlalchemy import func
            scan_types = dict(
                session.query(Scan.scan_type, func.count(Scan.id))
                .group_by(Scan.scan_type).all()
            )

            thirty_days_ago = datetime.utcnow().replace(hour=0, minute=0, second=0)
            daily_scans = session.query(
                func.date(Scan.created_at),
                func.count(Scan.id)
            ).filter(
                Scan.created_at >= thirty_days_ago
            ).group_by(func.date(Scan.created_at)).all()

            return {
                'total_users': total_users,
                'total_scans': total_scans,
                'phishing_detected': phishing_detected,
                'legitimate_detected': legitimate_detected,
                'scan_types': scan_types,
                'daily_scans': daily_scans,
                'phishing_rate': (phishing_detected / total_scans * 100) if total_scans > 0 else 0
            }
        finally:
            session.close()

    def get_recent_scans(self, limit=10):
        session = self.Session()
        try:
            scans = session.query(Scan, User.username)\
                .join(User, Scan.user_id == User.id)\
                .order_by(Scan.created_at.desc()).limit(limit).all()
            return [{
                'id': s.Scan.id,
                'username': s.username,
                'scan_type': s.Scan.scan_type,
                'result': s.Scan.result,
                'confidence': s.Scan.confidence,
                'created_at': s.Scan.created_at.isoformat() if s.Scan.created_at else None
            } for s in scans]
        finally:
            session.close()

    def check_username_exists(self, username):
        session = self.Session()
        try:
            return session.query(User).filter_by(username=username).first() is not None
        finally:
            session.close()

    def check_email_exists(self, email):
        session = self.Session()
        try:
            return session.query(User).filter_by(email=email).first() is not None
        finally:
            session.close()


db = Database()