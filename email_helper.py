"""
Transactional email for PhishSentry via Resend's HTTP API.
All sends are best-effort: a provider failure logs an error but never
crashes the request (graceful degradation).
"""
import os
import requests
from logging_config import get_logger

logger = get_logger("phishsentry.email")

RESEND_API_URL = "https://api.resend.com/emails"
RESEND_API_KEY = os.environ.get("RESEND_API_KEY")
RESEND_FROM = os.environ.get("RESEND_FROM", "PhishSentry <noreply@phishsentry.app>")
APP_BASE_URL = os.environ.get("APP_BASE_URL", "http://localhost:8001")


def _send(to, subject, html):
    """Low-level send. Returns True on success, False on failure. Never raises."""
    if not RESEND_API_KEY:
        logger.error("email_not_configured", reason="RESEND_API_KEY missing")
        return False
    try:
        resp = requests.post(
            RESEND_API_URL,
            headers={
                "Authorization": f"Bearer {RESEND_API_KEY}",
                "Content-Type": "application/json",
            },
            json={"from": RESEND_FROM, "to": [to], "subject": subject, "html": html},
            timeout=10,
        )
        if resp.status_code in (200, 201):
            logger.info("email_sent", to=to, subject=subject)
            return True
        logger.error("email_send_failed", to=to, status=resp.status_code, body=resp.text[:200])
        return False
    except requests.exceptions.RequestException as e:
        logger.error("email_send_exception", to=to, error=str(e))
        return False


def send_verification_email(to, token):
    link = f"{APP_BASE_URL}/verify/{token}"
    html = f"""
    <div style="font-family: Arial, sans-serif; max-width: 480px; margin: auto;">
      <h2>Verify your PhishSentry account</h2>
      <p>Welcome! Please confirm your email address to activate your account.</p>
      <p><a href="{link}" style="display:inline-block;padding:12px 20px;
         background:#4f46e5;color:#fff;text-decoration:none;border-radius:6px;">
         Verify my email</a></p>
      <p style="color:#666;font-size:13px;">Or paste this link into your browser:<br>{link}</p>
      <p style="color:#999;font-size:12px;">This link expires in 24 hours. If you didn't sign up, ignore this email.</p>
    </div>"""
    return _send(to, "Verify your PhishSentry account", html)


def send_reset_email(to, token):
    link = f"{APP_BASE_URL}/reset-password/{token}"
    html = f"""
    <div style="font-family: Arial, sans-serif; max-width: 480px; margin: auto;">
      <h2>Reset your PhishSentry password</h2>
      <p>We received a request to reset your password. Click below to choose a new one.</p>
      <p><a href="{link}" style="display:inline-block;padding:12px 20px;
         background:#4f46e5;color:#fff;text-decoration:none;border-radius:6px;">
         Reset password</a></p>
      <p style="color:#666;font-size:13px;">Or paste this link into your browser:<br>{link}</p>
      <p style="color:#999;font-size:12px;">This link expires in 1 hour. If you didn't request this, ignore this email and your password stays unchanged.</p>
    </div>"""
    return _send(to, "Reset your PhishSentry password", html)


def send_welcome_email(to, first_name):
    html = f"""
    <div style="font-family: Arial, sans-serif; max-width: 480px; margin: auto;">
      <h2>Welcome to PhishSentry, {first_name}!</h2>
      <p>Your email is verified and your account is active. You can now sign in and start scanning for phishing threats.</p>
      <p><a href="{APP_BASE_URL}/login" style="display:inline-block;padding:12px 20px;
         background:#4f46e5;color:#fff;text-decoration:none;border-radius:6px;">
         Go to PhishSentry</a></p>
    </div>"""
    return _send(to, "Welcome to PhishSentry", html)