# core/email_backends.py
"""
HTTP-based Django email backend using Brevo's transactional email API.

WHY THIS EXISTS
----------------
Render's FREE web services block all outbound traffic to SMTP ports
(25, 465, 587) - see https://render.com/changelog/free-web-services-will-no-longer-allow-outbound-traffic-to-smtp-ports.
That's exactly what caused:

    OSError: [Errno 101] Network is unreachable

when the old SMTP backend tried to connect to smtp.gmail.com:587. It
wasn't a bug in the contact form or the credentials - the free plan's
network simply refuses that connection no matter what SMTP server you
point it at.

The fix is to stop using SMTP entirely and send email over plain HTTPS
instead (port 443 is never blocked). Brevo's free tier (300 emails/day,
no credit card) exposes a normal HTTP API for this, so this backend
implements Django's standard BaseEmailBackend interface but POSTs to
Brevo's API under the hood. Every existing call site - send_mail(),
Django's built-in password-reset emails, the expiring-documents command -
keeps working unchanged, because they all go through EMAIL_BACKEND
rather than calling SMTP directly.

SETUP (free, ~2 minutes)
-------------------------
1. Sign up at https://www.brevo.com (free plan, no card required).
2. Go to Settings -> SMTP & API -> API Keys -> "Generate a new API key".
3. Set these environment variables (e.g. in Render's dashboard):
     BREVO_API_KEY=xkeysib-xxxxxxxx...
     DEFAULT_FROM_EMAIL=your-verified-sender@example.com
4. In Brevo, verify that sender email address (Settings -> Senders) -
   Brevo emails you a confirmation link. Until it's verified, sends fail.

If BREVO_API_KEY isn't set, send_messages() logs a clear warning and
returns 0 sent instead of raising - so a missing key never turns into a
500 error for the user, it just means the email silently isn't sent
(check the logs).
"""
import logging

import requests
from django.conf import settings
from django.core.mail.backends.base import BaseEmailBackend

logger = logging.getLogger(__name__)

BREVO_API_URL = "https://api.brevo.com/v3/smtp/email"
REQUEST_TIMEOUT = 15


class BrevoEmailBackend(BaseEmailBackend):
    """Sends Django EmailMessage objects via Brevo's HTTPS API instead of SMTP."""

    def send_messages(self, email_messages):
        if not email_messages:
            return 0

        api_key = getattr(settings, "BREVO_API_KEY", "")
        if not api_key:
            logger.warning(
                "BREVO_API_KEY isn't set - skipping email send (%d message(s) not sent). "
                "See core/email_backends.py for setup steps.",
                len(email_messages),
            )
            if not self.fail_silently:
                return 0
            return 0

        sent_count = 0
        for message in email_messages:
            if self._send_one(message, api_key):
                sent_count += 1
        return sent_count

    def _send_one(self, message, api_key: str) -> bool:
        from_email = message.from_email or getattr(settings, "DEFAULT_FROM_EMAIL", "")
        if not from_email:
            logger.error("Cannot send email: no from_email/DEFAULT_FROM_EMAIL configured.")
            if self.fail_silently:
                return False
            raise ValueError("No from_email configured for BrevoEmailBackend.")

        # Prefer the HTML alternative if the caller attached one (e.g.
        # EmailMultiAlternatives), otherwise fall back to plain text.
        html_body = None
        for alt_content, alt_mimetype in getattr(message, "alternatives", []) or []:
            if alt_mimetype == "text/html":
                html_body = alt_content
                break

        payload = {
            "sender": {"email": from_email},
            "to": [{"email": addr} for addr in message.to],
            "subject": message.subject,
            "textContent": message.body,
        }
        if html_body:
            payload["htmlContent"] = html_body
        if message.cc:
            payload["cc"] = [{"email": addr} for addr in message.cc]
        if message.bcc:
            payload["bcc"] = [{"email": addr} for addr in message.bcc]
        if message.reply_to:
            payload["replyTo"] = {"email": message.reply_to[0]}

        try:
            response = requests.post(
                BREVO_API_URL,
                headers={
                    "api-key": api_key,
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )
            if response.status_code >= 400:
                logger.error(
                    "Brevo email send failed (%s): %s", response.status_code, response.text
                )
                if not self.fail_silently:
                    return False
                return False
            return True
        except requests.exceptions.RequestException as e:
            logger.error("Brevo email send request failed: %s", e)
            if not self.fail_silently:
                return False
            return False
