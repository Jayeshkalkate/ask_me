# core/management/commands/check_expiring_documents.py
"""
Scan all users' documents for expiry-like fields (e.g. "Expiry Date",
"Valid Until", "Validity Expiry Date") and email users whose documents are
expiring soon.

Free to run - no paid scheduler needed. Options for triggering this on a
schedule at zero cost:
  1. A cron job on your own server:
        0 8 * * * cd /path/to/project && python manage.py check_expiring_documents
  2. A free GitHub Actions scheduled workflow that calls `python manage.py
     check_expiring_documents` (or hits a small protected Django view that
     calls this command) on a `schedule: cron:` trigger.
  3. A free external pinger (e.g. cron-job.org) hitting a protected endpoint
     that runs this command.

Usage:
    python manage.py check_expiring_documents                 # 30-day window (default)
    python manage.py check_expiring_documents --days 14        # custom window
    python manage.py check_expiring_documents --dry-run         # preview, no emails sent
"""
import logging
from datetime import timedelta

from dateutil import parser as date_parser
from dateutil.parser import ParserError

from django.conf import settings
from django.core.mail import send_mail
from django.core.management.base import BaseCommand
from django.utils import timezone

from core.models import Document
from core.utils import INTERNAL_KEYS

logger = logging.getLogger(__name__)

# Field-name substrings (case-insensitive) that indicate an expiry-style date.
EXPIRY_FIELD_HINTS = ("expiry", "valid until", "validity")


class Command(BaseCommand):
    help = "Email users whose documents have an expiry date within the given window."

    def add_arguments(self, parser):
        parser.add_argument(
            "--days",
            type=int,
            default=30,
            help="Notify about documents expiring within this many days (default: 30).",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Print what would be sent without actually emailing anyone.",
        )

    def handle(self, *args, **options):
        days_ahead = options["days"]
        dry_run = options["dry_run"]
        today = timezone.localdate()
        cutoff = today + timedelta(days=days_ahead)

        checked_fields = 0
        notified = 0

        documents = Document.objects.filter(processed=True).select_related("user")
        for document in documents:
            data = document.display_data
            if not data:
                continue

            for page_data in data.values():
                if not isinstance(page_data, dict):
                    continue

                for field_key, field_value in page_data.items():
                    if field_key.startswith("_") or field_key in INTERNAL_KEYS or not field_value:
                        continue
                    if not any(hint in field_key.lower() for hint in EXPIRY_FIELD_HINTS):
                        continue

                    checked_fields += 1
                    expiry_date = self._parse_date(str(field_value))
                    if not expiry_date:
                        continue

                    if today <= expiry_date <= cutoff:
                        self._notify_user(document, field_key, expiry_date, dry_run)
                        notified += 1

        summary = f"Checked {checked_fields} expiry-like field(s), notified about {notified} document(s)."
        self.stdout.write(self.style.SUCCESS(summary))

    def _parse_date(self, value: str):
        """Best-effort date parsing since OCR'd dates come in inconsistent formats."""
        try:
            # dayfirst=True because most of this app's supported document types are
            # Indian government IDs, which use DD/MM/YYYY.
            return date_parser.parse(value, dayfirst=True, fuzzy=True).date()
        except (ParserError, ValueError, OverflowError, TypeError):
            return None

    def _notify_user(self, document, field_key, expiry_date, dry_run):
        user = document.user
        if not user.email:
            logger.warning(f"User '{user.username}' has no email on file - skipping expiry notice.")
            return

        subject = f"⚠️ Your {document.get_doc_type_display()} is expiring soon"
        message = (
            f"Hi {user.username},\n\n"
            f'Your document "{document.get_doc_type_display()}" has a field '
            f'"{field_key}" set to {expiry_date.strftime("%d %b %Y")}, which is coming up soon.\n\n'
            f"Please renew it if needed, and update the document in ASK_ME once you do.\n\n"
            f"- ASK_ME"
        )

        if dry_run:
            self.stdout.write(f"[DRY RUN] Would email {user.email}: {subject}")
            return

        try:
            send_mail(subject, message, settings.EMAIL_HOST_USER, [user.email])
            self.stdout.write(f"Notified {user.email} about '{field_key}' on document {document.id}.")
        except Exception as e:
            logger.error(f"Failed to send expiry email to {user.email}: {e}")
