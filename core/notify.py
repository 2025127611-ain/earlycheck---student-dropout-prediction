"""Follow-up links: High risk -> WhatsApp, Medium risk -> e-mail.

Nothing is sent automatically and nothing is stored. The system only builds a link
(wa.me / Gmail / mailto) with a pre-written message that the advisor can edit and
approve before pressing Send in WhatsApp / the mail app.
"""
from __future__ import annotations

import re
from urllib.parse import quote

from . import config as cfg

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def valid_email(email: str) -> bool:
    return bool(_EMAIL_RE.match((email or "").strip()))


def normalise_phone(phone: str, country_code: str = cfg.DEFAULT_COUNTRY_CODE) -> str | None:
    """'012-345 6789' -> '60123456789'. Returns None if it does not look like a phone number."""
    p = (phone or "").strip()
    if not p:
        return None
    digits = re.sub(r"\D", "", p)
    if p.startswith("+"):
        pass                                   # already international
    elif digits.startswith("00"):
        digits = digits[2:]
    elif digits.startswith("0"):
        digits = country_code + digits[1:]
    elif not digits.startswith(country_code):
        digits = country_code + digits
    return digits if 9 <= len(digits) <= 15 else None


def default_message(level: str, prob: float, reasons: list[str], actions: list[str],
                    name: str = "") -> tuple[str, str]:
    """(subject, body) for the follow-up message."""
    hi = f"Hi {name}," if name else "Hi,"
    r = "\n".join(f"- {x}" for x in reasons) or "- (no single dominant factor)"
    a = "\n".join(f"- {x}" for x in actions)
    subject = f"Academic check-in ({level.lower()} priority)"
    body = (f"{hi}\n\nOur early-warning system flagged an academic follow-up "
            f"(estimated dropout risk {min(max(prob * 100, 1), 99):.0f}% - {level}). This is only a prediction to help us "
            f"offer support early, not a final judgement.\n\n"
            f"Main factors noticed:\n{r}\n\nSuggested support:\n{a}\n\n"
            f"Please reply or contact your academic advisor so we can help.\n\nThank you.")
    return subject, body


def whatsapp_link(phone: str, message: str) -> str | None:
    n = normalise_phone(phone)
    return f"https://wa.me/{n}?text={quote(message)}" if n else None


def gmail_link(email: str, subject: str, body: str) -> str | None:
    if not valid_email(email):
        return None
    return (f"https://mail.google.com/mail/?view=cm&fs=1&to={quote(email.strip())}"
            f"&su={quote(subject)}&body={quote(body)}")


def mailto_link(email: str, subject: str, body: str) -> str | None:
    if not valid_email(email):
        return None
    return f"mailto:{quote(email.strip())}?subject={quote(subject)}&body={quote(body)}"
