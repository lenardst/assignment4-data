from __future__ import annotations

import re


EMAIL_PLACEHOLDER = "|||EMAIL_ADDRESS|||"
PHONE_PLACEHOLDER = "|||PHONE_NUMBER|||"
IP_PLACEHOLDER = "|||IP_ADDRESS|||"


EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)

PHONE_PATTERN = re.compile(
    r"""
    (?<!\d)
    (
        \(\d{3}\)\s*[-.]?\s*\d{3}\s*[-.]?\s*\d{4}
        |
        \d{3}\s*[-.]?\s*\d{3}\s*[-.]?\s*\d{4}
    )
    (?!\d)
    """,
    re.VERBOSE,
)

IPV4_OCTET = r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
IPV4_PATTERN = re.compile(
    rf"(?<!\d)(?:{IPV4_OCTET}\.){{3}}{IPV4_OCTET}(?!\d)"
)


def _mask_with_pattern(text: str, pattern: re.Pattern[str], placeholder: str) -> tuple[str, int]:
    masked_text, replacements = pattern.subn(placeholder, text)
    return masked_text, replacements


def mask_emails(text: str) -> tuple[str, int]:
    return _mask_with_pattern(text, EMAIL_PATTERN, EMAIL_PLACEHOLDER)


def mask_phone_numbers(text: str) -> tuple[str, int]:
    return _mask_with_pattern(text, PHONE_PATTERN, PHONE_PLACEHOLDER)


def mask_ipv4_addresses(text: str) -> tuple[str, int]:
    return _mask_with_pattern(text, IPV4_PATTERN, IP_PLACEHOLDER)
