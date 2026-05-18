from __future__ import annotations

from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding


def extract_text_from_html_bytes(html_bytes: bytes) -> str:
    try:
        decoded_html = html_bytes.decode("utf-8")
    except UnicodeDecodeError:
        detected_encoding = detect_encoding(html_bytes) or "utf-8"
        decoded_html = html_bytes.decode(detected_encoding, errors="replace")

    return extract_plain_text(decoded_html)
