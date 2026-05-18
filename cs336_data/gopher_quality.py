from __future__ import annotations

import re


MIN_WORDS = 50
MAX_WORDS = 100_000
MIN_MEAN_WORD_LENGTH = 3.0
MAX_MEAN_WORD_LENGTH = 10.0
MAX_ELLIPSIS_LINE_RATIO = 0.30
MIN_ALPHA_WORD_RATIO = 0.80


def passes_gopher_quality_filter(text: str) -> bool:
    words = re.findall(r"\S+", text)
    if not (MIN_WORDS <= len(words) <= MAX_WORDS):
        return False

    mean_word_length = sum(len(word) for word in words) / len(words)
    if not (MIN_MEAN_WORD_LENGTH <= mean_word_length <= MAX_MEAN_WORD_LENGTH):
        return False

    non_empty_lines = [line.strip() for line in text.splitlines() if line.strip()]
    if non_empty_lines:
        ellipsis_lines = sum(1 for line in non_empty_lines if line.endswith("..."))
        ellipsis_ratio = ellipsis_lines / len(non_empty_lines)
        if ellipsis_ratio > MAX_ELLIPSIS_LINE_RATIO:
            return False

    alpha_word_count = sum(1 for word in words if any(char.isalpha() for char in word))
    alpha_word_ratio = alpha_word_count / len(words)
    if alpha_word_ratio < MIN_ALPHA_WORD_RATIO:
        return False

    return True
