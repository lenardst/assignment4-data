from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from urllib.request import urlretrieve

import fasttext

from cs336_data.common import get_shared_assets_path


FASTTEXT_LANGID_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"


def _model_path() -> Path:
    model_path = get_shared_assets_path() / "classifiers" / "lid.176.bin"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if not model_path.exists():
        urlretrieve(FASTTEXT_LANGID_URL, model_path)
    return model_path


@lru_cache(maxsize=1)
def _load_model():
    return fasttext.load_model(str(_model_path()))


def _normalize_label(raw_label: str) -> str:
    clean_label = raw_label.replace("__label__", "")
    if clean_label.startswith("zh"):
        return "zh"
    return clean_label


def identify_language(text: str) -> tuple[str, float]:
    if not text.strip():
        return "unknown", 0.0

    normalized_text = " ".join(text.split())
    model = _load_model()
    predicted_labels, predicted_scores = model.predict(normalized_text, k=1)
    language_code = _normalize_label(predicted_labels[0])
    confidence = float(predicted_scores[0])
    return language_code, confidence
