from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from urllib.request import urlretrieve

import fasttext

from cs336_data.common import get_shared_assets_path


NSFW_MODEL_FILE = "dolma_fasttext_nsfw_jigsaw_model.bin"
TOXIC_MODEL_FILE = "dolma_fasttext_hatespeech_jigsaw_model.bin"

NSFW_MODEL_URL = "https://huggingface.co/allenai/dolma-jigsaw-fasttext-bigrams-nsfw/resolve/main/model.bin"
TOXIC_MODEL_URL = "https://huggingface.co/allenai/dolma-jigsaw-fasttext-bigrams-hatespeech/resolve/main/model.bin"


def _model_path(model_file_name: str, model_url: str) -> Path:
    path = get_shared_assets_path() / "classifiers" / model_file_name
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        urlretrieve(model_url, path)
    return path


@lru_cache(maxsize=1)
def _load_nsfw_model():
    return fasttext.load_model(str(_model_path(NSFW_MODEL_FILE, NSFW_MODEL_URL)))


@lru_cache(maxsize=1)
def _load_toxic_model():
    return fasttext.load_model(str(_model_path(TOXIC_MODEL_FILE, TOXIC_MODEL_URL)))


def _predict_label_and_score(model, text: str) -> tuple[str, float]:
    normalized_text = " ".join(text.split())
    if not normalized_text:
        return "unknown", 0.0

    labels, scores = model.predict(normalized_text, k=1)
    label = labels[0].replace("__label__", "")
    score = float(scores[0])
    return label, score


def classify_nsfw(text: str) -> tuple[str, float]:
    return _predict_label_and_score(_load_nsfw_model(), text)


def classify_toxic_speech(text: str) -> tuple[str, float]:
    return _predict_label_and_score(_load_toxic_model(), text)
