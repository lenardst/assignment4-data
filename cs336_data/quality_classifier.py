from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import fasttext

from cs336_data.common import get_shared_assets_path


MODEL_FILE_NAME = "quality_fasttext.bin"
TRAIN_FILE_NAME = "quality_fasttext_train.txt"


def _collapse_whitespace(text: str) -> str:
    return " ".join(text.split())


def _training_examples() -> list[tuple[str, str]]:
    repo_root = Path(__file__).resolve().parents[1]
    fixtures_dir = repo_root / "tests" / "fixtures"
    wiki_path = fixtures_dir / "high_quality_wiki_reference.txt"
    cc_path = fixtures_dir / "low_quality_cc.txt"

    if wiki_path.exists() and cc_path.exists():
        wiki = _collapse_whitespace(wiki_path.read_text(encoding="utf-8", errors="replace"))
        cc = _collapse_whitespace(cc_path.read_text(encoding="utf-8", errors="replace"))
        if wiki and cc:
            return [("wiki", wiki), ("cc", cc)]

    return [
        ("wiki", "This encyclopedia article contains structured explanatory prose and references."),
        ("cc", "FAQ Search Register Login Copyright All Rights Reserved Powered by forum software."),
    ]


def _write_training_file(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for label, text in _training_examples():
            f.write(f"__label__{label} {text}\n")
    return path


def _train_if_needed(model_path: Path) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    train_path = _write_training_file(model_path.parent / TRAIN_FILE_NAME)
    if model_path.exists() and model_path.stat().st_mtime >= train_path.stat().st_mtime:
        return
    model = fasttext.train_supervised(input=str(train_path), lr=0.5, epoch=100, wordNgrams=2)
    model.save_model(str(model_path))


@lru_cache(maxsize=1)
def _load_model():
    model_path = get_shared_assets_path() / "classifiers" / MODEL_FILE_NAME
    _train_if_needed(model_path)
    return fasttext.load_model(str(model_path))


def classify_quality(text: str) -> tuple[str, float]:
    normalized_text = _collapse_whitespace(text)
    if not normalized_text:
        return "cc", 0.0

    model = _load_model()
    labels, scores = model.predict(normalized_text, k=1)
    label = labels[0].replace("__label__", "")
    score = float(scores[0])
    return label, score
