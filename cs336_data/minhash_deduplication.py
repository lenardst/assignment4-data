from __future__ import annotations

import os
import random
import re
import unicodedata
from itertools import combinations
from pathlib import Path

import mmh3
from xopen import xopen


def _normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = " ".join(text.split())
    return text


def _word_ngrams(text: str, ngrams: int) -> set[str]:
    words = text.split()
    if not words:
        return set()
    if len(words) < ngrams:
        return {" ".join(words)}
    return {" ".join(words[i : i + ngrams]) for i in range(len(words) - ngrams + 1)}


def _compute_signature(ngram_set: set[str], num_hashes: int) -> list[int]:
    if not ngram_set:
        return [2**63 - 1] * num_hashes

    signature: list[int] = []
    for seed in range(num_hashes):
        min_value = min(mmh3.hash64(ngram, seed=seed, signed=False)[0] for ngram in ngram_set)
        signature.append(min_value)
    return signature


def _jaccard_similarity(left: set[str], right: set[str]) -> float:
    union_size = len(left | right)
    if union_size == 0:
        return 0.0
    return len(left & right) / union_size


class _UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, index: int) -> int:
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
) -> None:
    documents = []
    for input_path in input_files:
        with xopen(input_path, "rt", encoding="utf-8", errors="replace") as input_file:
            original_text = input_file.read()
        normalized_text = _normalize_text(original_text)
        ngram_set = _word_ngrams(normalized_text, ngrams=ngrams)
        signature = _compute_signature(ngram_set, num_hashes=num_hashes)
        documents.append(
            {
                "path": Path(input_path),
                "text": original_text,
                "ngrams": ngram_set,
                "signature": signature,
            }
        )

    rows_per_band = num_hashes // num_bands
    band_buckets: dict[tuple[int, tuple[int, ...]], list[int]] = {}
    for doc_index, document in enumerate(documents):
        signature = document["signature"]
        for band in range(num_bands):
            start = band * rows_per_band
            end = start + rows_per_band
            band_key = (band, tuple(signature[start:end]))
            band_buckets.setdefault(band_key, []).append(doc_index)

    candidate_pairs: set[tuple[int, int]] = set()
    for bucket_docs in band_buckets.values():
        if len(bucket_docs) < 2:
            continue
        for left, right in combinations(bucket_docs, 2):
            candidate_pairs.add((min(left, right), max(left, right)))

    union_find = _UnionFind(size=len(documents))
    for left, right in candidate_pairs:
        similarity = _jaccard_similarity(documents[left]["ngrams"], documents[right]["ngrams"])
        if similarity >= jaccard_threshold:
            union_find.union(left, right)

    clusters: dict[int, list[int]] = {}
    for doc_index in range(len(documents)):
        root = union_find.find(doc_index)
        clusters.setdefault(root, []).append(doc_index)

    random_generator = random.Random(0)
    kept_indices: set[int] = set()
    for cluster_docs in clusters.values():
        kept_indices.add(random_generator.choice(cluster_docs))

    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    for doc_index in sorted(kept_indices):
        document = documents[doc_index]
        output_path = output_dir / document["path"].name
        with xopen(output_path, "wt", encoding="utf-8") as output_file:
            output_file.write(document["text"])
