from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm

from cs336_data.modal_utils import VOLUME_MOUNTS, app, build_image


TOKENIZER = AutoTokenizer.from_pretrained("gpt2")
# We intentionally keep full documents; avoid tokenizer length warnings.
TOKENIZER.model_max_length = 10**9


def _tokenize_and_add_eos(text: str) -> list[int]:
    token_ids = TOKENIZER.encode(text)
    return token_ids + [TOKENIZER.eos_token_id]


def _load_texts_from_jsonl(path: Path, text_key: str) -> list[str]:
    texts: list[str] = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = str(row.get(text_key, "")).strip()
            if text:
                texts.append(text)
    return texts


def tokenize_jsonl(
    input_jsonl: Path,
    output_bin: Path,
    text_key: str = "text",
    workers: int = max(1, mp.cpu_count() - 1),
    chunksize: int = 64,
    overwrite: bool = False,
) -> tuple[int, int]:
    input_path = Path(input_jsonl)
    output_path = Path(output_bin)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Use overwrite=True to replace it.")

    texts = _load_texts_from_jsonl(input_path, text_key=text_key)
    if not texts:
        raise ValueError(f"No non-empty texts found in {input_path}")

    with mp.Pool(workers) as pool:
        tokenized_docs = list(
            tqdm(
                pool.imap(_tokenize_and_add_eos, texts, chunksize=chunksize),
                total=len(texts),
                desc="Tokenizing documents",
            )
        )

    token_ids = [token_id for doc in tokenized_docs for token_id in doc]
    token_array = np.array(token_ids, dtype=np.uint16)
    token_array.tofile(output_path)
    return len(texts), len(token_ids)


@app.function(image=build_image(), volumes=VOLUME_MOUNTS, timeout=60 * 60 * 6, cpu=8)
def run_modal_tokenize_data(
    input_jsonl: str = "/root/data/filter_data/filtered_documents.jsonl",
    output_bin: str = "/root/data/filter_data/tokenized_filtered_data.bin",
    text_key: str = "text",
    workers: int = 8,
    chunksize: int = 128,
    overwrite: bool = False,
) -> dict[str, int | str]:
    num_documents, num_tokens = tokenize_jsonl(
        input_jsonl=Path(input_jsonl),
        output_bin=Path(output_bin),
        text_key=text_key,
        workers=workers,
        chunksize=chunksize,
        overwrite=overwrite,
    )
    return {"documents": num_documents, "tokens": num_tokens, "output_bin": output_bin}


@app.local_entrypoint()
def modal_main(
    input_jsonl: str = "/root/data/filter_data/filtered_documents.jsonl",
    output_bin: str = "/root/data/filter_data/tokenized_filtered_data.bin",
    text_key: str = "text",
    workers: int = 8,
    chunksize: int = 128,
    overwrite: bool = False,
) -> None:
    result = run_modal_tokenize_data.remote(
        input_jsonl=input_jsonl,
        output_bin=output_bin,
        text_key=text_key,
        workers=workers,
        chunksize=chunksize,
        overwrite=overwrite,
    )
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", default="outputs/filter_data/filtered_documents.jsonl")
    parser.add_argument("--output-bin", default="outputs/filter_data/tokenized_filtered_data.bin")
    parser.add_argument("--text-key", default="text")
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    parser.add_argument("--chunksize", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    num_documents, num_tokens = tokenize_jsonl(
        input_jsonl=Path(args.input_jsonl),
        output_bin=Path(args.output_bin),
        text_key=args.text_key,
        workers=args.workers,
        chunksize=args.chunksize,
        overwrite=args.overwrite,
    )
    print(f"Documents tokenized: {num_documents}")
    print(f"Total tokens: {num_tokens}")
    print(f"Wrote: {args.output_bin}")


if __name__ == "__main__":
    main()
