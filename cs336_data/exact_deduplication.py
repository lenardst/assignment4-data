from __future__ import annotations

import hashlib
import os
from pathlib import Path

from xopen import xopen


def _line_hash(line: str) -> bytes:
    return hashlib.blake2b(line.encode("utf-8"), digest_size=8).digest()


def exact_line_deduplication(input_files: list[os.PathLike], output_directory: os.PathLike) -> None:
    line_counts: dict[bytes, int] = {}

    for input_path in input_files:
        with xopen(input_path, "rt", encoding="utf-8", errors="replace") as input_file:
            for line in input_file:
                line_key = _line_hash(line)
                line_counts[line_key] = line_counts.get(line_key, 0) + 1

    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    for input_path in input_files:
        output_path = output_dir / Path(input_path).name
        with xopen(input_path, "rt", encoding="utf-8", errors="replace") as input_file, xopen(
            output_path, "wt", encoding="utf-8"
        ) as output_file:
            for line in input_file:
                if line_counts.get(_line_hash(line), 0) == 1:
                    output_file.write(line)
