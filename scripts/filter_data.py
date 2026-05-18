from __future__ import annotations

import argparse
import concurrent.futures
import glob
import json
import re
from collections import Counter
from pathlib import Path
from typing import TextIO

from fastwarc.warc import ArchiveIterator, WarcRecordType

from cs336_data.gopher_quality import passes_gopher_quality_filter
from cs336_data.harmful_content import classify_nsfw, classify_toxic_speech
from cs336_data.language_id import identify_language
from cs336_data.modal_utils import VOLUME_MOUNTS, app, build_image
from cs336_data.pii_masking import mask_emails, mask_ipv4_addresses, mask_phone_numbers

PREVIEW_CHARS = 300
TEMPLATE_MARKERS = {
    "home",
    "shop",
    "cart",
    "contact",
    "about",
    "privacy",
    "terms",
    "cookie",
    "login",
    "sign",
    "category",
    "tag",
    "search",
    "filter",
    "sort",
    "shipping",
    "apply",
    "read",
    "more",
    "all",
    "videos",
    "menu",
}
TEMPLATE_URI_PATTERN = re.compile(
    r"(?:/tag/|/category/|/listing/|/search|/author/|/wp-content|/page/\d+|\?page=\d+)",
    re.IGNORECASE,
)


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _write_jsonl_row(file: TextIO, row: dict) -> None:
    file.write(json.dumps(row, ensure_ascii=False) + "\n")


def _record_drop(stats: Counter, dropped_file: TextIO, reason: str, row: dict) -> None:
    stats[f"drop_{reason}"] += 1
    _write_jsonl_row(dropped_file, {"reason": reason, **row})


def _mask_pii(text: str) -> tuple[str, int, int, int]:
    masked_text, email_count = mask_emails(text)
    masked_text, phone_count = mask_phone_numbers(masked_text)
    masked_text, ip_count = mask_ipv4_addresses(masked_text)
    return masked_text, email_count, phone_count, ip_count


def _template_marker_density(text: str) -> float:
    tokens = re.findall(r"[a-z]+", text.lower())
    if not tokens:
        return 0.0
    marker_hits = sum(1 for token in tokens if token in TEMPLATE_MARKERS)
    return marker_hits / len(tokens)


def _looks_like_template_uri(uri: str) -> bool:
    return bool(TEMPLATE_URI_PATTERN.search(uri or ""))


def _duplicate_line_ratio(text: str) -> float:
    lines = [re.sub(r"\s+", " ", line.strip().lower()) for line in text.splitlines() if line.strip()]
    if len(lines) < 12:
        return 0.0
    line_counts = Counter(lines)
    duplicate_lines = sum(count - 1 for count in line_counts.values() if count > 1)
    return duplicate_lines / len(lines)


def _process_single_wet_file(
    wet_file: Path,
    worker_output_dir: Path,
    english_threshold: float,
    nsfw_threshold: float,
    toxic_threshold: float,
    template_density_threshold: float,
    duplicate_line_ratio_threshold: float,
    template_min_words: int,
) -> dict:
    worker_output_dir.mkdir(parents=True, exist_ok=True)
    kept_path = worker_output_dir / f"{wet_file.name}.kept.jsonl"
    dropped_path = worker_output_dir / f"{wet_file.name}.dropped.jsonl"

    stats = Counter()
    with kept_path.open("w", encoding="utf-8") as kept_file, dropped_path.open("w", encoding="utf-8") as dropped_file:
        with wet_file.open("rb") as stream:
            for record in ArchiveIterator(stream, parse_http=False):
                if record.record_type != WarcRecordType.conversion:
                    continue

                stats["total_documents"] += 1
                raw_text = record.reader.read().decode("utf-8", errors="replace")
                cleaned_text = _clean_text(raw_text)
                uri = record.headers.get("WARC-Target-URI", "")

                if not cleaned_text:
                    _record_drop(stats, dropped_file, "empty", {"uri": uri})
                    continue

                language, language_score = identify_language(cleaned_text)
                if language != "en" or language_score < english_threshold:
                    _record_drop(
                        stats,
                        dropped_file,
                        "language",
                        {
                            "uri": uri,
                            "language": language,
                            "language_score": language_score,
                            "preview": cleaned_text[:PREVIEW_CHARS],
                        },
                    )
                    continue

                nsfw_label, nsfw_score = classify_nsfw(cleaned_text)
                if nsfw_label == "nsfw" and nsfw_score >= nsfw_threshold:
                    _record_drop(
                        stats,
                        dropped_file,
                        "nsfw",
                        {
                            "uri": uri,
                            "nsfw_score": nsfw_score,
                            "preview": cleaned_text[:PREVIEW_CHARS],
                        },
                    )
                    continue

                toxic_label, toxic_score = classify_toxic_speech(cleaned_text)
                if toxic_label == "toxic" and toxic_score >= toxic_threshold:
                    _record_drop(
                        stats,
                        dropped_file,
                        "toxic",
                        {
                            "uri": uri,
                            "toxic_score": toxic_score,
                            "preview": cleaned_text[:PREVIEW_CHARS],
                        },
                    )
                    continue

                if not passes_gopher_quality_filter(cleaned_text):
                    _record_drop(
                        stats,
                        dropped_file,
                        "gopher",
                        {"uri": uri, "preview": cleaned_text[:PREVIEW_CHARS]},
                    )
                    continue

                word_count = len(cleaned_text.split())
                marker_density = _template_marker_density(cleaned_text)
                if (
                    word_count >= template_min_words
                    and _looks_like_template_uri(uri)
                    and marker_density >= template_density_threshold
                ):
                    _record_drop(
                        stats,
                        dropped_file,
                        "template_uri",
                        {
                            "uri": uri,
                            "marker_density": marker_density,
                            "preview": cleaned_text[:PREVIEW_CHARS],
                        },
                    )
                    continue

                duplicate_line_ratio = _duplicate_line_ratio(raw_text)
                if duplicate_line_ratio >= duplicate_line_ratio_threshold:
                    _record_drop(
                        stats,
                        dropped_file,
                        "template_repetition",
                        {
                            "uri": uri,
                            "duplicate_line_ratio": duplicate_line_ratio,
                            "preview": cleaned_text[:PREVIEW_CHARS],
                        },
                    )
                    continue

                masked_text, email_count, phone_count, ip_count = _mask_pii(cleaned_text)
                stats["kept_documents"] += 1
                stats["masked_emails"] += email_count
                stats["masked_phones"] += phone_count
                stats["masked_ips"] += ip_count
                _write_jsonl_row(
                    kept_file,
                    {
                        "uri": uri,
                        "text": masked_text,
                        "language_score": language_score,
                        "nsfw_score": nsfw_score,
                        "toxic_score": toxic_score,
                    },
                )

    return {"stats": dict(stats), "kept_path": str(kept_path), "dropped_path": str(dropped_path)}


def _merge_jsonl(shard_paths: list[Path], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        for shard_path in shard_paths:
            if not shard_path.exists():
                continue
            with shard_path.open(encoding="utf-8") as shard_file:
                for line in shard_file:
                    output_file.write(line)


def _resolve_output_directory(output_directory: str | Path, run_id: str | None) -> Path:
    output_dir = Path(output_directory)
    if run_id:
        output_dir = output_dir / run_id
    return output_dir


def _assert_output_directory_is_safe(output_directory: Path, overwrite: bool) -> None:
    existing_outputs = [
        output_directory / "filtered_documents.jsonl",
        output_directory / "dropped_documents.jsonl",
        output_directory / "filter_stats.json",
    ]
    if overwrite:
        return
    if any(path.exists() for path in existing_outputs):
        raise FileExistsError(
            f"Output directory already contains pipeline artifacts: {output_directory}. "
            "Use a new run_id or set overwrite=True."
        )


def filter_wet_files(
    wet_files: list[Path],
    output_directory: Path,
    english_threshold: float,
    nsfw_threshold: float,
    toxic_threshold: float,
    template_density_threshold: float = 0.035,
    duplicate_line_ratio_threshold: float = 0.30,
    template_min_words: int = 80,
    workers: int = 1,
) -> dict[str, int]:
    output_directory.mkdir(parents=True, exist_ok=True)
    kept_path = output_directory / "filtered_documents.jsonl"
    dropped_path = output_directory / "dropped_documents.jsonl"
    worker_output_dir = output_directory / "_worker_shards"
    worker_output_dir.mkdir(parents=True, exist_ok=True)

    worker_results = []
    if workers <= 1:
        for wet_file in wet_files:
            worker_results.append(
                _process_single_wet_file(
                    wet_file=wet_file,
                    worker_output_dir=worker_output_dir,
                    english_threshold=english_threshold,
                    nsfw_threshold=nsfw_threshold,
                    toxic_threshold=toxic_threshold,
                    template_density_threshold=template_density_threshold,
                    duplicate_line_ratio_threshold=duplicate_line_ratio_threshold,
                    template_min_words=template_min_words,
                )
            )
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    _process_single_wet_file,
                    wet_file,
                    worker_output_dir,
                    english_threshold,
                    nsfw_threshold,
                    toxic_threshold,
                    template_density_threshold,
                    duplicate_line_ratio_threshold,
                    template_min_words,
                )
                for wet_file in wet_files
            ]
            for future in concurrent.futures.as_completed(futures):
                worker_results.append(future.result())

    merged_stats = Counter()
    kept_shards = [Path(result["kept_path"]) for result in worker_results]
    dropped_shards = [Path(result["dropped_path"]) for result in worker_results]
    for result in worker_results:
        merged_stats.update(result["stats"])

    _merge_jsonl(kept_shards, kept_path)
    _merge_jsonl(dropped_shards, dropped_path)
    return dict(merged_stats)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-glob", default="shared-data/CC/*.warc.wet.gz")
    parser.add_argument("--output-directory", default="outputs/filter_data")
    parser.add_argument("--english-threshold", type=float, default=0.7)
    parser.add_argument("--nsfw-threshold", type=float, default=0.8)
    parser.add_argument("--toxic-threshold", type=float, default=0.9)
    parser.add_argument("--template-density-threshold", type=float, default=0.035)
    parser.add_argument("--duplicate-line-ratio-threshold", type=float, default=0.30)
    parser.add_argument("--template-min-words", type=int, default=80)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--run-id", default=None, help="Optional subdirectory name under output-directory.")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting existing output files.")
    args = parser.parse_args()

    wet_files = [Path(path) for path in sorted(glob.glob(args.input_glob))]
    if not wet_files:
        raise FileNotFoundError(f"No WET files matched: {args.input_glob}")

    output_directory = _resolve_output_directory(args.output_directory, args.run_id)
    _assert_output_directory_is_safe(output_directory, overwrite=args.overwrite)
    stats = filter_wet_files(
        wet_files=wet_files,
        output_directory=output_directory,
        english_threshold=args.english_threshold,
        nsfw_threshold=args.nsfw_threshold,
        toxic_threshold=args.toxic_threshold,
        template_density_threshold=args.template_density_threshold,
        duplicate_line_ratio_threshold=args.duplicate_line_ratio_threshold,
        template_min_words=args.template_min_words,
        workers=args.workers,
    )

    stats_path = output_directory / "filter_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    print(f"Wrote kept docs to {output_directory / 'filtered_documents.jsonl'}")
    print(f"Wrote dropped docs to {output_directory / 'dropped_documents.jsonl'}")
    print(f"Wrote stats to {stats_path}")


@app.function(image=build_image(), volumes=VOLUME_MOUNTS, timeout=60 * 60 * 12, cpu=8)
def run_modal_filter_data(
    input_glob: str = "/shared-data/CC/*.warc.wet.gz",
    output_directory: str = "/root/data/filter_data",
    english_threshold: float = 0.7,
    nsfw_threshold: float = 0.8,
    toxic_threshold: float = 0.9,
    template_density_threshold: float = 0.035,
    duplicate_line_ratio_threshold: float = 0.30,
    template_min_words: int = 80,
    workers: int = 8,
    run_id: str | None = None,
    overwrite: bool = False,
) -> dict[str, int]:
    wet_files = [Path(path) for path in sorted(glob.glob(input_glob))]
    if not wet_files:
        raise FileNotFoundError(f"No WET files matched: {input_glob}")

    output_dir_path = _resolve_output_directory(output_directory, run_id)
    _assert_output_directory_is_safe(output_dir_path, overwrite=overwrite)
    stats = filter_wet_files(
        wet_files=wet_files,
        output_directory=output_dir_path,
        english_threshold=english_threshold,
        nsfw_threshold=nsfw_threshold,
        toxic_threshold=toxic_threshold,
        template_density_threshold=template_density_threshold,
        duplicate_line_ratio_threshold=duplicate_line_ratio_threshold,
        template_min_words=template_min_words,
        workers=workers,
    )
    stats_path = output_dir_path / "filter_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


@app.local_entrypoint()
def modal_main(
    input_glob: str = "/shared-data/CC/*.warc.wet.gz",
    output_directory: str = "/root/data/filter_data",
    english_threshold: float = 0.7,
    nsfw_threshold: float = 0.8,
    toxic_threshold: float = 0.9,
    template_density_threshold: float = 0.035,
    duplicate_line_ratio_threshold: float = 0.30,
    template_min_words: int = 80,
    workers: int = 8,
    run_id: str | None = None,
    overwrite: bool = False,
) -> None:
    stats = run_modal_filter_data.remote(
        input_glob=input_glob,
        output_directory=output_directory,
        english_threshold=english_threshold,
        nsfw_threshold=nsfw_threshold,
        toxic_threshold=toxic_threshold,
        template_density_threshold=template_density_threshold,
        duplicate_line_ratio_threshold=duplicate_line_ratio_threshold,
        template_min_words=template_min_words,
        workers=workers,
        run_id=run_id,
        overwrite=overwrite,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
