from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from filter_data import run_modal_filter_data
from tokenize_data import run_modal_tokenize_data
from train import run_modal_training


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-glob", default="/shared-data/english-wet-data/*.warc.wet.gz")
    parser.add_argument("--run-root", default="/root/data/runs")
    parser.add_argument(
        "--run-id",
        default=None,
        help="Unique run identifier (default: UTC timestamp, e.g. 20260517_123456).",
    )
    parser.add_argument("--english-threshold", type=float, default=0.7)
    parser.add_argument("--nsfw-threshold", type=float, default=0.8)
    parser.add_argument("--toxic-threshold", type=float, default=0.9)
    parser.add_argument("--template-density-threshold", type=float, default=0.035)
    parser.add_argument("--duplicate-line-ratio-threshold", type=float, default=0.30)
    parser.add_argument("--template-min-words", type=int, default=80)
    parser.add_argument("--tokenize-workers", type=int, default=8)
    parser.add_argument("--tokenize-chunksize", type=int, default=128)
    parser.add_argument("--launch-training", action="store_true", help="Launch training after tokenization.")
    args = parser.parse_args()
    run_id = args.run_id or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_root = args.run_root.rstrip("/")
    output_directory = f"{run_root}/{run_id}/filter"
    tokenized_bin = f"{run_root}/{run_id}/tokenized/tokenized_filtered_data.bin"
    model_output = f"{run_root}/{run_id}/model"

    print("Starting Modal filter stage...")
    filter_stats = run_modal_filter_data.remote(
        input_glob=args.input_glob,
        output_directory=output_directory,
        english_threshold=args.english_threshold,
        nsfw_threshold=args.nsfw_threshold,
        toxic_threshold=args.toxic_threshold,
        template_density_threshold=args.template_density_threshold,
        duplicate_line_ratio_threshold=args.duplicate_line_ratio_threshold,
        template_min_words=args.template_min_words,
        overwrite=False,
    )
    print("Filter stats:")
    print(json.dumps(filter_stats, indent=2))

    filtered_jsonl = f"{output_directory}/filtered_documents.jsonl"
    print("Starting Modal tokenize stage...")
    tokenize_stats = run_modal_tokenize_data.remote(
        input_jsonl=filtered_jsonl,
        output_bin=tokenized_bin,
        workers=args.tokenize_workers,
        chunksize=args.tokenize_chunksize,
        overwrite=False,
    )
    print("Tokenize stats:")
    print(json.dumps(tokenize_stats, indent=2))

    if args.launch_training:
        print("Launching Modal training stage...")
        function_call = run_modal_training.spawn(
            train_bin=tokenized_bin,
            model_output=model_output,
            overwrite_model_output=False,
        )
        print(f"Training launched. Function call id: {function_call.object_id}")
        print("Use Modal dashboard/app logs to monitor progress.")


if __name__ == "__main__":
    main()
