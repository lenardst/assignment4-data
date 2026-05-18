from __future__ import annotations

import argparse
import json
from pathlib import Path

import modal
import numpy as np
import torch
import torch.nn.functional as F

from cs336_basics.data import get_batch
from cs336_basics.model import BasicsTransformerLM
from cs336_data.modal_utils import VOLUME_MOUNTS, app, build_image


def _normalize_state_dict_keys(raw_state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    normalized: dict[str, torch.Tensor] = {}
    for key, value in raw_state_dict.items():
        key = key.removeprefix("module.")
        key = key.removeprefix("_orig_mod.")
        key = key.removeprefix("module._orig_mod.")
        normalized[key] = value
    return normalized


@torch.no_grad()
def estimate_dev_loss(
    model: BasicsTransformerLM,
    dev_dataset: np.ndarray,
    batch_size: int,
    eval_iters: int,
    device: str,
    context_length: int,
) -> float:
    model.eval()
    losses = torch.zeros(eval_iters, device=device)
    for i in range(eval_iters):
        batch_x, batch_y = get_batch(
            dev_dataset,
            batch_size=batch_size,
            context_length=context_length,
            device=device,
        )
        logits = model(batch_x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), batch_y.view(-1))
        losses[i] = loss.item()
    return float(losses.mean().item())


@app.function(image=build_image(), volumes=VOLUME_MOUNTS, gpu="B200:1", timeout=60 * 60)
def evaluate_remote(
    model_dir: str,
    valid_bin: str = "/shared-data/tokenized_paloma_c4_100_domains_validation.bin",
    eval_batch_size: int = 128,
    eval_iters: int = 1000,
    output_json: str | None = None,
) -> float:
    model_dir_path = Path(model_dir)
    model_config = json.loads((model_dir_path / "model_config.json").read_text())
    model = BasicsTransformerLM(**model_config)
    state_dict = torch.load(model_dir_path / "model.pt", map_location="cpu")
    model.load_state_dict(_normalize_state_dict_keys(state_dict), strict=False)
    model = model.to("cuda")

    dev_data = np.memmap(valid_bin, dtype=np.uint16, mode="r")
    loss = estimate_dev_loss(
        model=model,
        dev_dataset=dev_data,
        batch_size=eval_batch_size,
        eval_iters=eval_iters,
        device="cuda",
        context_length=model_config["context_length"],
    )
    if output_json:
        output_path = Path(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps({"validation_loss": float(loss), "eval_iters": eval_iters}), encoding="utf-8")
    return float(loss)


@app.local_entrypoint()
def modal_main(
    model_dir: str = "/root/data/output/your_data",
    valid_bin: str = "/shared-data/tokenized_paloma_c4_100_domains_validation.bin",
    eval_batch_size: int = 128,
    eval_iters: int = 1000,
    output_json: str | None = None,
) -> None:
    loss = evaluate_remote.remote(
        model_dir=model_dir,
        valid_bin=valid_bin,
        eval_batch_size=eval_batch_size,
        eval_iters=eval_iters,
        output_json=output_json,
    )
    print(f"Estimated validation loss: {loss:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="/root/data/output/your_data")
    parser.add_argument("--valid-bin", default="/shared-data/tokenized_paloma_c4_100_domains_validation.bin")
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--eval-iters", type=int, default=1000)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    if modal.is_local():
        loss = evaluate_remote.local(
            model_dir=args.model_dir,
            valid_bin=args.valid_bin,
            eval_batch_size=args.eval_batch_size,
            eval_iters=args.eval_iters,
            output_json=args.output_json,
        )
        print(f"Estimated validation loss: {loss:.6f}")
