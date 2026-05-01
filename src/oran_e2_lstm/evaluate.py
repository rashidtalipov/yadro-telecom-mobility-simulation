from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from common import load_json, save_json
from model import ModelConfig, MultitaskLstmPredictor
from train import create_loader, evaluate_loop, resolve_device


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate multitask LSTM handover predictor.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", type=str, default="test", choices=("train", "val", "test"))
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--window-stride", type=int, default=1)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    metadata = load_json(args.dataset_dir / "metadata.json")
    split_frame = pd.read_parquet(args.dataset_dir / f"{args.split}_rows.parquet")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_config = ModelConfig(**checkpoint["model_config"])
    candidate_top_k = checkpoint["training_args"].get("candidate_top_k")
    if candidate_top_k in ("None", "", None):
        candidate_top_k = None
    else:
        candidate_top_k = int(candidate_top_k)

    loader = create_loader(
        frame=split_frame,
        seq_len=int(metadata["seq_len"]),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        window_stride=args.window_stride,
        candidate_top_k=candidate_top_k if model_config.target_mode == "candidate" else None,
        num_cells=len(metadata["cell_ids"]),
    )

    model = MultitaskLstmPredictor(model_config)
    model.load_state_dict(checkpoint["model_state"])

    device = resolve_device(args.device)
    model = model.to(device)
    trigger_criterion = torch.nn.BCEWithLogitsLoss()
    metrics, losses = evaluate_loop(
        model=model,
        loader=loader,
        trigger_criterion=trigger_criterion,
        device=device,
        target_loss_weight=float(checkpoint["training_args"]["target_loss_weight"]),
        target_mode=model_config.target_mode,
        limit_batches=None,
    )
    summary = {
        "split": args.split,
        "target_mode": model_config.target_mode,
        "candidate_top_k": candidate_top_k,
        "losses": losses,
        "metrics": metrics,
        "checkpoint": str(args.checkpoint.resolve()),
    }
    print(summary)
    if args.output_json is not None:
        save_json(args.output_json, summary)


if __name__ == "__main__":
    main()
