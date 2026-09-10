#!/usr/bin/env python
"""Evaluate a fine-tuned LAMAR-m6A model across cell-line test splits.

Use this script for the paper-style cross-cell evaluation:

    model trained on A549 -> predict A549/CD8T/HEK293/HeLa/MOLM13 test CSVs

It reads CSV split files such as:

    data/processed/full_splits/test_m6a_full_hek293_801.csv
    data/processed/mature_splits/test_m6a_full_hek293_801.csv

and writes per-target predictions plus a metrics summary table.
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
from pathlib import Path

import numpy as np


DEFAULT_CELL_LINES = ["a549", "cd8t", "hek293", "hela", "molm13"]
DEFAULT_LENGTHS = [51, 101, 201, 401, 801, 1601]


def import_object(import_path: str):
    module_name, object_name = import_path.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, object_name)


def split_dir(data_dir: Path, mode: str) -> Path:
    if mode == "full":
        return data_dir / "processed" / "full_splits"
    if mode == "mature":
        return data_dir / "processed" / "mature_splits"
    raise ValueError(f"Unknown mode: {mode}")


def csv_path_for(data_dir: Path, mode: str, split: str, target_cell: str, length: int) -> Path:
    return split_dir(data_dir, mode) / f"{split}_m6a_full_{target_cell}_{length}.csv"


def find_model_state(path: Path) -> Path:
    """Resolve a checkpoint directory or file to a model state file."""
    if path.is_file():
        return path

    candidates = [
        path / "final_model" / "model.safetensors",
        path / "final_model" / "pytorch_model.bin",
        path / "model.safetensors",
        path / "pytorch_model.bin",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    checkpoint_dirs = [p for p in path.glob("checkpoint-*") if p.is_dir()]
    checkpoint_dirs.sort(key=lambda p: int(re.findall(r"\d+", p.name)[-1]) if re.findall(r"\d+", p.name) else -1)
    for ckpt in reversed(checkpoint_dirs):
        for name in ["model.safetensors", "pytorch_model.bin"]:
            candidate = ckpt / name
            if candidate.exists():
                return candidate

    raise FileNotFoundError(f"No model state file found under: {path}")


def build_checkpoint_path(args, model_cell: str, length: int) -> Path:
    if args.checkpoint_path is not None:
        return args.checkpoint_path
    template = args.checkpoint_template
    return Path(
        template.format(
            mode=args.mode,
            model_cell=model_cell,
            cell=model_cell,
            length=length,
            batch_size=args.batch_size,
            accum_steps=args.gradient_accumulation_steps,
        )
    )


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    probs = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return probs / np.sum(probs, axis=1, keepdims=True)


def compute_metrics(y_true, y_pred, y_prob) -> dict[str, float]:
    from sklearn.metrics import (
        accuracy_score,
        auc,
        confusion_matrix,
        f1_score,
        matthews_corrcoef,
        precision_recall_curve,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }

    if len(set(y_true.tolist())) == 2:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
        out.update(
            {
                "roc_auc": float(roc_auc_score(y_true, y_prob)),
                "pr_auc": float(auc(recall_curve, precision_curve)),
                "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
            }
        )
    else:
        out.update({"roc_auc": float("nan"), "pr_auc": float("nan"), "specificity": float("nan")})
    return out


def load_model(args, model_state_path: Path, tokenizer):
    import torch
    from safetensors.torch import load_model as load_safetensors_model
    from transformers import AutoConfig

    model_cls = import_object(args.model_class)
    config = AutoConfig.from_pretrained(
        args.model_config,
        vocab_size=len(tokenizer),
        pad_token_id=tokenizer.pad_token_id,
        mask_token_id=tokenizer.mask_token_id,
        num_labels=args.num_labels,
        token_dropout=args.token_dropout,
        positional_embedding_type=args.positional_embedding_type,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_attention_heads=args.num_attention_heads,
        num_hidden_layers=args.num_hidden_layers,
    )
    model = model_cls(
        config,
        head_type=args.head_type,
        freeze=False,
        kernel_sizes=args.kernel_sizes,
        ocs=args.ocs,
    )
    if str(model_state_path).endswith(".safetensors"):
        load_safetensors_model(model, filename=str(model_state_path), strict=args.strict_load)
    elif str(model_state_path).endswith(".bin"):
        model.load_state_dict(torch.load(model_state_path, map_location="cpu"), strict=args.strict_load)
    else:
        raise ValueError(f"Unsupported model state file: {model_state_path}")
    return model


def predict_csv(args, model, tokenizer, csv_path: Path, output_path: Path) -> dict[str, float] | None:
    import pandas as pd
    import torch
    from tqdm import tqdm

    df = pd.read_csv(csv_path)
    if "seq" not in df.columns:
        raise ValueError(f"{csv_path} must contain a 'seq' column")

    seqs = df["seq"].astype(str).tolist()
    device = torch.device(args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    model = model.to(device)
    model.eval()

    predict_probs: list[float] = []
    predict_labels: list[int] = []

    with torch.no_grad():
        for i in tqdm(range(0, len(seqs), args.infer_batch_size), desc=csv_path.name):
            batch_seqs = seqs[i : i + args.infer_batch_size]
            batch = tokenizer(
                batch_seqs,
                max_length=args.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            probs = stable_softmax(logits.detach().cpu().numpy())
            predict_probs.extend(probs[:, 1].tolist())
            predict_labels.extend(np.argmax(probs, axis=1).astype(int).tolist())

    result = df.copy()
    result["predict_label"] = predict_labels
    result["predict_prob"] = predict_probs
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    if "label" not in df.columns:
        return None
    metrics = compute_metrics(df["label"].tolist(), predict_labels, predict_probs)
    metrics["n_samples"] = int(len(df))
    return metrics


def run(args) -> None:
    import pandas as pd
    from transformers import AutoTokenizer

    args.model_max_length = args.model_max_length or args.length + 2
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path,
        model_max_length=args.model_max_length,
        padding_side=args.padding_side,
        use_fast=args.use_fast_tokenizer,
    )

    model_cells = args.model_cells or [args.model_cell]
    target_cells = args.target_cells or DEFAULT_CELL_LINES
    summary_rows = []

    for model_cell in model_cells:
        checkpoint_root = build_checkpoint_path(args, model_cell, args.length)
        model_state_path = find_model_state(checkpoint_root)
        print(f"model_cell={model_cell} checkpoint={model_state_path}")
        model = load_model(args, model_state_path, tokenizer)

        for target_cell in target_cells:
            csv_path = args.input_csv or csv_path_for(args.data_dir, args.mode, args.split, target_cell, args.length)
            if not csv_path.exists():
                print(f"skip missing CSV: {csv_path}")
                continue

            output_path = (
                args.outputs_dir
                / "predictions"
                / "cross_cell"
                / args.mode
                / f"model-{model_cell}_target-{target_cell}_{args.length}_{args.split}_pred.csv"
            )
            metrics = predict_csv(args, model, tokenizer, csv_path, output_path)
            if metrics is not None:
                row = {
                    "mode": args.mode,
                    "model_cell": model_cell,
                    "target_cell": target_cell,
                    "length": args.length,
                    "split": args.split,
                    "checkpoint": str(model_state_path),
                    "prediction_csv": str(output_path),
                }
                row.update(metrics)
                summary_rows.append(row)

    if summary_rows:
        metrics_dir = args.outputs_dir / "metrics" / "cross_cell" / args.mode
        metrics_dir.mkdir(parents=True, exist_ok=True)
        summary_path = metrics_dir / f"cross_cell_{args.length}_{args.split}.csv"
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
        print(f"saved metrics summary: {summary_path}")
        json_path = summary_path.with_suffix(".json")
        json_path.write_text(json.dumps(summary_rows, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["full", "mature"], default="full")
    parser.add_argument("--model-cell", choices=DEFAULT_CELL_LINES, default="a549")
    parser.add_argument("--model-cells", nargs="+", choices=DEFAULT_CELL_LINES, default=None)
    parser.add_argument("--target-cells", nargs="+", choices=DEFAULT_CELL_LINES, default=None)
    parser.add_argument("--length", type=int, choices=DEFAULT_LENGTHS, default=801)
    parser.add_argument("--split", default="test")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--input-csv", type=Path, default=None, help="Override input CSV; mostly for debugging.")
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--checkpoint-template",
        default="outputs/checkpoints/{mode}/{model_cell}-{length}-bs{batch_size}-acc{accum_steps}",
    )
    parser.add_argument("--checkpoint-path", type=Path, default=None, help="Override model checkpoint file or directory.")

    parser.add_argument("--tokenizer-path", type=Path, default=Path("tokenizer/single_nucleotide"))
    parser.add_argument("--model-config", type=Path, default=Path("config/config_150M.json"))
    parser.add_argument("--model-class", default="scripts.sequence_classification_patch:EsmForSequenceClassification")
    parser.add_argument("--model-max-length", type=int, default=None)
    parser.add_argument("--padding-side", choices=["left", "right"], default="left")
    parser.add_argument("--use-fast-tokenizer", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--num-labels", type=int, default=2)
    parser.add_argument("--token-dropout", action="store_true")
    parser.add_argument("--positional-embedding-type", default="rotary")
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--intermediate-size", type=int, default=3072)
    parser.add_argument("--num-attention-heads", type=int, default=12)
    parser.add_argument("--num-hidden-layers", type=int, default=12)
    parser.add_argument("--head-type", default="Linear")
    parser.add_argument("--kernel-sizes", nargs="+", type=int, default=None)
    parser.add_argument("--ocs", type=int, default=None)
    parser.add_argument("--strict-load", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batch-size", type=int, default=16, help="Used only for checkpoint-template formatting.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4, help="Used only for checkpoint-template formatting.")
    parser.add_argument("--infer-batch-size", type=int, default=128)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
