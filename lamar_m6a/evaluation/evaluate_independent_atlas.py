#!/usr/bin/env python
"""Evaluate LAMAR-m6A checkpoints on independent m6A-Atlas CSV files.

Use this script for independent-test evaluation and later baseline comparison.
It batch-predicts every CSV file in an input folder and saves:

    outputs/predictions/independent_atlas/{mode}/...
    outputs/metrics/independent_atlas/{mode}/...

Each input CSV must contain a 'seq' column. If a 'label' column exists, metrics
are computed; otherwise only predictions are saved.
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
from pathlib import Path

import numpy as np


DEFAULT_LENGTHS = [51, 101, 201, 401, 801, 1601]


def import_object(import_path: str):
    module_name, object_name = import_path.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, object_name)


def find_model_state(path: Path) -> Path:
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
    input_files = sorted(args.input_folder.glob(args.pattern))
    if not input_files:
        raise FileNotFoundError(f"No CSV files matched {args.input_folder / args.pattern}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path,
        model_max_length=args.model_max_length,
        padding_side=args.padding_side,
        use_fast=args.use_fast_tokenizer,
    )
    model_state_path = find_model_state(args.checkpoint_path)
    model = load_model(args, model_state_path, tokenizer)

    prediction_dir = args.outputs_dir / "predictions" / "independent_atlas" / args.mode / args.run_name
    metrics_dir = args.outputs_dir / "metrics" / "independent_atlas" / args.mode
    summary_rows = []

    for csv_path in input_files:
        output_path = prediction_dir / f"{csv_path.stem}_pred.csv"
        metrics = predict_csv(args, model, tokenizer, csv_path, output_path)
        if metrics is not None:
            row = {
                "mode": args.mode,
                "run_name": args.run_name,
                "length": args.length,
                "input_csv": str(csv_path),
                "checkpoint": str(model_state_path),
                "prediction_csv": str(output_path),
            }
            row.update(metrics)
            summary_rows.append(row)

    if summary_rows:
        metrics_dir.mkdir(parents=True, exist_ok=True)
        summary_path = metrics_dir / f"{args.run_name}_{args.length}_metrics.csv"
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
        print(f"saved metrics summary: {summary_path}")
        summary_path.with_suffix(".json").write_text(
            json.dumps(summary_rows, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["full", "mature"], default="full")
    parser.add_argument("--length", type=int, choices=DEFAULT_LENGTHS, default=801)
    parser.add_argument("--input-folder", type=Path, required=True)
    parser.add_argument("--pattern", default="*.csv")
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--run-name", default="lamar")
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))

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
    parser.add_argument("--infer-batch-size", type=int, default=512)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
