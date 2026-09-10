#!/usr/bin/env python
"""Compute SHAP values for LAMAR-m6A and save plotting pickle files.

This script computes SHAP values for selected label=1 and label=0 sequences and
saves two pickle files, matching the old plotting inputs:

    label1_test_500.pkl
    label0_test_500.pkl

It uses SHAP KernelExplainer on token IDs, so it can be slow. Use small sample
sizes first to validate paths and memory.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from lamar_m6a.evaluation.evaluate_single_csv import find_model_state, load_model, stable_softmax


def save_pickle(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(obj, handle)


def load_pickle(path: str | Path):
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def select_sequences(
    input_csv: str | Path,
    label: int,
    max_sequences: int,
    seed: int,
    require_no_n: bool = True,
) -> list[str]:
    df = pd.read_csv(input_csv)
    if "seq" not in df.columns:
        raise ValueError(f"{input_csv} must contain a seq column")
    if "label" not in df.columns:
        raise ValueError(f"{input_csv} must contain a label column")
    selected = df[df["label"].astype(int) == int(label)].copy()
    if require_no_n:
        selected = selected[~selected["seq"].astype(str).str.contains("N", case=False, na=False)].copy()
    valid_bases = set("ATGC")
    selected = selected[selected["seq"].astype(str).map(lambda s: bool(s) and set(s.upper()) <= valid_bases)].copy()
    if selected.empty:
        raise ValueError(f"No sequences found for label={label}")
    selected = selected.sample(n=min(max_sequences, len(selected)), random_state=seed)
    return selected["seq"].astype(str).str.upper().tolist()


def class_shap_vector(shap_values, sample_index: int, class_index: int) -> np.ndarray:
    values = shap_values
    if isinstance(values, list):
        if len(values) > class_index:
            arr = np.asarray(values[class_index])
            return np.asarray(arr[sample_index]).squeeze()
        arr = np.asarray(values[0])
        return np.asarray(arr[sample_index]).squeeze()

    arr = np.asarray(values)
    if arr.ndim == 3:
        if arr.shape[-1] > class_index:
            return arr[sample_index, :, class_index]
        if arr.shape[0] > class_index:
            return arr[class_index, sample_index, :]
    if arr.ndim == 2:
        return arr[sample_index]
    return np.asarray(arr).squeeze()


def compute_shap_for_sequences(
    sequences: list[str],
    tokenizer,
    model,
    device,
    output_pkl: str | Path,
    model_max_length: int,
    class_index: int,
    nsamples: int,
    batch_size: int,
    background_size: int,
) -> list[np.ndarray]:
    import shap
    import torch
    from tqdm import tqdm

    output_pkl = Path(output_pkl)
    if output_pkl.exists():
        shap_values_list = load_pickle(output_pkl)
        print(f"loaded checkpoint {output_pkl}: {len(shap_values_list)} records")
    else:
        shap_values_list = []

    def model_predict(input_ids_np: np.ndarray) -> np.ndarray:
        input_ids = torch.tensor(input_ids_np, dtype=torch.long, device=device)
        attention_mask = (input_ids != tokenizer.pad_token_id).long()
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            probs = stable_softmax(logits.detach().cpu().numpy())
        return probs

    background = np.full((background_size, model_max_length), tokenizer.pad_token_id, dtype=int)
    explainer = shap.KernelExplainer(model_predict, background)

    start = len(shap_values_list)
    remaining = sequences[start:]
    for offset in tqdm(range(0, len(remaining), batch_size), desc=f"SHAP -> {output_pkl.name}"):
        batch_seqs = remaining[offset : offset + batch_size]
        encoded = tokenizer(
            batch_seqs,
            max_length=model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )
        input_ids = encoded["input_ids"]
        batch_shap = explainer.shap_values(input_ids, nsamples=nsamples)
        for i in range(len(batch_seqs)):
            shap_values_list.append(class_shap_vector(batch_shap, i, class_index=class_index))
        save_pickle(shap_values_list, output_pkl)

    return shap_values_list


def run_shap_analysis(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoTokenizer

    model_max_length = args.model_max_length or args.length + 2
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path,
        model_max_length=model_max_length,
        padding_side=args.padding_side,
        use_fast=args.use_fast_tokenizer,
    )

    model_state_path = find_model_state(args.checkpoint_path)
    model = load_model(args, model_state_path, tokenizer)
    device = torch.device(args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    model = model.to(device)
    model.eval()

    label1_sequences = select_sequences(args.input_csv, label=1, max_sequences=args.max_sequences, seed=args.seed)
    label0_sequences = select_sequences(args.input_csv, label=0, max_sequences=args.max_sequences, seed=args.seed)

    compute_shap_for_sequences(
        sequences=label1_sequences,
        tokenizer=tokenizer,
        model=model,
        device=device,
        output_pkl=args.label1_output,
        model_max_length=model_max_length,
        class_index=args.class_index,
        nsamples=args.nsamples,
        batch_size=args.batch_size,
        background_size=args.background_size,
    )
    compute_shap_for_sequences(
        sequences=label0_sequences,
        tokenizer=tokenizer,
        model=model,
        device=device,
        output_pkl=args.label0_output,
        model_max_length=model_max_length,
        class_index=args.class_index,
        nsamples=args.nsamples,
        batch_size=args.batch_size,
        background_size=args.background_size,
    )

    print(f"saved label=1 SHAP: {args.label1_output}")
    print(f"saved label=0 SHAP: {args.label0_output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True, help="CSV with seq,label columns.")
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--label1-output", type=Path, default=Path("outputs/interpretability/shap/label1_test_500.pkl"))
    parser.add_argument("--label0-output", type=Path, default=Path("outputs/interpretability/shap/label0_test_500.pkl"))

    parser.add_argument("--max-sequences", type=int, default=500)
    parser.add_argument("--length", type=int, default=801)
    parser.add_argument("--model-max-length", type=int, default=None)
    parser.add_argument("--class-index", type=int, default=1)
    parser.add_argument("--nsamples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--background-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--tokenizer-path", type=Path, default=Path("tokenizer/single_nucleotide"))
    parser.add_argument("--model-config", type=Path, default=Path("config/config_150M.json"))
    parser.add_argument("--model-class", default="scripts.sequence_classification_patch:EsmForSequenceClassification")
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
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    run_shap_analysis(parse_args())


if __name__ == "__main__":
    main()
