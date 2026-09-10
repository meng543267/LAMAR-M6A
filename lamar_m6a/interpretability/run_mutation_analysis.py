#!/usr/bin/env python
"""Run in-silico single-point mutation analysis for LAMAR-m6A.

This script does two things:

1. Select sequences from a prediction CSV, usually correctly predicted label=1
   examples.
2. Generate all single-point mutants and run LAMAR-m6A prediction on them.

The final predicted mutation CSV can later be used for plotting mutation
effect heatmaps.
"""

from __future__ import annotations

import argparse
from argparse import Namespace
from pathlib import Path

import pandas as pd

from lamar_m6a.evaluation.evaluate_single_csv import predict_csv


PREDICT_LABEL_COLUMNS = ["predict_label", "predict", "pred_label", "y_pred"]
PREDICT_PROB_COLUMNS = ["predict_prob", "predict_probs", "pred_prob", "probability", "score"]


def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower = {str(col).lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    return None


def select_sequences(
    input_csv: str | Path,
    label: int = 1,
    predicted_label: int | None = 1,
    max_sequences: int = 100,
    seed: int = 42,
    require_no_n: bool = True,
) -> list[str]:
    df = pd.read_csv(input_csv)
    if "seq" not in df.columns:
        raise ValueError(f"{input_csv} must contain a seq column")
    if "label" not in df.columns:
        raise ValueError(f"{input_csv} must contain a label column for sequence selection")

    selected = df[df["label"].astype(int) == int(label)].copy()
    pred_col = find_column(selected, PREDICT_LABEL_COLUMNS)
    if predicted_label is not None:
        if pred_col is None:
            raise ValueError(
                f"{input_csv} must contain one prediction-label column when --predicted-label is set. "
                f"Accepted names: {PREDICT_LABEL_COLUMNS}"
            )
        selected = selected[selected[pred_col].astype(int) == int(predicted_label)].copy()

    if require_no_n:
        selected = selected[~selected["seq"].astype(str).str.contains("N", case=False, na=False)].copy()

    valid_bases = set("ATGC")
    selected = selected[selected["seq"].astype(str).map(lambda s: bool(s) and set(s.upper()) <= valid_bases)].copy()
    if selected.empty:
        raise ValueError("No sequences matched the selection criteria.")

    selected = selected.sample(n=min(max_sequences, len(selected)), random_state=seed)
    return selected["seq"].astype(str).str.upper().tolist()


def generate_single_point_mutations(seq: str, skip_center_a: bool = False) -> list[dict]:
    bases = ["A", "T", "G", "C"]
    seq = seq.upper()
    rows = [{"seq": seq, "is_original": 1, "original_seq": seq, "position": pd.NA, "mutated_base": pd.NA}]
    center = len(seq) // 2
    for pos, current_base in enumerate(seq):
        if skip_center_a and pos == center and current_base == "A":
            continue
        for new_base in bases:
            if new_base == current_base:
                continue
            mutated = list(seq)
            mutated[pos] = new_base
            rows.append(
                {
                    "seq": "".join(mutated),
                    "is_original": 0,
                    "original_seq": seq,
                    "position": pos + 1,
                    "mutated_base": new_base,
                }
            )
    return rows


def build_mutation_csv(
    input_csv: str | Path,
    output_csv: str | Path,
    label: int,
    predicted_label: int | None,
    max_sequences: int,
    seed: int,
    skip_center_a: bool,
    require_no_n: bool,
) -> pd.DataFrame:
    sequences = select_sequences(
        input_csv=input_csv,
        label=label,
        predicted_label=predicted_label,
        max_sequences=max_sequences,
        seed=seed,
        require_no_n=require_no_n,
    )
    rows = []
    for seq_index, seq in enumerate(sequences):
        for row in generate_single_point_mutations(seq, skip_center_a=skip_center_a):
            row["source_index"] = seq_index
            rows.append(row)

    out = pd.DataFrame(rows)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    print(f"saved mutation CSV: {output_csv}")
    return out


def normalize_prediction_columns(path: str | Path) -> None:
    df = pd.read_csv(path)
    rename = {}
    if "predict_label" in df.columns and "predict" not in df.columns:
        rename["predict_label"] = "predict"
    if "predict_prob" in df.columns and "predict_probs" not in df.columns:
        rename["predict_prob"] = "predict_probs"
    if rename:
        df = df.rename(columns=rename)
        df.to_csv(path, index=False)


def run_mutation_analysis(args: argparse.Namespace) -> None:
    build_mutation_csv(
        input_csv=args.input_csv,
        output_csv=args.mutation_csv,
        label=args.label,
        predicted_label=args.predicted_label,
        max_sequences=args.max_sequences,
        seed=args.seed,
        skip_center_a=args.skip_center_a,
        require_no_n=not args.allow_n,
    )

    eval_args = Namespace(
        input_csv=args.mutation_csv,
        output_csv=args.output_csv,
        checkpoint_path=args.checkpoint_path,
        metrics_json=None,
        metrics_csv=None,
        length=args.length,
        tokenizer_path=args.tokenizer_path,
        model_config=args.model_config,
        model_class=args.model_class,
        model_max_length=args.model_max_length or args.length + 2,
        padding_side=args.padding_side,
        use_fast_tokenizer=args.use_fast_tokenizer,
        num_labels=2,
        token_dropout=False,
        positional_embedding_type=args.positional_embedding_type,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_attention_heads=args.num_attention_heads,
        num_hidden_layers=args.num_hidden_layers,
        head_type=args.head_type,
        kernel_sizes=args.kernel_sizes,
        ocs=args.ocs,
        freeze=args.freeze,
        load_mode=args.load_mode,
        strict_load=args.strict_load,
        batch_size=args.batch_size,
        device=args.device,
    )
    predict_csv(eval_args)
    normalize_prediction_columns(args.output_csv)
    print(f"saved predicted mutation CSV: {args.output_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True, help="Prediction CSV containing seq,label,predict columns.")
    parser.add_argument("--mutation-csv", type=Path, required=True, help="Output CSV of generated mutants before prediction.")
    parser.add_argument("--output-csv", type=Path, required=True, help="Output CSV after model prediction.")
    parser.add_argument("--checkpoint-path", type=Path, required=True)

    parser.add_argument("--label", type=int, default=1)
    parser.add_argument("--predicted-label", type=int, default=1)
    parser.add_argument("--max-sequences", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-center-a", action="store_true")
    parser.add_argument("--allow-n", action="store_true")

    parser.add_argument("--length", type=int, default=801)
    parser.add_argument("--tokenizer-path", type=Path, default=Path("tokenizer/single_nucleotide"))
    parser.add_argument("--model-config", type=Path, default=Path("config/config_150M.json"))
    parser.add_argument("--model-class", default="scripts.sequence_classification_patch:EsmForSequenceClassification")
    parser.add_argument("--model-max-length", type=int, default=None)
    parser.add_argument("--padding-side", choices=["left", "right"], default="left")
    parser.add_argument("--use-fast-tokenizer", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--positional-embedding-type", default="rotary")
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--intermediate-size", type=int, default=3072)
    parser.add_argument("--num-attention-heads", type=int, default=12)
    parser.add_argument("--num-hidden-layers", type=int, default=12)
    parser.add_argument("--head-type", default="Linear")
    parser.add_argument("--kernel-sizes", nargs="+", type=int, default=None)
    parser.add_argument("--ocs", type=int, default=None)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--load-mode", choices=["full", "backbone-only"], default="full")
    parser.add_argument("--strict-load", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    run_mutation_analysis(parse_args())


if __name__ == "__main__":
    main()
