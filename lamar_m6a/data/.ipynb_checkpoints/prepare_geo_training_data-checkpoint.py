#!/usr/bin/env python
"""Prepare LAMAR-m6A GEO training datasets for five cell lines.

This script is a clean, generic replacement for the notebook-style GEO
processing workflow. It expects GEO BED sites, a genome FASTA, a GTF file,
and an optional m6A-Atlas table. It produces positive/negative sequence CSV
files, Atlas-filtered negative pools, length-specific datasets, train/valid/test
splits, and optional HuggingFace datasets.
"""

from __future__ import annotations

import argparse
import gzip
import json
import multiprocessing as mp
import pickle
import random
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm


A_MOTIF = re.compile(r"(?=([ACGT][ACGT]A[ACGT][ACGT]))")

ATLAS_COLUMNS = [
    "seqnames",
    "start",
    "end",
    "width",
    "strand",
    "GSE_tech_species_cellLine_treatment_C_count_Coverage",
    "SeqCont",
    "annotation",
    "Gene_Name",
    "Gene_Source",
    "Gene_Biotype",
    "Ensembl_ID",
    "ID",
    "RBP_Num",
    "miRNA_Num",
    "SplicingSite_Num",
    "SNP_Num",
    "Cell_Line",
    "Cell_Line_Num",
    "Technique",
    "Technique_Num",
]

CELL4_ATLAS_GSES = ["GSE63753", "GSE71154", "GSE86336"]
CELL5_ATLAS_GSES = ["GSE63753", "GSE71154", "GSE86336", "GSE98623"]


@dataclass
class Transcript:
    gene_id: str
    transcript_id: str
    chrom: str
    strand: str
    start: int
    end: int
    seq: str

    @property
    def length(self) -> int:
        return len(self.seq)


def open_text(path: str | Path):
    path = str(path)
    if path.endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "r")


def parse_gtf_attributes(attr: str) -> dict[str, str]:
    out = {}
    for part in attr.strip().split(";"):
        part = part.strip()
        if not part:
            continue
        fields = part.split(" ", 1)
        if len(fields) != 2:
            continue
        key, value = fields
        out[key] = value.strip().strip('"')
    return out


def read_gtf_exons(gtf_path: str | Path, transcript_biotype: str | None) -> pd.DataFrame:
    rows = []
    with open_text(gtf_path) as handle:
        for line in tqdm(handle, desc="Reading GTF"):
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9 or parts[2] != "exon":
                continue
            attrs = parse_gtf_attributes(parts[8])
            biotype = attrs.get("transcript_biotype") or attrs.get("gene_biotype")
            if transcript_biotype and biotype != transcript_biotype:
                continue
            gene_id = attrs.get("gene_id")
            transcript_id = attrs.get("transcript_id")
            if not gene_id or not transcript_id:
                continue
            rows.append(
                {
                    "chrom": normalize_chrom(parts[0]),
                    "start": int(parts[3]) - 1,
                    "end": int(parts[4]),
                    "strand": parts[6],
                    "gene_id": gene_id.split(".")[0],
                    "transcript_id": transcript_id.split(".")[0],
                }
            )
    return pd.DataFrame(rows)


def normalize_chrom(chrom: str) -> str:
    chrom = str(chrom)
    if chrom.startswith("chr"):
        chrom = chrom[3:]
    if chrom == "M":
        chrom = "MT"
    return chrom


def read_fasta(fasta_path: str | Path) -> dict[str, str]:
    seqs: dict[str, list[str]] = {}
    name = None
    with open_text(fasta_path) as handle:
        for line in tqdm(handle, desc="Reading FASTA"):
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                raw = line[1:].split()[0].split(".")[0]
                name = normalize_chrom(raw)
                seqs.setdefault(name, [])
            else:
                if name is None:
                    raise ValueError("FASTA sequence found before header")
                seqs[name].append(line.upper().replace("U", "T"))
    return {k: "".join(v) for k, v in seqs.items()}


def reverse_complement(seq: str) -> str:
    table = str.maketrans("ACGTNacgtn", "TGCANtgcan")
    return seq.translate(table)[::-1].upper()


def build_full_transcripts(exons: pd.DataFrame, genome: dict[str, str]) -> list[Transcript]:
    transcripts = []
    grouped = exons.groupby(["gene_id", "transcript_id"], sort=False)
    for (gene_id, transcript_id), sub in tqdm(grouped, desc="Building transcripts"):
        chrom = sub.iloc[0]["chrom"]
        strand = sub.iloc[0]["strand"]
        start = int(sub["start"].min())
        end = int(sub["end"].max())
        if chrom not in genome:
            continue
        seq = genome[chrom][start:end].upper()
        if strand == "-":
            seq = reverse_complement(seq)
        transcripts.append(Transcript(gene_id, transcript_id, chrom, strand, start, end, seq))
    return transcripts


def build_chrom_index(transcripts: list[Transcript]) -> dict[str, list[Transcript]]:
    by_chrom: dict[str, list[Transcript]] = {}
    for tx in transcripts:
        by_chrom.setdefault(tx.chrom, []).append(tx)
    return by_chrom


def genome_to_full_pos(genome_pos_0: int, tx: Transcript) -> int:
    if not (tx.start <= genome_pos_0 < tx.end):
        return -1
    if tx.strand == "+":
        return genome_pos_0 - tx.start
    return tx.end - genome_pos_0 - 1


def iter_bed_sites(bed_path: str | Path):
    with open_text(bed_path) as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 6:
                continue
            chrom = normalize_chrom(parts[0])
            start = int(parts[1])
            strand = parts[5]
            yield chrom, start, strand


def map_bed_to_transcript_sites(
    bed_files: list[str],
    by_chrom: dict[str, list[Transcript]],
) -> pd.DataFrame:
    rows = []
    for bed in bed_files:
        for chrom, genome_pos, strand in tqdm(list(iter_bed_sites(bed)), desc=f"Mapping {Path(bed).name}"):
            candidates = []
            for tx in by_chrom.get(chrom, []):
                if tx.strand != strand:
                    continue
                if not (tx.start <= genome_pos < tx.end):
                    continue
                pos = genome_to_full_pos(genome_pos, tx)
                if pos < 2 or pos + 2 >= tx.length:
                    continue
                if not A_MOTIF.search(tx.seq[pos - 2 : pos + 3]):
                    continue
                if tx.seq[pos] != "A":
                    continue
                candidates.append((tx, pos))
            if not candidates:
                continue
            tx, pos = max(candidates, key=lambda item: item[0].length)
            rows.append(
                {
                    "gene_id": tx.gene_id,
                    "transcript_id": tx.transcript_id,
                    "chrom": tx.chrom,
                    "strand": tx.strand,
                    "genome_pos_0based": genome_pos,
                    "transcript_pos_0based": pos,
                    "source_bed": str(bed),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["gene_id", "transcript_id", "transcript_pos_0based"])
    return pd.DataFrame(rows).drop_duplicates(
        subset=["gene_id", "transcript_id", "transcript_pos_0based"]
    )


def motif_a_positions(seq: str) -> set[int]:
    positions = set()
    for match in A_MOTIF.finditer(seq):
        positions.add(match.start() + 2)
    return positions


def pad_and_extract(seq: str, center: int, length: int) -> str:
    half = (length - 1) // 2
    left = center - half
    right = center + half + 1
    left_pad = "N" * max(0, -left)
    right_pad = "N" * max(0, right - len(seq))
    fragment = seq[max(0, left) : min(len(seq), right)]
    return left_pad + fragment + right_pad


def split_by_label(df: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pos = df[df["label"] == 1]
    neg = df[df["label"] == 0]

    test_pos, temp_pos = train_test_split(pos, test_size=0.5, random_state=seed)
    train_pos, valid_pos = train_test_split(temp_pos, test_size=0.2, random_state=seed)

    test_neg, temp_neg = train_test_split(neg, test_size=0.5, random_state=seed)
    train_neg, valid_neg = train_test_split(temp_neg, test_size=0.2, random_state=seed)

    train = pd.concat([train_pos, train_neg]).sample(frac=1, random_state=seed).reset_index(drop=True)
    valid = pd.concat([valid_pos, valid_neg]).sample(frac=1, random_state=seed).reset_index(drop=True)
    test = pd.concat([test_pos, test_neg]).sample(frac=1, random_state=seed).reset_index(drop=True)
    return train, valid, test


def make_center_101_index(df: pd.DataFrame) -> dict[str, list[int]]:
    index: dict[str, list[int]] = {}
    for i, row in df[df["label"] == 0].iterrows():
        key = pad_and_extract(str(row["seq"]), int(row["center"]), 101)
        index.setdefault(key, []).append(int(i))
    return index


def read_m6a_atlas_table(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", low_memory=False)
    if "SeqCont" not in df.columns:
        df = pd.read_csv(path, sep="\t", names=ATLAS_COLUMNS, header=0, low_memory=False)
    return df


def extract_gse_seq(atlas_df: pd.DataFrame, target_gses: list[str]) -> pd.DataFrame:
    gse_col = "GSE_tech_species_cellLine_treatment_C_count_Coverage"
    pattern = "|".join(target_gses)
    rows = atlas_df[atlas_df[gse_col].astype(str).str.contains(pattern, na=False)]
    extracted = []
    for _, row in rows.iterrows():
        gse_info = str(row[gse_col])
        seq = row["SeqCont"]
        for gse in target_gses:
            if gse in gse_info:
                extracted.append([gse, seq])
    return pd.DataFrame(extracted, columns=["gse", "seq"]).dropna().drop_duplicates().reset_index(drop=True)


def center_101_from_seq(seq: str) -> str:
    seq = str(seq)
    if len(seq) >= 101:
        start = (len(seq) - 101) // 2
        return seq[start : start + 101]
    return seq


def prepare_atlas_outputs(
    atlas_table: str | Path,
    atlas_dir: Path,
    extra_atlas_csv: str | Path | None = None,
) -> set[str]:
    atlas_dir.mkdir(parents=True, exist_ok=True)
    atlas_df = read_m6a_atlas_table(atlas_table)

    cell4_seq = extract_gse_seq(atlas_df, CELL4_ATLAS_GSES)
    cell4_seq.to_csv(atlas_dir / "cell4_seq.csv", index=False)

    cell5_seq = extract_gse_seq(atlas_df, CELL5_ATLAS_GSES)
    cell5_seq.to_csv(atlas_dir / "cell5_seq.csv", index=False)

    for gse in CELL5_ATLAS_GSES:
        gse_seq = cell5_seq[cell5_seq["gse"] == gse].drop_duplicates().reset_index(drop=True)
        gse_seq.to_csv(atlas_dir / f"{gse}_seq.csv", index=False)
        print(f"{gse}_seq.csv", len(gse_seq))

    all_seq = atlas_df[["SeqCont"]].dropna().drop_duplicates().rename(columns={"SeqCont": "seq"})
    all_seq.to_csv(atlas_dir / "m6a_atlas_seq_data.csv", index=False)

    atlas_sources = [all_seq[["seq"]]]
    if extra_atlas_csv:
        extra = pd.read_csv(extra_atlas_csv)
        if "seq" not in extra.columns:
            raise ValueError(f"{extra_atlas_csv} must contain a seq column")
        atlas_sources.append(extra[["seq"]])

    atlas_for_filter = pd.concat(atlas_sources, ignore_index=True).drop_duplicates()
    atlas_101 = set(atlas_for_filter["seq"].astype(str).map(center_101_from_seq))

    print("cell4_seq.csv", len(cell4_seq))
    print("cell5_seq.csv", len(cell5_seq))
    print("m6a_atlas_seq_data.csv", len(all_seq))
    print("Atlas 101nt keys for filtering", len(atlas_101))
    return atlas_101


def save_tokenized_dataset(
    splits_dir: Path,
    tokenized_dir: Path,
    tokenizer_path: str | Path,
    cell: str,
    length: int,
) -> None:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, model_max_length=length + 2)
    dataset = load_dataset(
        "csv",
        data_files={
            "train": str(splits_dir / f"train_m6a_full_{cell}_{length}.csv"),
            "valid": str(splits_dir / f"valid_m6a_full_{cell}_{length}.csv"),
            "test": str(splits_dir / f"test_m6a_full_{cell}_{length}.csv"),
        },
    )

    def process_function(examples):
        tokenized = tokenizer(examples["seq"], max_length=length + 2, truncation=True)
        tokenized["labels"] = examples["label"]
        return tokenized

    processed = dataset.map(
        process_function,
        batched=True,
        num_proc=mp.cpu_count(),
        remove_columns=dataset["train"].column_names,
    )
    out_path = tokenized_dir / f"full-{cell}" / f"m6a_full_{cell}_{length}"
    processed.save_to_disk(out_path, num_proc=mp.cpu_count())
    print("saved tokenized dataset", out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="geo_training_config.example.json")
    parser.add_argument("--out-dir", default="data")
    parser.add_argument("--m6a-atlas-table", default="data/raw/m6a_atlas/hg38_CL_Tech.txt")
    parser.add_argument("--atlas-seq-csv", default=None)
    parser.add_argument("--use-atlas-filter", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--lengths", nargs="+", type=int, default=[51, 101, 201, 401, 801, 1601])
    parser.add_argument("--pool-negative-ratio", type=int, default=100)
    parser.add_argument("--final-negative-ratio", type=int, default=10)
    parser.add_argument("--negative-ratio", type=int, default=None, help="Deprecated alias for --final-negative-ratio.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.negative_ratio is not None:
        args.final_negative_ratio = args.negative_ratio

    random.seed(args.seed)
    out_dir = Path(args.out_dir)
    index_dir = out_dir / "intermediate" / "full_index"
    atlas_dir = out_dir / "processed" / "atlas"
    full_dir = out_dir / "processed" / "full_sequences"
    pool_dir = out_dir / "processed" / "full_negative_pools"
    balanced_dir = out_dir / "processed" / "full_balanced"
    lengths_dir = out_dir / "processed" / "full_lengths"
    splits_dir = out_dir / "processed" / "full_splits"
    tokenized_dir = out_dir / "processed" / "full_tokenized"
    for path in [index_dir, atlas_dir, full_dir, pool_dir, balanced_dir, lengths_dir, splits_dir, tokenized_dir]:
        path.mkdir(parents=True, exist_ok=True)

    if args.use_atlas_filter:
        atlas_table = Path(args.m6a_atlas_table)
        if not atlas_table.exists():
            raise FileNotFoundError(
                f"m6A-Atlas table not found: {atlas_table}. "
                "Put hg38_CL_Tech.txt there or rerun with --no-use-atlas-filter."
            )
        atlas_101 = prepare_atlas_outputs(atlas_table, atlas_dir, args.atlas_seq_csv)
    else:
        atlas_101 = set()

    config = json.loads(Path(args.config).read_text())
    ref = config["reference"]
    exons = read_gtf_exons(ref["gtf"], ref.get("transcript_biotype"))
    genome = read_fasta(ref["fasta"])
    transcripts = build_full_transcripts(exons, genome)
    tx_by_key = {(tx.gene_id, tx.transcript_id): tx for tx in transcripts}
    by_chrom = build_chrom_index(transcripts)

    positive_by_cell: dict[str, pd.DataFrame] = {}
    global_positive: dict[tuple[str, str], set[int]] = {}

    for cell, meta in config["cell_lines"].items():
        pos_df = map_bed_to_transcript_sites(meta["bed_files"], by_chrom)
        pos_df["cell_line"] = cell
        positive_by_cell[cell] = pos_df
        for row in pos_df.itertuples():
            key = (row.gene_id, row.transcript_id)
            global_positive.setdefault(key, set()).add(int(row.transcript_pos_0based))

    for cell, pos_df in positive_by_cell.items():
        pos_records = []
        neg_records = []
        for (gene_id, transcript_id), sub in pos_df.groupby(["gene_id", "transcript_id"]):
            tx = tx_by_key.get((gene_id, transcript_id))
            if tx is None:
                continue
            pos_set = set(map(int, sub["transcript_pos_0based"]))
            all_global_pos = global_positive.get((gene_id, transcript_id), set())
            neg_candidates = motif_a_positions(tx.seq) - all_global_pos

            for pos in sorted(pos_set):
                pos_records.append({"seq": tx.seq, "center": pos, "label": 1, "gene_id": gene_id, "transcript_id": transcript_id})
            for pos in sorted(neg_candidates):
                neg_records.append({"seq": tx.seq, "center": pos, "label": 0, "gene_id": gene_id, "transcript_id": transcript_id})

        pos_table = pd.DataFrame(pos_records).drop_duplicates(subset=["transcript_id", "center"])
        neg_table = pd.DataFrame(neg_records).drop_duplicates(subset=["transcript_id", "center"])
        pos_table.to_csv(full_dir / f"m6a_positive_{cell}.csv", index=False)
        neg_table.to_csv(full_dir / f"m6a_negative_{cell}.csv", index=False)

        n_pool_neg = len(pos_table) * args.pool_negative_ratio
        if len(neg_table) < n_pool_neg:
            raise ValueError(f"{cell}: not enough negatives for 1:{args.pool_negative_ratio} pool: need {n_pool_neg}, found {len(neg_table)}")
        neg_pool = neg_table.sample(n=n_pool_neg, random_state=args.seed).reset_index(drop=True)
        pool_table = pd.concat([pos_table, neg_pool], ignore_index=True)
        pool_table.to_csv(pool_dir / f"m6a_{cell}_sample_100negative.csv", index=False)

        index_101 = make_center_101_index(pool_table)
        with open(index_dir / f"m6a_full_{cell}_sample100_101_index.pkl", "wb") as handle:
            pickle.dump(index_101, handle)

        if atlas_101:
            filtered_neg = neg_pool.copy()
            filtered_neg["center_101"] = [
                pad_and_extract(seq, int(center), 101)
                for seq, center in zip(filtered_neg["seq"], filtered_neg["center"])
            ]
            filtered_neg = filtered_neg[~filtered_neg["center_101"].isin(atlas_101)].drop(columns=["center_101"])
            filtered_neg = filtered_neg.reset_index(drop=True)
        else:
            filtered_neg = neg_pool
        filtered_neg.to_csv(pool_dir / f"m6a_full_{cell}_sample100_allneg_filtered.csv", index=False)

        n_final_neg = len(pos_table) * args.final_negative_ratio
        if len(filtered_neg) < n_final_neg:
            raise ValueError(f"{cell}: not enough negatives for 1:{args.final_negative_ratio}: need {n_final_neg}, found {len(filtered_neg)}")
        final_neg = filtered_neg.sample(n=n_final_neg, random_state=args.seed).reset_index(drop=True)
        site_table = pd.concat([pos_table, final_neg], ignore_index=True)
        site_table.to_csv(balanced_dir / f"m6a_full_{cell}_sample_pos_10neg.csv", index=False)

        for length in args.lengths:
            df = site_table.copy()
            df["seq"] = [pad_and_extract(seq, int(center), length) for seq, center in zip(df["seq"], df["center"])]
            df = df[["seq", "label", "gene_id", "transcript_id", "center"]]
            df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
            df.to_csv(lengths_dir / f"m6a_full_{cell}_{length}.csv", index=False)

            train, valid, test = split_by_label(df, args.seed)
            train.to_csv(splits_dir / f"train_m6a_full_{cell}_{length}.csv", index=False)
            valid.to_csv(splits_dir / f"valid_m6a_full_{cell}_{length}.csv", index=False)
            test.to_csv(splits_dir / f"test_m6a_full_{cell}_{length}.csv", index=False)

            print(cell, length, "pos", int((df["label"] == 1).sum()), "neg", int((df["label"] == 0).sum()))
            if args.tokenizer_path:
                save_tokenized_dataset(splits_dir, tokenized_dir, args.tokenizer_path, cell, length)


if __name__ == "__main__":
    main()
