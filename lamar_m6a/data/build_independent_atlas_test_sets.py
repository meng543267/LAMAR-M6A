#!/usr/bin/env python
"""Build independent m6A-Atlas test-set CSV files for LAMAR-m6A.

This script cleans up the independent-test logic from newdesign-origin-atlas.
It supports two useful stages:

1. Build an intermediate .data pickle from a raw m6A-Atlas species table.
2. Export final per-experiment CSV files from that .data pickle.

The final CSV files have the schema used by the old rat-full directory:

    seq,label

By default, final files are written to:

    data/processed/independent_test_sets/atlas/full/{species}/

To reproduce an old path exactly, pass for example:

    --output-dir lamar-data-new/atlas-a/rat-full
"""

from __future__ import annotations

import argparse
import gzip
import pickle
import random
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from tqdm import tqdm


A_MOTIF = re.compile(r"(?=([ACGT][ACGT]A[ACGT][ACGT]))")

DEFAULT_ATLAS_TABLES = {
    "human": ["data/raw/independent_atlas/m6A-Atlas2_Human_Site.txt"],
    "mouse": [
        "data/raw/independent_atlas/m6A-Atlas2_Mouse_Site_1.txt",
        "data/raw/independent_atlas/m6A-Atlas2_Mouse_Site_2.txt",
    ],
    "rat": ["data/raw/independent_atlas/m6A-Atlas2_Rat_Site.txt"],
}

DEFAULT_REFERENCE_FILES = {
    "human": {
        "gtf": "data/raw/reference/Homo_sapiens.GRCh38.79.gtf.gz",
        "fasta": "data/raw/reference/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz",
    },
    "mouse": {
        "gtf": "data/raw/reference/Mus_musculus.GRCm38.102.gtf.gz",
        "fasta": "data/raw/reference/Mus_musculus.GRCm38.dna.primary_assembly.fa.gz",
    },
    "rat": {
        "gtf": "data/raw/reference/Rattus_norvegicus.Rnor_6.0.104.gtf.gz",
        "fasta": "data/raw/reference/Rattus_norvegicus.Rnor_6.0.dna.toplevel.fa.gz",
    },
}


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


def normalize_chrom(chrom: str) -> str:
    chrom = str(chrom)
    if chrom.startswith("chr"):
        chrom = chrom[3:]
    chrom = chrom.split(".")[0]
    if chrom == "M":
        chrom = "MT"
    return chrom


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


def read_gtf_exons(gtf_path: str | Path, transcript_biotype: str | None = "protein_coding") -> pd.DataFrame:
    rows = []
    with open_text(gtf_path) as handle:
        for line in tqdm(handle, desc="Reading GTF"):
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9 or parts[2] != "exon":
                continue
            attrs = parse_gtf_attributes(parts[8])
            biotype = (
                attrs.get("transcript_biotype")
                or attrs.get("transcript_type")
                or attrs.get("gene_biotype")
                or attrs.get("gene_type")
            )
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
    if not rows:
        raise ValueError(f"No exon rows were read from {gtf_path}")
    return pd.DataFrame(rows)


def read_fasta(fasta_path: str | Path) -> dict[str, str]:
    seqs: dict[str, list[str]] = {}
    name = None
    with open_text(fasta_path) as handle:
        for line in tqdm(handle, desc="Reading FASTA"):
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                raw = line[1:].split()[0]
                name = normalize_chrom(raw)
                seqs.setdefault(name, [])
            else:
                if name is None:
                    raise ValueError("FASTA sequence found before any header")
                seqs[name].append(line.upper().replace("U", "T"))
    return {k: "".join(v) for k, v in seqs.items()}


def reverse_complement(seq: str) -> str:
    table = str.maketrans("ACGTNacgtn", "TGCANtgcan")
    return seq.translate(table)[::-1].upper()


def build_full_transcripts(exons: pd.DataFrame, genome: dict[str, str]) -> list[Transcript]:
    transcripts = []
    grouped = exons.groupby(["gene_id", "transcript_id"], sort=False)
    for (gene_id, transcript_id), sub in tqdm(grouped, desc="Building full transcripts"):
        chrom = str(sub.iloc[0]["chrom"])
        strand = str(sub.iloc[0]["strand"])
        start = int(sub["start"].min())
        end = int(sub["end"].max())
        if chrom not in genome:
            continue
        seq = genome[chrom][start:end].upper()
        if strand == "-":
            seq = reverse_complement(seq)
        transcripts.append(Transcript(gene_id, transcript_id, chrom, strand, start, end, seq))
    if not transcripts:
        raise ValueError("No transcripts were built. Check chromosome names and reference files.")
    return transcripts


def build_chrom_index(transcripts: list[Transcript]) -> dict[str, list[Transcript]]:
    out: dict[str, list[Transcript]] = {}
    for tx in transcripts:
        out.setdefault(tx.chrom, []).append(tx)
    return out


def genome_to_full_pos(genome_pos_0: int, tx: Transcript) -> int:
    if not (tx.start <= genome_pos_0 < tx.end):
        return -1
    if tx.strand == "+":
        return genome_pos_0 - tx.start
    return tx.end - genome_pos_0 - 1


def motif_a_positions(seq: str) -> set[int]:
    return {match.start() + 2 for match in A_MOTIF.finditer(str(seq).upper())}


def pad_and_extract(seq: str, center: int, length: int) -> str:
    half = (length - 1) // 2
    left = center - half
    right = center + half + 1
    left_pad = "N" * max(0, -left)
    right_pad = "N" * max(0, right - len(seq))
    fragment = seq[max(0, left) : min(len(seq), right)]
    return left_pad + fragment + right_pad


def safe_name(name: str) -> str:
    name = str(name).strip()
    name = name.replace("/", "_").replace("\\", "_").replace(" ", "_")
    name = re.sub(r"[^A-Za-z0-9_.+-]+", "_", name)
    return name.strip("_") or "experiment"


def load_atlas_tables(paths: list[str | Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        frames.append(pd.read_csv(path, sep="\t", low_memory=False))
    if not frames:
        raise ValueError("At least one m6A-Atlas table is required.")
    df = pd.concat(frames, ignore_index=True)
    required = {"seqnames", "start", "strand", "Cell_Line", "Technique"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"m6A-Atlas table is missing columns: {sorted(missing)}")
    return df


def map_atlas_sites(atlas_df: pd.DataFrame, by_chrom: dict[str, list[Transcript]]) -> pd.DataFrame:
    rows = []
    for _, row in tqdm(atlas_df.iterrows(), total=len(atlas_df), desc="Mapping Atlas sites"):
        chrom = normalize_chrom(row["seqnames"])
        strand = str(row["strand"])
        genome_pos = int(row["start"]) - 1
        cell_lines = [x.strip() for x in str(row["Cell_Line"]).split(";") if x.strip()]
        technique = str(row["Technique"]).strip()
        candidates = []
        for tx in by_chrom.get(chrom, []):
            if tx.strand != strand:
                continue
            pos = genome_to_full_pos(genome_pos, tx)
            if pos < 2 or pos + 2 >= tx.length:
                continue
            window = tx.seq[pos - 2 : pos + 3]
            if tx.seq[pos] != "A" or not A_MOTIF.search(window):
                continue
            candidates.append((tx, pos))
        if not candidates:
            continue
        tx, pos = max(candidates, key=lambda item: item[0].length)
        for cell_line in cell_lines:
            experiment = f"{cell_line}_{technique}".strip("_")
            rows.append(
                {
                    "gene_id": tx.gene_id,
                    "transcript_id": tx.transcript_id,
                    "experiment": experiment,
                    "pos": int(pos),
                }
            )
    if not rows:
        raise ValueError("No m6A-Atlas sites could be mapped to transcript sequences.")
    return pd.DataFrame(rows).drop_duplicates()


def build_intermediate_data(
    atlas_tables: list[str | Path],
    gtf_path: str | Path,
    fasta_path: str | Path,
    output_data: str | Path,
    transcript_biotype: str | None,
    negative_ratio: int,
    seed: int,
) -> None:
    random.seed(seed)
    exons = read_gtf_exons(gtf_path, transcript_biotype=transcript_biotype)
    genome = read_fasta(fasta_path)
    transcripts = build_full_transcripts(exons, genome)
    tx_index = {(tx.gene_id, tx.transcript_id): tx for tx in transcripts}
    by_chrom = build_chrom_index(transcripts)

    atlas_df = load_atlas_tables(atlas_tables)
    mapped = map_atlas_sites(atlas_df, by_chrom)
    experiment_names = sorted(mapped["experiment"].dropna().astype(str).unique())

    rows = []
    for key, tx in tqdm(tx_index.items(), desc="Collecting transcript rows"):
        rows.append(
            {
                "id": tx.gene_id,
                "trans": tx.transcript_id,
                "chrom": tx.chrom,
                "strand": tx.strand,
                "start": tx.start,
                "end": tx.end,
                "length": tx.length,
                "seq": tx.seq,
            }
        )
    df = pd.DataFrame(rows).set_index(["id", "trans"], drop=False)

    grouped = mapped.groupby(["gene_id", "transcript_id", "experiment"])["pos"].agg(lambda x: set(map(int, x)))
    for experiment in experiment_names:
        df[experiment] = [set() for _ in range(len(df))]

    for (gene_id, transcript_id, experiment), positions in grouped.items():
        key = (gene_id, transcript_id)
        if key in df.index:
            df.at[key, experiment] = positions

    def union_positive(row: pd.Series) -> set[int]:
        out: set[int] = set()
        for experiment in experiment_names:
            out.update(row[experiment])
        return out

    df["m6a_pos"] = df.apply(union_positive, axis=1)
    df = df[df["m6a_pos"].map(len) > 0].copy()
    df["m6a_neg"] = df.apply(lambda row: motif_a_positions(row["seq"]) - set(row["m6a_pos"]), axis=1)

    for experiment in experiment_names:
        neg_col = f"{experiment}_neg"

        def sample_neg(row: pd.Series) -> set[int]:
            positives = list(row[experiment])
            candidates = list(row["m6a_neg"])
            k = len(positives) * negative_ratio
            if not positives or not candidates or k <= 0:
                return set()
            return set(random.choices(candidates, k=k))

        df[neg_col] = df.apply(sample_neg, axis=1)

    ndf = pd.DataFrame(
        [
            {
                "id": tx.gene_id,
                "trans": tx.transcript_id,
                "chrom": tx.chrom,
                "strand": tx.strand,
                "start": tx.start,
                "end": tx.end,
                "length": tx.length,
                "seq": tx.seq,
            }
            for tx in transcripts
        ]
    )
    output_data = Path(output_data)
    output_data.parent.mkdir(parents=True, exist_ok=True)
    with output_data.open("wb") as handle:
        pickle.dump([df.reset_index(drop=True), experiment_names, ndf], handle)
    print(f"Saved intermediate data: {output_data}")
    print(f"Experiments: {len(experiment_names)}")
    print(f"Transcripts with positives: {len(df)}")


def load_intermediate_data(path: str | Path):
    with Path(path).open("rb") as handle:
        obj = pickle.load(handle)
    if isinstance(obj, (list, tuple)) and len(obj) >= 2:
        df = obj[0]
        cls = list(obj[1])
        ndf = obj[2] if len(obj) > 2 else None
        return df, cls, ndf
    if isinstance(obj, pd.DataFrame):
        return obj, [], None
    raise ValueError(f"Unsupported .data object in {path}")


def infer_experiment_columns(df: pd.DataFrame, cls: list[str]) -> list[str]:
    ignore = {
        "id",
        "trans",
        "cds",
        "chrom",
        "strand",
        "start",
        "end",
        "range",
        "length",
        "splice",
        "min",
        "max",
        "seq",
        "m6a_pos",
        "m6a_neg",
        "_test_len",
        "_neg",
        "m6a_pos_set",
        "m6a_neg_set",
    }
    if cls:
        return [col for col in cls if col in df.columns and f"{col}_neg" in df.columns]
    return [
        col
        for col in df.columns
        if str(col) not in ignore
        and not str(col).endswith("_neg")
        and f"{col}_neg" in df.columns
    ]


def export_csv_files(
    data_path: str | Path,
    output_dir: str | Path,
    length: int,
    min_positive: int = 1,
    min_negative: int = 1,
) -> pd.DataFrame:
    df, cls, _ = load_intermediate_data(data_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    experiments = infer_experiment_columns(df, cls)
    if not experiments:
        raise ValueError(f"No experiment columns with matching *_neg columns were found in {data_path}")

    summary_rows = []
    for experiment in experiments:
        neg_col = f"{experiment}_neg"
        out_rows = []
        for _, row in df.iterrows():
            seq = str(row["seq"]).upper().replace("U", "T")
            positives = row[experiment] if row[experiment] == row[experiment] else set()
            negatives = row[neg_col] if row[neg_col] == row[neg_col] else set()
            for pos in sorted(set(positives)):
                out_rows.append({"seq": pad_and_extract(seq, int(pos), length), "label": 1})
            for pos in sorted(set(negatives)):
                out_rows.append({"seq": pad_and_extract(seq, int(pos), length), "label": 0})

        out_df = pd.DataFrame(out_rows, columns=["seq", "label"])
        pos_n = int((out_df["label"] == 1).sum()) if not out_df.empty else 0
        neg_n = int((out_df["label"] == 0).sum()) if not out_df.empty else 0
        if pos_n < min_positive or neg_n < min_negative:
            print(f"Skip {experiment}: positives={pos_n}, negatives={neg_n}")
            continue
        out_file = output_dir / f"{safe_name(experiment)}_merged.csv"
        out_df.to_csv(out_file, index=False)
        summary_rows.append(
            {
                "experiment": experiment,
                "file": str(out_file),
                "positive": pos_n,
                "negative": neg_n,
                "total": int(len(out_df)),
                "length": int(length),
            }
        )
        print(f"Saved {out_file}: positives={pos_n}, negatives={neg_n}")

    summary = pd.DataFrame(summary_rows)
    summary_file = output_dir / "summary.csv"
    summary.to_csv(summary_file, index=False)
    print(f"Saved summary: {summary_file}")
    return summary


def default_data_path(species: str) -> Path:
    return Path("data/intermediate/independent_atlas") / species / f"atlas_{species}_a_full.data"


def default_output_dir(species: str) -> Path:
    return Path("data/processed/independent_test_sets/atlas/full") / species


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["all", "build-data", "export"], default="all")
    parser.add_argument("--species", choices=["human", "mouse", "rat"], default="rat")
    parser.add_argument("--atlas-table", nargs="*", default=None, help="Raw m6A-Atlas species table(s).")
    parser.add_argument("--gtf", default=None, help="Species GTF file. Defaults depend on --species.")
    parser.add_argument("--fasta", default=None, help="Species genome FASTA file. Defaults depend on --species.")
    parser.add_argument("--data-path", default=None, help="Intermediate .data path.")
    parser.add_argument("--output-dir", default=None, help="Final CSV output directory.")
    parser.add_argument("--length", type=int, default=801)
    parser.add_argument("--negative-ratio", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--transcript-biotype", default="protein_coding")
    parser.add_argument("--min-positive", type=int, default=1)
    parser.add_argument("--min-negative", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data_path) if args.data_path else default_data_path(args.species)
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(args.species)

    if args.stage in {"all", "build-data"}:
        atlas_tables = args.atlas_table or DEFAULT_ATLAS_TABLES[args.species]
        gtf_path = args.gtf or DEFAULT_REFERENCE_FILES[args.species]["gtf"]
        fasta_path = args.fasta or DEFAULT_REFERENCE_FILES[args.species]["fasta"]
        missing = [str(p) for p in atlas_tables if not Path(p).exists()]
        if missing:
            raise FileNotFoundError(
                "Raw m6A-Atlas table(s) not found: "
                + ", ".join(missing)
                + ". Use --atlas-table to point to your files."
            )
        if not Path(gtf_path).exists():
            raise FileNotFoundError(f"GTF file not found: {gtf_path}. Use --gtf to point to your file.")
        if not Path(fasta_path).exists():
            raise FileNotFoundError(f"FASTA file not found: {fasta_path}. Use --fasta to point to your file.")
        build_intermediate_data(
            atlas_tables=atlas_tables,
            gtf_path=gtf_path,
            fasta_path=fasta_path,
            output_data=data_path,
            transcript_biotype=args.transcript_biotype,
            negative_ratio=args.negative_ratio,
            seed=args.seed,
        )

    if args.stage in {"all", "export"}:
        if not data_path.exists():
            raise FileNotFoundError(f"Intermediate .data file not found: {data_path}")
        export_csv_files(
            data_path=data_path,
            output_dir=output_dir,
            length=args.length,
            min_positive=args.min_positive,
            min_negative=args.min_negative,
        )


if __name__ == "__main__":
    main()
