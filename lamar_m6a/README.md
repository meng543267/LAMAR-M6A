# LAMAR-m6A Data Pipeline

This folder contains GitHub-friendly one-command pipelines for LAMAR-m6A training data.

There are two sequence modes:

```text
full    : transcript genomic span style workflow
mature  : exon-spliced mature transcript workflow
```

## Required Input Files

Put raw files under `data/raw/` before running.

```text
data/raw/reference/Homo_sapiens.GRCh37.87.gtf.gz
data/raw/reference/Homo_sapiens.GRCh37.dna.primary_assembly.fa.gz

data/raw/m6a_atlas/hg38_CL_Tech.txt

data/raw/geo/GSE71154/GSM1828596_precise_m6A_sites_A549_hg19.bed
data/raw/geo/GSE71154/GSM1828594_precise_m6A_sites_CD8T_hg19.bed
data/raw/geo/GSE63753/GSE63753_hek293.abcam.CIMS.m6A.9536.bed
data/raw/geo/GSE63753/GSE63753_hek293.sysy.CITS.m6A.12051.bed
data/raw/geo/GSE86336/GSM2300426_precise_m6A_sites_HeLa_CA_hg19.bed.gz
data/raw/geo/GSE86336/GSM2300427_precise_m6A_sites_HeLa_NU_hg19.bed.gz
data/raw/geo/GSE86336/GSM2300428_precise_m6A_sites_HeLa_CY_hg19.bed.gz
data/raw/geo/GSE86336/GSM2300429_precise_m6A_sites_HeLa_CYpolyA_hg19.bed.gz
data/raw/geo/GSE98623/GSM2602070_molm13_rep1.tag.uniq.sites.bed.gz
data/raw/geo/GSE98623/GSM2602071_molm13_rep2.tag.uniq.sites.bed.gz
data/raw/geo/GSE98623/GSM2602072_molm13_rep3.tag.uniq.sites.bed.gz
```

## Run Full-Transcript Version

```bash
python prepare_geo_training_data.py
```

## Run Mature-Transcript Version

```bash
python prepare_geo_training_data_mature.py
```

The mature version splices exon sequences first, maps BED genome positions to mature transcript coordinates, and then cuts sequence windows from mature transcript sequences.

## Output Structure

The processed directory follows this structure:

```text
data/processed/
  atlas/
  full_balanced/
  full_lengths/
  full_negative_pools/
  full_sequences/
  full_splits/
  mature_balanced/
  mature_lengths/
  mature_negative_pools/
  mature_sequences/
  mature_splits/
```

Index files are stored outside `processed/`:

```text
data/intermediate/full_index/
data/intermediate/mature_index/
```

Optional tokenized HuggingFace datasets are stored in:

```text
data/processed/full_tokenized/
data/processed/mature_tokenized/
```

## Main Output Files

Full version:

```text
data/processed/full_sequences/m6a_positive_{cell}.csv
data/processed/full_sequences/m6a_negative_{cell}.csv
data/processed/full_negative_pools/m6a_{cell}_sample_100negative.csv
data/processed/full_negative_pools/m6a_full_{cell}_sample100_allneg_filtered.csv
data/processed/full_balanced/m6a_full_{cell}_sample_pos_10neg.csv
data/processed/full_lengths/m6a_full_{cell}_{length}.csv
data/processed/full_splits/train_m6a_full_{cell}_{length}.csv
data/processed/full_splits/valid_m6a_full_{cell}_{length}.csv
data/processed/full_splits/test_m6a_full_{cell}_{length}.csv
```

Mature version:

```text
data/processed/mature_sequences/m6a_positive_{cell}.csv
data/processed/mature_sequences/m6a_negative_{cell}.csv
data/processed/mature_negative_pools/m6a_{cell}_sample_100negative.csv
data/processed/mature_negative_pools/m6a_full_{cell}_sample100_allneg_filtered.csv
data/processed/mature_balanced/m6a_full_{cell}_sample_pos_10neg.csv
data/processed/mature_lengths/m6a_full_{cell}_{length}.csv
data/processed/mature_splits/train_m6a_full_{cell}_{length}.csv
data/processed/mature_splits/valid_m6a_full_{cell}_{length}.csv
data/processed/mature_splits/test_m6a_full_{cell}_{length}.csv
```

Atlas files shared by both versions:

```text
data/processed/atlas/cell4_seq.csv
data/processed/atlas/cell5_seq.csv
data/processed/atlas/GSE63753_seq.csv
data/processed/atlas/GSE71154_seq.csv
data/processed/atlas/GSE86336_seq.csv
data/processed/atlas/GSE98623_seq.csv
data/processed/atlas/m6a_atlas_seq_data.csv
```

## Notebook Versions

```text
prepare_geo_training_data_notebook.ipynb
prepare_geo_training_data_mature_notebook.ipynb
```
