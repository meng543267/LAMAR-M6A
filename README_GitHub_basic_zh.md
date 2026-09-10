# LAMAR-m6A

LAMAR-m6A 是基于 RNA foundation language model LAMAR 微调得到的 m6A 位点预测模型。本仓库整理了从原始 GEO / m6A-Atlas 数据处理，到模型微调、评估、baseline 对比、可解释性分析和论文绘图的完整复现流程。

当前项目根目录：

```text
LAMAR-M6A/
```

## 项目结构

```text
LAMAR-M6A/
  configs/                 # 模型和数据处理配置
  data/                    # 原始数据、中间文件、处理后数据
  dataset/                 # HuggingFace tokenized datasets
  lamar_m6a/               # 主流程代码
  notebooks/               # Jupyter notebook 版本
  outputs/                 # 模型、预测、指标、可解释性和绘图输出
  pretrain_model/          # LAMAR 预训练权重
  scripts/                 # 模型结构和 patch 代码
  tokenizer/               # single-nucleotide tokenizer
```

主代码位于：

```text
lamar_m6a/data/
lamar_m6a/finetune/
lamar_m6a/evaluation/
lamar_m6a/interpretability/
```

## 环境

推荐使用 Python 3.10 或以上版本。

主要依赖：

```text
torch
transformers
datasets
safetensors
pandas
numpy
scikit-learn
matplotlib
seaborn
shap
logomaker
biopython
pyfaidx
tqdm
```

安装示例：

```bash
pip install -r requirements.txt
```

如果服务器 CUDA / PyTorch 版本需要单独指定，请先安装匹配 CUDA 的 PyTorch，再安装其它依赖。详细说明见：

```text
INSTALL_zh.md
```

所有命令建议在项目根目录下运行：

```bash
cd LAMAR-M6A
```

## 数据准备

原始数据放在：

```text
data/raw/
  geo/
  independent_atlas/
  ma6_atlas/
  reference/
```

注意：当前服务器目录名为 `data/raw/ma6_atlas/`，本文档保持该路径。

GEO 训练数据处理：

```bash
python -m lamar_m6a.data.prepare_geo_training_data
```

mature transcript 版本：

```bash
python -m lamar_m6a.data.prepare_geo_training_data_mature
```

主要输出：

```text
data/processed/full_splits/
data/processed/mature_splits/
data/processed/atlas/
```

其中 split 文件命名为：

```text
train_m6a_full_{cell}_{length}.csv
valid_m6a_full_{cell}_{length}.csv
test_m6a_full_{cell}_{length}.csv
```

## m6A-Atlas 独立测试集

构建 human / mouse / rat 独立测试集：

```bash
python -m lamar_m6a.data.build_independent_atlas_test_sets --stage all --species human
python -m lamar_m6a.data.build_independent_atlas_test_sets --stage all --species mouse
python -m lamar_m6a.data.build_independent_atlas_test_sets --stage all --species rat
```

当前输出路径：

```text
data/processed/independent_test_sets/atlas_a/human_full/
data/processed/independent_test_sets/atlas_a/human_mature/
data/processed/independent_test_sets/atlas_a/mouse-full/
data/processed/independent_test_sets/atlas_a/rat-full/
```

## 构建 Tokenized Dataset

```bash
python -m lamar_m6a.data.build_tokenized_dataset \
  --mode both \
  --tokenizer-path tokenizer/single_nucleotide
```

输出：

```text
dataset/full/m6a_full_{cell}_{length}/
dataset/mature/m6a_mature_{cell}_{length}/
```

常用细胞系命名：

```text
a549
cd8t
hek293
hela
molm
cell5
```

其中 `cell5` 表示五细胞系合并数据/模型，`molm` 表示 MOLM13。

## 模型微调

单细胞系模型：

```bash
python -m lamar_m6a.finetune.finetune_m6a \
  --mode full \
  --cell hek293 \
  --length 801 \
  --tokenizer-path tokenizer/single_nucleotide \
  --model-config configs/config_150M.json \
  --pretrain-state-path pretrain_model/model.safetensors
```

五细胞系合并模型：

```bash
python -m lamar_m6a.finetune.finetune_m6a \
  --mode full \
  --cell cell5 \
  --length 801 \
  --tokenizer-path tokenizer/single_nucleotide \
  --model-config configs/config_150M.json \
  --pretrain-state-path pretrain_model/model.safetensors
```

输出：

```text
outputs/finetune/full/{cell}-{length}-bs16-acc4/
```

## Cross-Cell Evaluation

```bash
python -m lamar_m6a.evaluation.evaluate_cross_cell \
  --mode full \
  --model-cells a549 cd8t hek293 hela molm cell5 \
  --length 801 \
  --split test \
  --checkpoint-template "outputs/finetune/{mode}/{model_cell}-{length}-bs{batch_size}-acc{accum_steps}"
```

输出：

```text
outputs/predictions/cross_cell/full/
outputs/metrics/cross_cell/full/
```

## Independent m6A-Atlas Evaluation

以 rat 为例：

```bash
python -m lamar_m6a.evaluation.evaluate_independent_atlas \
  --mode full \
  --length 801 \
  --input-folder data/processed/independent_test_sets/atlas_a/rat-full \
  --checkpoint-path outputs/finetune/full/cell5-801-bs16-acc4 \
  --run-name cell5_801_rat
```

输出：

```text
outputs/predictions/independent_atlas/full/
outputs/metrics/independent_atlas/full/
```

## 单个 CSV 预测

```bash
python -m lamar_m6a.evaluation.evaluate_single_csv \
  --input-csv data/processed/full_splits/test_m6a_full_cell5_801.csv \
  --output-csv outputs/predictions/single_csv/full/model_cell5_target_cell5_801_test_pred.csv \
  --checkpoint-path outputs/finetune/full/cell5-801-bs16-acc4 \
  --length 801 \
  --model-config configs/config_150M.json
```

## Baseline 对比

DeepSRAMP 和 DeepPromise 建议在独立 baseline model 环境中运行。LAMAR-M6A 仓库负责提供独立测试集和汇总 baseline metrics。

汇总 baseline 结果：

```bash
python -m lamar_m6a.evaluation.collect_baseline_metrics \
  --species rat \
  --result-file \
    outputs/baseline_raw/rat/atlas_rat_full_400_a549_single_a.csv \
    outputs/baseline_raw/rat/atlas_rat_full_400_a549_dp_a.csv \
  --model-name deepSRAMP DeepPromise \
  --output-csv outputs/metrics/baselines/rat_baseline_metrics.csv
```

## SHAP 分析

```bash
python -m lamar_m6a.interpretability.run_shap_analysis \
  --input-csv data/processed/full_splits/test_m6a_full_cell5_801.csv \
  --checkpoint-path outputs/finetune/full/cell5-801-bs16-acc4 \
  --label1-output outputs/interpretability/shap/full/cell5_801/label1_test_500.pkl \
  --label0-output outputs/interpretability/shap/full/cell5_801/label0_test_500.pkl \
  --max-sequences 500 \
  --length 801 \
  --model-config configs/config_150M.json
```

输出：

```text
outputs/interpretability/shap/full/cell5_801/label1_test_500.pkl
outputs/interpretability/shap/full/cell5_801/label0_test_500.pkl
```

## Mutation 分析

```bash
python -m lamar_m6a.interpretability.run_mutation_analysis \
  --input-csv outputs/predictions/single_csv/full/model_cell5_target_cell5_801_test_pred.csv \
  --checkpoint-path outputs/finetune/full/cell5-801-bs16-acc4 \
  --mutation-csv outputs/interpretability/mutation/cell5_mutated_sequences_801_label1_100.csv \
  --output-csv outputs/interpretability/mutation/cell5_mutated_sequences_801_label1_100_predict_linear.csv \
  --label 1 \
  --predicted-label 1 \
  --max-sequences 100 \
  --length 801 \
  --model-config configs/config_150M.json
```

最终用于绘图的 mutation 文件：

```text
outputs/interpretability/mutation/cell5_mutated_sequences_801_label1_100_predict_linear.csv
```

## 论文绘图

绘图 notebook：

```text
notebooks/plot/plot_paper_figures_notebook.ipynb
```

输出：

```text
outputs/plot/paperfig/
```

绘图主要读取：

```text
outputs/predictions/
outputs/metrics/
outputs/interpretability/
```

## Notebook 入口

```text
notebooks/data/
notebooks/finetune/
notebooks/evaluation/
notebooks/interpretability/
notebooks/plot/
```

## 推荐运行顺序

```text
01 GEO 训练数据构建
02 m6A-Atlas 独立测试集构建
03 tokenized dataset 构建
04 单细胞系模型微调
05 五细胞系 cell5 模型微调或复用
06 cross-cell evaluation
07 independent m6A-Atlas evaluation
08 baseline metric 汇总
09 single CSV prediction
10 SHAP analysis
11 mutation analysis
12 paper figure plotting
```