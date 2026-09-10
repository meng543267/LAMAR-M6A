# 安装说明

推荐先创建独立环境：

```bash
conda create -n lamar-m6a python=3.10 -y
conda activate lamar-m6a
```

安装依赖：

```bash
pip install -r requirements.txt
```

如果服务器需要指定 CUDA 版本，建议先按服务器环境安装 PyTorch，再安装其它依赖。例如：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

如果 PyTorch 已经装好，也可以跳过 requirements 中的 torch，或使用：

```bash
pip install -r requirements.txt --no-deps
```

但 `--no-deps` 只适合你已经确认大部分依赖都安装完成的情况。

常用检查：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import transformers, datasets, safetensors, shap, logomaker; print('ok')"
```
