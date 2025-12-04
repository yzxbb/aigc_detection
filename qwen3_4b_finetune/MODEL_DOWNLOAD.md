# 模型下载指南

## 问题说明

如果遇到 `huggingface-cli` 命令不可用的问题（如 `ModuleNotFoundError: No module named 'huggingface_hub.commands'`），可以使用Python脚本下载模型。

## 方法1: 使用Python脚本下载（推荐）

### 下载Qwen3-4B模型

```bash
cd /home/zym/aigc_detection/qwen3_4b_finetune
python scripts/download_model.py
```

### 指定自定义路径

```bash
python scripts/download_model.py /path/to/your/model/dir
```

### 使用镜像加速（国内推荐）

```bash
# 设置镜像环境变量
export HF_ENDPOINT=https://hf-mirror.com

# 然后运行下载脚本
python scripts/download_model.py
```

## 方法2: 在Python代码中直接下载

```python
from huggingface_hub import snapshot_download

# 下载模型
snapshot_download(
    repo_id="Qwen/Qwen3-4B-Instruct-2507",
    local_dir="/home/zym/models/Qwen/Qwen3-4B-Instruct-2507",
    local_dir_use_symlinks=False,
    resume_download=True,
)
```

## 方法3: 使用git-lfs（如果已安装）

```bash
# 安装git-lfs（如果未安装）
# sudo apt-get install git-lfs  # Ubuntu/Debian
# brew install git-lfs  # macOS

git lfs install
git clone https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507 /home/zym/models/Qwen/Qwen3-4B-Instruct-2507
```

## 方法4: 修复huggingface-cli（可选）

如果希望使用 `huggingface-cli` 命令，可以尝试：

```bash
# 方法A: 在conda环境中安装
conda install -c conda-forge huggingface_hub

# 方法B: 使用pip强制重新安装
pip uninstall huggingface_hub -y
pip install huggingface_hub --upgrade

# 方法C: 检查conda和pip的路径冲突
which huggingface-cli
python -c "import huggingface_hub; print(huggingface_hub.__file__)"
```

## 下载其他模型

### 下载Qwen2.5-14B-Instruct（用于计算困惑度）

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen2.5-14B-Instruct",
    local_dir="/home/zym/models/Qwen/Qwen2.5-14B-Instruct",
    local_dir_use_symlinks=False,
    resume_download=True,
)
```

或者修改 `scripts/download_model.py` 中的 `model_id` 参数。

## 验证下载

下载完成后，检查模型文件：

```bash
ls -lh /home/zym/models/Qwen/Qwen3-4B-Instruct-2507/
```

应该看到以下文件：
- `config.json` - 模型配置
- `tokenizer.json` - 分词器
- `model-*.safetensors` 或 `pytorch_model.bin` - 模型权重
- 其他相关文件

## 常见问题

### Q1: 下载速度慢？
- 使用镜像：`export HF_ENDPOINT=https://hf-mirror.com`
- 检查网络连接
- 使用代理（如果适用）

### Q2: 下载中断？
- `snapshot_download` 支持断点续传，重新运行即可
- 检查磁盘空间是否充足

### Q3: 权限错误？
- 确保有写入权限：`chmod -R 755 /home/zym/models/`
- 或使用其他有权限的目录

### Q4: 磁盘空间不足？
- 检查可用空间：`df -h`
- Qwen3-4B模型约需要8-10GB空间
- Qwen2.5-14B模型约需要28-30GB空间

## 配置模型路径

下载完成后，更新配置文件：

1. **config.py** - 计算困惑度的模型路径：
```python
model2compute_ppl = "/home/zym/models/Qwen/Qwen2.5-14B-Instruct"
```

2. **train_config.yaml** - 微调的模型路径：
```yaml
model_name_or_path: /home/zym/models/Qwen/Qwen3-4B-Instruct-2507
```

