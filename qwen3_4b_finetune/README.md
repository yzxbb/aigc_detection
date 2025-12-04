# Qwen3-4B AIGC检测模型微调

本项目基于CCKS2025大模型文本生成检测Top3方案，使用Qwen3-4B模型进行微调，实现AIGC内容检测。

## 项目结构

```
qwen3_4b_finetune/
├── config.py                    # 配置文件
├── train_config.yaml            # LLama Factory训练配置
├── train.sh                     # 训练脚本
├── README.md                    # 本文件
├── data/                        # 数据目录（链接到原项目数据）
├── output/                      # 输出目录（困惑度、训练数据、模型checkpoint）
├── model_output/                # 模型预测输出
└── scripts/                     # 脚本目录
    ├── step1_compute_ppls.py           # 计算困惑度
    ├── step2_get_predict_json_v2.py    # 数据处理（分块、mask、构建训练数据）
    ├── step3_compute_token_prob.py     # 推理预测
    └── prepare_dataset.py              # 准备LLama Factory数据集
```

## 环境准备

### 1. 安装依赖

```bash
# 安装基础依赖
pip install transformers torch pandas numpy tqdm openpyxl

# 安装LLama Factory（用于模型训练）
pip install llamafactory
# 或者从源码安装
# git clone https://github.com/hiyouga/LLaMA-Factory.git
# cd LLaMA-Factory
# pip install -e .
```

### 2. 下载模型

- **Qwen3-4B-Instruct**: 用于微调的基座模型
  - 可以从ModelScope或HuggingFace下载
  - 路径示例: `/path/to/Qwen3-4B-Instruct`

- **Qwen2.5-14B-Instruct** (可选): 用于计算困惑度
  - 如果使用其他模型计算困惑度，可以修改`config.py`中的`model2compute_ppl`

## 使用步骤

### 步骤1: 配置参数

编辑 `config.py`，修改以下路径：

```python
# 计算困惑度的模型路径
model2compute_ppl = "/path/to/Qwen2.5-14B-Instruct"

# 数据路径（已指向原项目数据）
source_train_json_path = r"../CCKS2025-Large-Model-Text-Generation-Detection-Top3-method-main/whole_process_a/data/train.jsonl"
test_set_json_path = r"../CCKS2025-Large-Model-Text-Generation-Detection-Top3-method-main/whole_process_a/data/test.jsonl"
```

### 步骤2: 计算困惑度

```bash
cd /home/zym/aigc_detection/qwen3_4b_finetune
python scripts/step1_compute_ppls.py
```

这将生成：
- `output/ppls_df_train.xlsx` - 训练集困惑度
- `output/ppls_df_test.xlsx` - 测试集困惑度

### 步骤3: 数据处理

```bash
python scripts/step2_get_predict_json_v2.py
```

这将生成：
- `output/train_data.json` - 训练数据（包含文本分块、mask增强等）
- `output/test-mask0.04-repeat5.json` - 测试数据

### 步骤4: 准备LLama Factory数据集

```bash
python scripts/prepare_dataset.py
```

然后将生成的数据文件复制到LLama Factory的data目录：

```bash
# 假设LLama Factory路径为 /path/to/LLaMA-Factory
cp output/train_data_llamafactory.json /path/to/LLaMA-Factory/data/train_sft_llm_detect_mask_aug_qwen3_4b.json
```

### 步骤5: 配置训练参数

编辑 `train_config.yaml`：

```yaml
# 修改模型路径
model_name_or_path: /path/to/Qwen3-4B-Instruct

# 根据GPU显存调整batch size
per_device_train_batch_size: 4
gradient_accumulation_steps: 4

# 输出路径
output_dir: ./output/qwen3-4b-lora
```

### 步骤6: 开始训练

**方法1: 使用训练脚本**

```bash
# 先修改train.sh中的LLAMA_FACTORY_PATH
chmod +x train.sh
./train.sh
```

**方法2: 直接使用LLama Factory命令**

```bash
cd /path/to/LLaMA-Factory
llamafactory-cli train /home/zym/aigc_detection/qwen3_4b_finetune/train_config.yaml
```

训练过程中会保存checkpoint，建议使用checkpoint-3000（约训练2轮）。

### 步骤7: 模型权重合并（可选）

如果需要将LoRA权重合并到基座模型：

```bash
cd /path/to/LLaMA-Factory
llamafactory-cli export \
    --model_name_or_path /path/to/Qwen3-4B-Instruct \
    --adapter_name_or_path ./output/qwen3-4b-lora/checkpoint-3000 \
    --export_dir ./output/qwen3-4b-merged \
    --export_size 2 \
    --export_device cpu
```

### 步骤8: 推理预测

1. 修改 `config.py` 中的模型路径：

```python
model_name = "./output/qwen3-4b-lora/checkpoint-3000"  # 或合并后的模型路径
```

2. 运行推理：

```bash
python scripts/step3_compute_token_prob.py
```

这将生成：
- `model_output/probs_df_data_out-*.xlsx` - 详细概率结果
- `model_output/mean_median-*-0.998688126.txt` - 最终预测结果（使用阈值0.998688126）

## 核心方法说明

### 1. 困惑度计算
- 使用Qwen2.5-14B-Instruct计算文本前512个字符的交叉熵均值作为困惑度
- 根据训练集的困惑度分布，将文本分为7个等级：非常小、较小、小、中、大、较大、非常大

### 2. 文本处理
- **文本分块**: 按照分隔符（句号、换行符、冒号、空格）将长文本分割为不超过1500字符的块
- **Mask增强**: 
  - 训练时：15%概率mask单词，每个样本生成5个不同mask版本
  - 推理时：4%概率mask单词，使用多个随机种子生成多个版本

### 3. 训练数据格式
每个训练样本包含：
- `instruction`: 系统提示词（定义角色和任务）
- `input`: 用户输入（包含困惑度等级和文本内容）
- `output`: 标签（`<label>0</label>`或`<label>1</label>`）

### 4. 预测方法
- 对同一文本的多个分块和mask版本分别预测
- 计算所有预测概率的均值和中位数
- 使用mean和median的平均值作为最终概率
- 阈值0.998688126判断是否为AI生成

## 参数调整建议

### 针对Qwen3-4B的调整

1. **Batch Size**: 4B模型显存占用较小，可以适当增大batch size
2. **Learning Rate**: 可以尝试5e-5到1e-4
3. **LoRA参数**: 
   - rank: 256 (可以尝试128或512)
   - alpha: 256 (通常为rank的1-2倍)
4. **训练步数**: 建议训练到checkpoint-3000左右（约2轮）

## 注意事项

1. **数据路径**: 确保数据路径正确，本项目默认使用原项目的数据
2. **模型路径**: 所有模型路径都需要修改为实际路径
3. **设备**: 代码会自动检测GPU/CPU，但建议使用GPU训练
4. **显存**: 4B模型在单卡24GB GPU上可以正常训练，batch_size=4时约需12-16GB显存

## 参考

- 原项目: CCKS2025-Large-Model-Text-Generation-Detection-Top3-method-main
- LLama Factory: https://github.com/hiyouga/LLaMA-Factory
- Qwen模型: https://github.com/QwenLM/Qwen

## 许可证

本项目基于原CCKS2025项目，遵循相同的许可证。

