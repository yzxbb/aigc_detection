# 快速开始指南

## 前置条件

1. **Python环境**: Python 3.8+
2. **GPU**: 建议使用NVIDIA GPU（至少12GB显存）
3. **模型文件**: 
   - Qwen3-4B-Instruct（用于微调）
   - Qwen2.5-14B-Instruct（用于计算困惑度，可选）

## 安装步骤

### 1. 安装依赖

```bash
cd /home/zym/aigc_detection/qwen3_4b_finetune
pip install -r requirements.txt
```

### 2. 安装LLama Factory

**方法1: pip安装**
```bash
pip install llamafactory
```

**方法2: 从源码安装（推荐）**
```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e .
```

## 配置

### 1. 修改config.py

```python
# 计算困惑度的模型（可以使用Qwen2.5-14B或其他大模型）
model2compute_ppl = "/path/to/Qwen2.5-14B-Instruct"

# 数据路径（已指向原项目，通常不需要修改）
source_train_json_path = r"../CCKS2025-Large-Model-Text-Generation-Detection-Top3-method-main/whole_process_a/data/train.jsonl"
test_set_json_path = r"../CCKS2025-Large-Model-Text-Generation-Detection-Top3-method-main/whole_process_a/data/test.jsonl"
```

### 2. 修改train_config.yaml

```yaml
# 修改为你的Qwen3-4B模型路径
model_name_or_path: /path/to/Qwen3-4B-Instruct

# 根据GPU显存调整
per_device_train_batch_size: 4  # 显存不足可以减小到2或1
gradient_accumulation_steps: 4  # 相应增大以保持有效batch size
```

## 执行流程

### 方式1: 使用自动化脚本（推荐）

```bash
# 执行完整的数据准备流程
./run_all.sh
```

### 方式2: 手动执行

#### 步骤1: 计算困惑度
```bash
python scripts/step1_compute_ppls.py
```

#### 步骤2: 数据处理
```bash
python scripts/step2_get_predict_json_v2.py
```

#### 步骤3: 准备训练数据
```bash
python scripts/prepare_dataset.py
```

#### 步骤4: 复制数据到LLama Factory
```bash
# 假设LLama Factory安装在 /path/to/LLaMA-Factory
cp output/train_data_llamafactory.json /path/to/LLaMA-Factory/data/train_sft_llm_detect_mask_aug_qwen3_4b.json
```

#### 步骤5: 开始训练
```bash
# 修改train.sh中的LLAMA_FACTORY_PATH后执行
./train.sh

# 或直接使用LLama Factory命令
cd /path/to/LLaMA-Factory
llamafactory-cli train /home/zym/aigc_detection/qwen3_4b_finetune/train_config.yaml
```

#### 步骤6: 推理预测
```bash
# 修改config.py中的model_name为训练后的模型路径
# 例如: model_name = "./output/qwen3-4b-lora/checkpoint-3000"

python scripts/step3_compute_token_prob.py
```

## 常见问题

### Q1: 显存不足怎么办？
- 减小`per_device_train_batch_size`（如改为2或1）
- 增大`gradient_accumulation_steps`以保持有效batch size
- 使用`deepspeed`进行分布式训练（在train_config.yaml中配置）

### Q2: 训练需要多长时间？
- 取决于数据量和GPU性能
- 通常checkpoint-3000需要几小时到十几小时
- 可以在训练过程中查看loss曲线判断是否收敛

### Q3: 如何选择checkpoint？
- 原项目使用checkpoint-3000（约训练2轮）
- 可以查看训练日志中的loss，选择loss最低的checkpoint
- 也可以在验证集上测试不同checkpoint的效果

### Q4: 如何调整超参数？
主要可调整的参数：
- `learning_rate`: 学习率（默认5e-5）
- `lora_rank`: LoRA秩（默认256，可尝试128或512）
- `lora_alpha`: LoRA alpha（默认256，通常为rank的1-2倍）
- `warmup_ratio`: 预热比例（默认0.1）

### Q5: 数据路径错误？
- 确保原项目数据存在
- 检查`config.py`中的路径是否正确
- 可以使用绝对路径

## 下一步

训练完成后，可以：
1. 在测试集上评估模型性能
2. 调整超参数进行进一步优化
3. 尝试不同的数据增强策略
4. 集成到实际应用中

## 获取帮助

如遇问题，请检查：
1. 所有路径是否正确
2. 依赖是否完整安装
3. GPU驱动和CUDA是否正确安装
4. 模型文件是否完整下载

