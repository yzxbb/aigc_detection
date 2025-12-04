#!/bin/bash
# Qwen3-4B微调训练脚本

# 设置LLama Factory路径（请修改为实际路径）
LLAMA_FACTORY_PATH="/path/to/LLaMA-Factory"

# 训练配置文件路径
CONFIG_FILE="./train_config.yaml"

# 检查LLama Factory是否存在
if [ ! -d "$LLAMA_FACTORY_PATH" ]; then
    echo "错误: LLama Factory路径不存在: $LLAMA_FACTORY_PATH"
    echo "请修改train.sh中的LLAMA_FACTORY_PATH为实际路径"
    exit 1
fi

# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误: 配置文件不存在: $CONFIG_FILE"
    exit 1
fi

# 进入LLama Factory目录
cd "$LLAMA_FACTORY_PATH"

# 运行训练
echo "开始训练..."
llamafactory-cli train "$CONFIG_FILE"

echo "训练完成!"

