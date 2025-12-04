#!/bin/bash
# 完整流程执行脚本

set -e  # 遇到错误立即退出

echo "=========================================="
echo "Qwen3-4B AIGC检测模型微调 - 完整流程"
echo "=========================================="

# 检查Python环境
if ! command -v python &> /dev/null; then
    echo "错误: 未找到python命令"
    exit 1
fi

# 步骤1: 计算困惑度
echo ""
echo "步骤1: 计算困惑度..."
echo "----------------------------------------"
python scripts/step1_compute_ppls.py
if [ $? -ne 0 ]; then
    echo "错误: 困惑度计算失败"
    exit 1
fi

# 步骤2: 数据处理
echo ""
echo "步骤2: 数据处理（分块、mask、构建训练数据）..."
echo "----------------------------------------"
python scripts/step2_get_predict_json_v2.py
if [ $? -ne 0 ]; then
    echo "错误: 数据处理失败"
    exit 1
fi

# 步骤3: 准备LLama Factory数据集
echo ""
echo "步骤3: 准备LLama Factory数据集..."
echo "----------------------------------------"
python scripts/prepare_dataset.py
if [ $? -ne 0 ]; then
    echo "错误: 数据集准备失败"
    exit 1
fi

echo ""
echo "=========================================="
echo "数据准备完成!"
echo "=========================================="
echo ""
echo "下一步:"
echo "1. 将生成的数据文件复制到LLama Factory的data目录"
echo "2. 修改train_config.yaml中的模型路径"
echo "3. 运行训练: ./train.sh 或使用LLama Factory命令"
echo ""

