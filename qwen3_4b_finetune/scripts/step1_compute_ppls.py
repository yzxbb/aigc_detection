"""
计算困惑度(交叉熵均值)
使用Qwen2.5-14B-Instruct或其他大模型来计算文本的困惑度
"""
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd
from tqdm import tqdm
import numpy as np
import sys
import os

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (model2compute_ppl, source_train_json_path, test_set_json_path,
                   train_ppl_path, test_ppl_path)

tqdm.pandas()

# ===========================================================
# 加载模型
# ===========================================================
print("加载模型用于计算困惑度...")
try:
    tokenizer = AutoTokenizer.from_pretrained(model2compute_ppl, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model2compute_ppl,
        device_map="auto",  # 自动选择设备
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    model = model.eval()  # 切换到评估模式
    print("模型加载完成")
except Exception as e:
    print(f"模型加载失败: {e}")
    print("请检查model2compute_ppl路径是否正确")
    sys.exit(1)


# ===========================================================
# 计算困惑度的函数
# ===========================================================
def get_ppl(input_text):
    """
    计算文本的困惑度(实际为交叉熵均值)
    只取前512个字符进行计算
    """
    # 只取前512个字符
    input_text = input_text[:512]
    
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )
    
    # 自动选择设备
    device = next(model.parameters()).device
    input_ids = inputs.input_ids.to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            labels=input_ids  # 自回归任务中，标签与输入相同
        )
        loss = outputs.loss  # 获取损失值
    return loss.item()


def compute_ppl(filepath, output_filepath):
    """计算测试集的困惑度"""
    print(f"正在处理: {filepath}")
    df_data = pd.read_json(filepath, lines=True)
    df_data["id"] = np.arange(df_data.shape[0])
    df_data.rename({"text": "txt"}, axis=1, inplace=True)
    print(f"共 {len(df_data)} 条数据")
    
    ppls = df_data["txt"].progress_apply(lambda x: get_ppl(x))
    df_data["ppls"] = ppls
    df_data.to_excel(output_filepath, index=False)
    print(f"结果已保存到: {output_filepath}")


def compute_ppl_train(filepath, output_filepath):
    """计算训练集的困惑度"""
    print(f"正在处理训练集: {filepath}")
    df_data = pd.read_json(filepath, lines=True)
    np.random.seed(1)

    # 按照原项目的逻辑进行数据划分
    train_index = df_data.sample(frac=0.98).index
    valid_index = df_data.loc[~df_data.index.isin(train_index), :].sample(frac=0.5).index
    test_index = df_data.loc[~(df_data.index.isin(train_index) | df_data.index.isin(valid_index)), :].index

    df_train = df_data.loc[train_index, :]
    df_val = df_data.loc[valid_index, :]
    df_test = df_data.loc[test_index, :]
    df_data = pd.concat([df_train, df_val, df_test])

    df_data["id"] = np.arange(df_data.shape[0])
    df_data.rename({"text": "txt"}, axis=1, inplace=True)
    print(f"共 {len(df_data)} 条数据")
    
    ppls = df_data["txt"].progress_apply(lambda x: get_ppl(x))
    df_data["ppls"] = ppls
    df_data.to_excel(output_filepath, index=False)
    print(f"结果已保存到: {output_filepath}")


# ===========================================================
# 计算训练集和测试集的ppls
# ===========================================================
if __name__ == "__main__":
    # 创建输出目录
    os.makedirs(os.path.dirname(train_ppl_path), exist_ok=True)
    os.makedirs(os.path.dirname(test_ppl_path), exist_ok=True)
    
    # 计算训练集的困惑度
    if source_train_json_path and os.path.exists(source_train_json_path):
        compute_ppl_train(source_train_json_path, train_ppl_path)
    else:
        print(f"警告: 训练集文件不存在: {source_train_json_path}")
    
    # 计算测试集的困惑度
    if test_set_json_path and os.path.exists(test_set_json_path):
        compute_ppl(test_set_json_path, test_ppl_path)
    else:
        print(f"警告: 测试集文件不存在: {test_set_json_path}")

