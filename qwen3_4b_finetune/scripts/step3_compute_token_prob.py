"""
计算token概率，进行预测
"""
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
import numpy as np
import sys
import os
import re
import json

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import model_output_dir, model_name, suffixes, out_json_file_path

tqdm.pandas()

# ===========================================================
# 加载模型
# ===========================================================
print("加载模型...")
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    print("模型加载完成")
except Exception as e:
    print(f"模型加载失败: {e}")
    print("请检查model_name路径是否正确")
    sys.exit(1)

# 找到 `1` 和 `0` 的 token ID
token_1_id = tokenizer.convert_tokens_to_ids("1")
token_0_id = tokenizer.convert_tokens_to_ids("0")

if token_1_id == tokenizer.unk_token_id or token_0_id == tokenizer.unk_token_id:
    # 如果直接转换失败，尝试其他方法
    token_1_id = tokenizer.encode("1", add_special_tokens=False)[0]
    token_0_id = tokenizer.encode("0", add_special_tokens=False)[0]

print(f"Token ID - '1': {token_1_id}, '0': {token_0_id}")


def get_prob(instruction, input_txt):
    """
    计算模型输出0和1的概率
    """
    messages = [
        {"role": "user", "content": instruction + input_txt},
        {"role": "assistant", "content": "<label>"}
    ]
    
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        # 提取到<label>之前的内容
        match = re.search(r"(.*<label>)", text, re.S)
        if match:
            text = match.group(1)
        else:
            # 如果匹配失败，直接使用
            text = instruction + input_txt + "<label>"
    except Exception as e:
        # 如果apply_chat_template失败，手动构建
        text = instruction + input_txt + "<label>"
    
    model_inputs_with_prefix = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**model_inputs_with_prefix, return_dict=True)
        last_token_logits = outputs.logits[0, -1, :]  # Shape (vocab_size,)
        prob_for_1, prob_for_0 = F.softmax(last_token_logits[[token_1_id, token_0_id]], dim=-1)
        
    return prob_for_0.item()


# ===========================================================
# 读取数据并计算概率
# ===========================================================
print(f"读取数据: {out_json_file_path}")
with open(out_json_file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

df_data = pd.DataFrame(data)
print(f"共 {len(df_data)} 条数据")

# 计算概率
print("计算概率...")
df_data["prob_0"] = df_data.progress_apply(
    lambda row: get_prob(row["instruction"], row["input"]), axis=1
)

# 计算每个id的均值和中位数
print("聚合概率...")
df_grouped = df_data.groupby("id")["prob_0"].agg(["mean", "median"]).reset_index()
df_grouped["mean_median"] = (df_grouped["mean"] + df_grouped["median"]) / 2

# 保存结果
os.makedirs(model_output_dir, exist_ok=True)

# 保存详细概率
output_file = os.path.join(model_output_dir, f"probs_df_data_out{suffixes}.xlsx")
df_data.to_excel(output_file, index=False)
print(f"详细概率已保存到: {output_file}")

# 保存聚合结果
output_file_grouped = os.path.join(model_output_dir, f"probs_df_data_out{suffixes}_ret.xlsx")
df_grouped.to_excel(output_file_grouped, index=False)
print(f"聚合结果已保存到: {output_file_grouped}")

# 保存最终预测结果（使用阈值0.998688126）
threshold = 0.998688126
df_grouped["pred"] = (df_grouped["mean_median"] > threshold).astype(int)
output_file_final = os.path.join(model_output_dir, f"mean_median{suffixes}-{threshold}.txt")
df_grouped[["id", "pred"]].to_csv(output_file_final, sep="\t", index=False, header=False)
print(f"最终预测结果已保存到: {output_file_final}")

print("\n预测完成!")

