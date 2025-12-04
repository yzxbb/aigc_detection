from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd
from tqdm import tqdm
import numpy as np
from config import (model2compute_ppl, source_train_json_path, test_set_json_path,
                   train_ppl_path, test_ppl_path)
tqdm.pandas()

# ===========================================================
# 加载模型
# ===========================================================
print("加载模型")
tokenizer = AutoTokenizer.from_pretrained(model2compute_ppl)
model = AutoModelForCausalLM.from_pretrained(
    model2compute_ppl,
    device_map="npu",
    trust_remote_code=True,
    torch_dtype=torch.float16)
# model.to("npu:0")
model = model.eval()  # 切换到评估模式
print("加载模型 done")


# ===========================================================
# 计算困惑度的函数
# ===========================================================
def get_ppl(input_text):
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512  # 根据模型最大序列长度调整
    )
    input_ids = inputs.input_ids.to("npu")  # 移动到GPU加速

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            labels=input_ids  # 自回归任务中，标签与输入相同
        )
        loss = outputs.loss  # 获取损失值
    return loss.item()

def compute_ppl(filepath, output_filepath):
    df_data = pd.read_json(filepath, lines=True)
    df_data["id"] = np.arange(df_data.shape[0])
    df_data.rename({"text": "txt"}, axis=1, inplace=True)
    ppls = df_data["txt"].progress_apply(lambda x: get_ppl(x))
    df_data["ppls"] = ppls
    df_data.to_excel(output_filepath, index=False)

def compute_ppl_train(filepath, output_filepath):
    df_data = pd.read_json(filepath, lines=True)
    np.random.seed(1)

    train_index = df_data.sample(frac=0.98).index
    valid_index = df_data.loc[~df_data.index.isin(train_index), :].sample(frac=0.5).index
    test_index = df_data.loc[~(df_data.index.isin(train_index) | df_data.index.isin(valid_index)), :].index

    df_train = df_data.loc[train_index, :]
    df_val = df_data.loc[valid_index, :]
    df_test = df_data.loc[test_index, :]
    df_data = pd.concat([df_train, df_val, df_test])

    df_data["id"] = np.arange(df_data.shape[0])
    df_data.rename({"text": "txt"}, axis=1, inplace=True)
    ppls = df_data["txt"].progress_apply(lambda x: get_ppl(x))
    df_data["ppls"] = ppls
    df_data.to_excel(output_filepath, index=False)

# ===========================================================
# 计算训练集的ppls
# ===========================================================
if source_train_json_path is not None:
    compute_ppl_train(source_train_json_path, train_ppl_path)
compute_ppl(test_set_json_path, test_ppl_path)