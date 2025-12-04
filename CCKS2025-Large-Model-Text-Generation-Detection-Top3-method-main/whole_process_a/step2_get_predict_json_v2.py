# %% [markdown]
# # 1 定义相关函数

# %%
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import List, Dict
import json
import gc
import random
from config import (source_train_json_path, test_set_json_path,
                   train_ppl_path, test_ppl_path, mask_ratio, train_mask_ratio,
                   train_output_json_path, test_output_json_path, repeat_time,
                   train_repeat_time, random_state_add_seed_out)

# %% [markdown]
# ## 1.1 文本分块相关的函数

# %%
def split_txt_equall_length(txts: List[str],
                            max_len: int,
                            interval_row=0,
                            sep="") -> List[str]:
    """用于将文本按照指定长度进行切分，但是不会讲句子从中间切开，不会将完整的句子切开
    """
    ret_txts = []
    # 重新组织数据，将多行数据进行合并，提高判断效率，每段文本最多`max_len`个字符
    new_txt_i = ""
    i = 0
    last_i = 0
    while i < len(txts):
        new_txt_j = txts[i]
        if len(new_txt_j) == 0:
            i += 1
            continue

        if len(new_txt_i) + len(new_txt_j) <= max_len:
            if len(new_txt_i) == 0 or new_txt_i[-1] == sep:
                new_txt_i += new_txt_j
            else:
                new_txt_i += sep + new_txt_j
        else:
            if len(new_txt_i) > 0:
                ret_txts.append(new_txt_i)
                if last_i < i - interval_row:
                    i = max(i - interval_row, 0)
                    new_txt_j = txts[i]
                last_i = i
            new_txt_i = new_txt_j
        i += 1

    if len(new_txt_i) > 0:
        ret_txts.append(new_txt_i)
    return ret_txts

def txt2block(
    input_txt: str, max_len: int, sep=[".", "。"], interval_row=5
) -> List[str]:
    """
    @author: bohanyang
    @将原本固定分隔符的方法修改为了可自定义分隔符

    对文本进行分块，max_len表示每个分块最大的长度。
    对于文本进行如下操作：
    以句号、换行符为分隔符进行分割。

    Args:
        input_txt (str): 输入的文本。
        sep (List[str]): 分隔符列表。
        max_len (int): 每个分块的最大长度。

    Returns:
        List[str]: 分块后的文本列表。
    """
    if len(input_txt) < max_len:
        return [input_txt]
    for separator in sep:
        input_txt = input_txt.replace(separator, separator + "@@<sep>@@")
    ret = [i for i in input_txt.split(r"@@<sep>@@") if len(i) > 0]
    ret = split_txt_equall_length(
        ret,
        max_len,
        interval_row=interval_row,
        sep="",
    )
    return ret

def txt2block_df(temp: pd.Series, max_len: int=512, sep=[".", "。", ","], interval_row=2, sep_col="txt"):
    ret = []
    split_txts = txt2block(temp[sep_col], max_len, sep=sep, interval_row=interval_row)
    for txt_i in split_txts:
        temp = temp.copy()
        temp[sep_col] = txt_i
        ret.append(temp)
    return pd.DataFrame(ret)

def split_txt_equall_length_df(df, max_len: int=512, sep=[".", "。", ","],
                               interval_row=2, sep_col="txt"):
    df_split = []
    for _, row_i in tqdm(df.iterrows()):
        df_split.append(txt2block_df(row_i,
                                     max_len=max_len,
                                     sep=sep,
                                     interval_row=interval_row,
                                     sep_col=sep_col))
    return pd.concat(df_split)

# %% [markdown]
# # 2 读取语料

def get_train_df(filepath):
    df_data = pd.read_json(filepath, lines=True)
    np.random.seed(1)

    train_index = df_data.sample(frac=0.98).index
    valid_index = df_data.loc[~df_data.index.isin(train_index), :].sample(frac=0.5).index
    test_index = df_data.loc[~(df_data.index.isin(train_index) | df_data.index.isin(valid_index)), :].index

    df_train = df_data.loc[train_index, :]
    df_val = df_data.loc[valid_index, :]
    df_test = df_data.loc[test_index, :]
    df_data = pd.concat([df_train, df_val, df_test])
    return df_data

# %%
# 训练家
df_train_raw = get_train_df(source_train_json_path)
# df_train_raw = pd.read_json(source_train_json_path, lines=True)
df_train_raw["id"] = np.arange(df_train_raw.shape[0])
df_train_raw.rename({"text": "txt"}, axis=1, inplace=True)
df_train_raw.drop_duplicates(["txt", "label"], inplace=True)
# 测试集
df_out_raw = pd.read_json(test_set_json_path, lines=True)
df_out_raw["id"] = np.arange(df_out_raw.shape[0])
df_out_raw.rename({"text": "txt"}, axis=1, inplace=True)
# ppls计算结果
data_out_ppl = pd.read_excel(test_ppl_path)
data_train_ppl = pd.read_excel(train_ppl_path)
data_train_ppl.drop_duplicates(["txt", "label"], inplace=True)

# 加入ppls
df_train_raw = pd.merge(df_train_raw, data_train_ppl.loc[:, ["txt", "ppls"]], on="txt", how="left", validate="1:1")
df_out_raw = pd.merge(df_out_raw, data_out_ppl.loc[:, ["txt", "ppls"]], on="txt", how="left", validate="1:1")

# %%
# 原始数据集
data_info_raw = {
    "训练集": df_train_raw,
    "样本外": df_out_raw,
}

# 过滤后的数据集
data_info = {}

# %%
# 提示词模板
instruction =\
"""
- Role: 人工智能文本生成检测专家
- Profile: 你是一位在人工智能文本分析领域具有深厚专业知识的专家，对大模型生成文本和人类撰写文本的特征有着敏锐的洞察力，能够快速识别两者之间的差异。对大模型生成文本和人类撰写文本的特征差异有着深刻的洞察力，能够快速识别并准确判断文本的来源。
用户会提供一段文本中的内容(可能是部分或者全文)，以及整段文本对应的困惑度等级（从小到大等级为：非常小，较小，小，中，大，较大，非常大），
注意，内容中“[MASK]”表示被遮盖的单词，不用考虑“[MASK]”的语义信息。
- OutputFormat:大模型生成的内容，则输出:<label>1</label>,如果输入是人类撰写，则输出:<label>0</label>。
- Workflow:
  1. 接收输入文本,忽略[MASK]的语义信息。
  2. 分析文本的语言风格、逻辑连贯性、用词习惯等特征,判断是否为大模型生成。
  3. 根据分析结果，判断文本来源并输出相应的标签。
- Tips:
  1. 分析过程中考虑AI常用的高频词。
  2. 可以结合句子长短，用词风格，语言风格等进行考虑。
  3. 注意语言的风格，AI生成的语言通常语法上会更完美，而人类的语音有时候会有一些表达不合理。
  4. 可以考虑其他这里未提及的重要语义特征。
- user_input:
"""

input_prompt = """
全文困惑度等级为: {}
内容(可能是部分或者全文)：
{}
"""

def output_json(df: pd.DataFrame,
                output_json_path: str,
                instruction_txt: str=instruction,
                input_col: str="txt",
                output_col: str="label"):
    outputs = []
    with open(output_json_path, "w", encoding="utf-8") as f:
        for _, row in tqdm(df.iterrows()):
            row_i = {"instruction": instruction_txt, "input": input_prompt.format(row["ppl_bin"], row[input_col]),
                     "id": row["id"]}
            if output_col in row:
                row_i["output"] = "<label>" + str(row[output_col]) + "</label>"
            outputs.append(row_i)
        json.dump(outputs, f, ensure_ascii=False, indent=4)


# %% [markdown]
# ## 2.1 进行文本分块

# %%
def split_txt_chunck(data_info_raw: Dict[str, pd.DataFrame],
                     max_len=1500,
                     sep=[[".", "\n",], [":"], [" "]],
                     interval_row=3,
                     sep_col="txt") -> Dict[str, pd.DataFrame]:
    """_summary_

    Args:
        data_info_raw (Dict[pd.DataFrame]): 需要处理的数据
        max_len (int, optional): 每个分块最大的长度. Defaults to 1500.
        sep (list, optional): 每个分块分隔的符号.
            Defaults to [[".", "\n",], [":"], [" "]].
        interval_row (int, optional): overlap的分块数量（每个分块的分隔符为sep）.
            Defaults to 3.
        sep_col (str, optional): 待处理的字段. Defaults to "txt".

    Returns:
        Dict[pd.DataFrame]: 分块后的数据
    """
    # 过滤后的数据集
    data_info = {}
    for data_name, df_i_raw in data_info_raw.items():
        print("="*80)
        print(f"{data_name} 处理")
        df_j = df_i_raw.copy()
        assert df_j["id"].is_unique, "数据id"
        for sep_i in sep:
            # 文本分块
            df_j = split_txt_equall_length_df(
                df_j,
                max_len=max_len,
                sep=sep_i,
                interval_row=interval_row,
                sep_col=sep_col
            )
            print(f"文本切分后,按照 {sep_i} ")
            print(f"{data_name}数量：", df_j.shape[0])
        # 去重
        df_j.drop_duplicates(["txt"], inplace=True)
        # 文本块长度
        df_j["txt_length"] = df_j["txt"].apply(lambda x: len(x))
        # 过滤长度
        df_j = df_j.query("txt_length > 190").copy()
        print(f"过滤长度, {data_name}数量：", df_j.shape[0])
        if data_name != "训练集":
            # 补充缺失的原始文本
            df_j = pd.concat([df_j, df_i_raw.loc[~df_i_raw["id"].isin(df_j["id"].unique()), :]])
            print(f"补充数据后")
        # 文本块长度
        df_j["txt_length"] = df_j["txt"].apply(lambda x: len(x))
        print(f"{data_name}数量：", df_j["txt_length"].describe())
        data_info[data_name] = df_j
        print("="*80)
    return data_info

# %%
data_info = split_txt_chunck(data_info_raw, max_len=1500)

# %%
assert len(df_out_raw["id"].unique()) == len(data_info["样本外"]["id"].unique()), "样本外数据集变短"

# %%


# %% [markdown]
# ## 2.2 随机进行词语遮盖

# %%


def mask_random_words(text, alpha, random_state: int=1):
    """
    Randomly masks words in a given text with a 'MASK' token based on a specified probability.

    Parameters:
    - text (str): The input text to process.
    - alpha (float): The probability of masking a word.

    Returns:
    - str: The processed text with some words masked.
    """
    random.seed(random_state)
    words = text.split()
    masked_words = []
    # 控制不要连续多个mask
    skip_time = 0

    for word in words:
        if random.random() < alpha and skip_time == 0:
            masked_words.append("[MASK]")
            skip_time = 5
        else:
            if skip_time > 0:
                skip_time -= 1
            masked_words.append(word)

    return " ".join(masked_words)

# %%
test_random_state_list = random_state_add_seed_out
train_random_state_list = [i for i in list(range(train_repeat_time))]

for data_name, df_i in data_info.items():
    if data_name == "训练集":
        mask = train_mask_ratio
        random_state_list = train_random_state_list
    else:
        mask = mask_ratio
        random_state_list = test_random_state_list
    
    df_all = []
    for i in random_state_list:
        df_j = df_i.copy()
        df_j["txt"] = df_j["txt"].apply(lambda x: mask_random_words(x, mask, random_state=i))
        df_all.append(df_j)
    df_all = pd.concat(df_all)
    data_info[data_name] = df_all


# 定义分位数边界
quantiles = [0, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 1]
# 计算分位数值
train_bins = data_info_raw["训练集"]["ppls"].quantile(quantiles).tolist()

def quantile_binning(series):
    # 定义标签
    labels = ['非常小', '较小', '小', '中', '大', '较大', '非常大']
    # 使用pd.cut进行分箱
    binned_series = pd.cut(series, bins=train_bins, labels=labels, include_lowest=True)
    return binned_series

# %%
for data_nam, df_i in data_info.items():
    df_i["ppl_bin"] = quantile_binning(df_i["ppls"])

# %%

def balance_label(df, label_col='label'):
    # 计算每个类别的样本数量
    class_counts = df[label_col].value_counts().sort_index()

    # 找出最小的样本数量
    min_count = class_counts.min()

    # 对每个类别进行采样
    balanced_data = []
    for class_label, count in class_counts.items():
        class_data = df[df[label_col] == class_label]
        if count > min_count:
            class_data = class_data.sample(min_count, random_state=42)
        balanced_data.append(class_data)

    # 合并所有采样后的数据
    balanced_df = pd.concat(balanced_data)

    # 重新设置索引
    balanced_df = balanced_df.reset_index(drop=True)
    balanced_df = balanced_df.sample(frac=1)

    return balanced_df


# %%
# 进行训练集的样本平衡
data_info["训练集"] = balance_label(data_info["训练集"], label_col="label")


# %% [markdown]
# ## 3 输出语料

# %%
output_json(data_info["训练集"],
            output_json_path=train_output_json_path,
            instruction_txt=instruction,
            input_col = "txt",
            output_col = "label")

output_json(data_info["样本外"],
            output_json_path=test_output_json_path,
            instruction_txt=instruction,
            input_col = "txt",
            output_col = "label")



