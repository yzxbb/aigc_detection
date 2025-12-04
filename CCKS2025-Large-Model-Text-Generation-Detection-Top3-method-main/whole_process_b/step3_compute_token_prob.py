from modelscope import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
import numpy as np
from config import model_output_dir, model_name, suffixes, out_json_file_path, best_ret_file_path
import re
tqdm.pandas()

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="npu"
)
# Ensure the model is in evaluation mode
model.eval()

# 找到 `1` 和 `0` 的 token ID
token_1_id = tokenizer.convert_tokens_to_ids("1")
token_0_id = tokenizer.convert_tokens_to_ids("0")

def get_prob(instruction, input_txt):
    messages = [
        {"role": "user", "content": instruction+input_txt},
        {"role": "assistant", "content": "<label>"}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        # enable_thinking=False
    )
    text = re.search("(.*<label>)<|im_end|>", text, re.S).group(1)
    model_inputs_with_prefix = tokenizer([text], return_tensors="pt").to(model.device)
    

    with torch.no_grad():
        outputs = model(**model_inputs_with_prefix, return_dict=True)
        last_token_logits = outputs.logits[0, -1, :] # Shape (vocab_size,)
        prob_for_1, prob_for_0 = F.softmax(last_token_logits[[token_1_id, token_0_id]], dim=-1)
        
    return prob_for_0.item()

df_data = pd.read_json(out_json_file_path)
prob_zeros = df_data.progress_apply(lambda x: get_prob(x["instruction"], x["input"]), axis=1)
df_data["pred_zero"] = prob_zeros
df_data["pred_one"] = 1 - prob_zeros
df_data.to_excel(f"{model_output_dir}/probs_df_data_out{suffixes}.xlsx", index=False)
print(df_data["pred_one"].describe())

import pandas as pd
import numpy as np
from pathlib import Path
df =pd.read_excel(f"{model_output_dir}/probs_df_data_out{suffixes}.xlsx")

df_ret = df.groupby('id')['pred_one'].agg(["mean", "median"])
df_ret["mean_median"] = (df_ret["mean"]+df_ret["median"])/2
df_ret["mean_median"] = (df_ret["mean_median"] >= 0.998688126)*1  # todo

# AAA-BEST04 mask0.04 0.998688126


df_ret["mean_median"].to_csv(f"{model_output_dir}/mean_median{suffixes}-0.998688126.txt", index=False, header=False)  # todo
df_ret.to_excel(f"{model_output_dir}/probs_df_data_out{suffixes}_ret-0.998688126.xlsx", index=True)  # todo

if best_ret_file_path is not None:
    df_reference = pd.read_csv(best_ret_file_path, header=None)
    df_ret["reference"] = df_reference[0].values
    print("和最优答案不一致的比例", 1-np.mean((df_ret["reference"] == df_ret["mean_median"])*1))
    df_ret.to_excel(f"{model_output_dir}/probs_df_data_out{suffixes}_ret-0.998688126.xlsx", index=True)  # todo