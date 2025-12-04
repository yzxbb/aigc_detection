from pathlib import Path

# mask比例
train_mask_ratio = 0.1
# mask_ratio = 0.04
mask_ratio = 0.04  # todo


# 重复的次数
random_state_add_seed_out = [168, 158, 1230, 820, 1225, 10086, 10010, 10000]
train_repeat_time = 5
repeat_time = len(random_state_add_seed_out)  # todo

# 用于计算交叉熵损失（用于作为困惑度使用）
model2compute_ppl = "/home/models/Qwen/Qwen2.5-14B-Instruct"

# 训练集路径
# 源文件
source_train_json_path = r"./data/train.jsonl"
# 困惑度计算结果
train_ppl_path = r"./output/ppls_df_train.xlsx"
# 训练集输出json
train_output_json_path = r"./output/train_data.json"

# 测试集路径
# 源文件
test_set_json_path = r"./data/test0717.jsonl"
# 困惑度计算结果
test_ppl_path = r"./output/ppls_df_test0717.xlsx"
# 测试集输出json
test_output_json_path = f"./output/test0717-mask{mask_ratio}-reapt{repeat_time}.json"


# 预测相关的路径
# 预测结果输出路径
model_output_dir = "./model_output"
# model路径
model_name = "/home/jovyan/merge-model/AAA-BEST04-Qwen2.5-14B-lora0622-v2.7-cp3000"
# 样本外数据
out_json_file_path = test_output_json_path
# 输出路径后缀
model_name_tag = Path(model_name).name
out_json_file_path_tag = Path(out_json_file_path).name
suffixes = f"-modelName-{model_name_tag}-mask{mask_ratio}-reapt{repeat_time}-{out_json_file_path_tag}"
# 参考最佳结果
best_ret_file_path = "./B榜提交结果/mean_median-modelName-AAA-BEST04-Qwen2.5-14B-lora0622-v2.7-cp3000-mask0.04-reapt8-test0717-mask0.04-reapt8-0720.json-0.998688126.txt"
# best_ret_file_path = None
