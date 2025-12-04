from pathlib import Path

# mask比例
train_mask_ratio = 0.1  # 训练时mask比例
mask_ratio = 0.04  # 推理时mask比例

# 重复的次数
random_state_add_seed_out = [0, 1, 2, 3, 4]  # 推理时使用的随机种子
train_repeat_time = 5  # 训练时重复次数
repeat_time = len(random_state_add_seed_out)

# 用于计算交叉熵损失（用于作为困惑度使用）
# 注意: 可以使用Qwen2.5-14B-Instruct或其他大模型来计算困惑度
model2compute_ppl = "/home/models/Qwen/Qwen2.5-14B-Instruct"  # 请修改为实际路径

# 训练集路径
# 源文件 - 使用原项目的数据
source_train_json_path = r"../CCKS2025-Large-Model-Text-Generation-Detection-Top3-method-main/whole_process_a/data/train.jsonl"
# 困惑度计算结果
train_ppl_path = r"./output/ppls_df_train.xlsx"
# 训练集输出json
train_output_json_path = r"./output/train_data.json"

# 测试集路径
# 源文件
test_set_json_path = r"../CCKS2025-Large-Model-Text-Generation-Detection-Top3-method-main/whole_process_a/data/test.jsonl"
# 困惑度计算结果
test_ppl_path = r"./output/ppls_df_test.xlsx"
# 测试集输出json
test_output_json_path = f"./output/test-mask{mask_ratio}-repeat{repeat_time}.json"

# 预测相关的路径
# 预测结果输出路径
model_output_dir = "./model_output"
# model路径 - 训练后的模型路径
model_name = "./output/checkpoint-3000"  # 请修改为实际训练后的模型路径
# 样本外数据
out_json_file_path = test_output_json_path
# 输出路径后缀
model_name_tag = Path(model_name).name if model_name else "qwen3-4b"
out_json_file_path_tag = Path(out_json_file_path).name
suffixes = f"-modelName-{model_name_tag}-mask{mask_ratio}-repeat{repeat_time}-{out_json_file_path_tag}"

