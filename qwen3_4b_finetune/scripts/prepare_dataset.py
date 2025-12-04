"""
准备LLama Factory训练数据集
将生成的训练数据转换为LLama Factory需要的格式
"""
import json
import sys
import os

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import train_output_json_path

def convert_to_llamafactory_format(input_file, output_file):
    """
    将训练数据转换为LLama Factory格式
    LLama Factory需要的格式: instruction, input, output
    """
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # LLama Factory格式
    llamafactory_data = []
    for item in data:
        llamafactory_item = {
            "instruction": item["instruction"],
            "input": item["input"],
            "output": item.get("output", "")
        }
        llamafactory_data.append(llamafactory_item)
    
    # 保存为LLama Factory格式
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(llamafactory_data, f, ensure_ascii=False, indent=2)
    
    print(f"已转换 {len(llamafactory_data)} 条数据")
    print(f"输出文件: {output_file}")
    print("\n请将输出文件复制到LLama Factory的data目录下")
    print("例如: cp {} /path/to/LLaMA-Factory/data/train_sft_llm_detect_mask_aug_qwen3_4b.json".format(output_file))

if __name__ == "__main__":
    # 输出文件路径
    output_file = train_output_json_path.replace(".json", "_llamafactory.json")
    
    if not os.path.exists(train_output_json_path):
        print(f"错误: 训练数据文件不存在: {train_output_json_path}")
        print("请先运行 step2_get_predict_json_v2.py 生成训练数据")
        sys.exit(1)
    
    convert_to_llamafactory_format(train_output_json_path, output_file)

