"""
使用Python代码下载Qwen3-4B模型
替代huggingface-cli命令的解决方案
"""
import os
import sys
from huggingface_hub import snapshot_download

def download_qwen3_4b(model_id="Qwen/Qwen3-4B-Instruct-2507", local_dir="/home/zym/models/Qwen/Qwen3-4B-Instruct-2507"):
    """
    下载Qwen3-4B模型
    
    Args:
        model_id: HuggingFace模型ID
        local_dir: 本地保存路径
    """
    print(f"开始下载模型: {model_id}")
    print(f"保存路径: {local_dir}")
    
    # 创建目录
    os.makedirs(os.path.dirname(local_dir), exist_ok=True)
    
    try:
        # 下载模型
        snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,  # 不使用符号链接，直接复制文件
            resume_download=True,  # 支持断点续传
        )
        print(f"\n模型下载完成！")
        print(f"模型路径: {local_dir}")
        return local_dir
    except Exception as e:
        print(f"下载失败: {e}")
        print("\n提示:")
        print("1. 检查网络连接")
        print("2. 如果使用镜像，可以设置环境变量:")
        print("   export HF_ENDPOINT=https://hf-mirror.com")
        print("3. 或者使用其他下载方式")
        return None

if __name__ == "__main__":
    import sys
    
    # 处理命令行参数
    if len(sys.argv) > 1:
        if sys.argv[1] in ["-h", "--help", "help"]:
            print("使用方法:")
            print("  python download_model.py [保存路径]")
            print("")
            print("示例:")
            print("  python download_model.py")
            print("  python download_model.py /home/zym/models/Qwen/Qwen3-4B-Instruct-2507")
            print("")
            print("环境变量:")
            print("  HF_ENDPOINT: 设置镜像地址，如 https://hf-mirror.com")
            sys.exit(0)
        local_dir = sys.argv[1]
    else:
        local_dir = "/home/zym/models/Qwen/Qwen3-4B-Instruct-2507"
    
    # 模型ID
    model_id = "Qwen/Qwen3-4B-Instruct-2507"
    
    # 如果设置了镜像环境变量，使用镜像
    if os.getenv("HF_ENDPOINT"):
        print(f"使用镜像: {os.getenv('HF_ENDPOINT')}")
    
    download_qwen3_4b(model_id, local_dir)

