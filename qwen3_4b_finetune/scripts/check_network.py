"""
网络连接检查脚本
用于诊断网络认证问题
"""
import socket
import subprocess
import sys
import os

def check_port(host, port, timeout=3):
    """检查端口是否可达"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception as e:
        print(f"检查端口时出错: {e}")
        return False

def check_ping(host):
    """检查ping是否可达"""
    try:
        result = subprocess.run(
            ['ping', '-c', '3', host],
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.returncode == 0
    except Exception as e:
        print(f"Ping检查时出错: {e}")
        return False

def check_bitsrun_config():
    """检查bitsrun配置"""
    config_paths = [
        os.path.expanduser("~/.config/bitsrun/config.json"),
        os.path.expanduser("~/.bitsrun/config.json"),
        os.path.expanduser("~/.config/bitsrun/config.yaml"),
    ]
    
    found_configs = []
    for path in config_paths:
        if os.path.exists(path):
            found_configs.append(path)
            print(f"✓ 找到配置文件: {path}")
            try:
                with open(path, 'r') as f:
                    content = f.read()
                    print(f"  内容预览: {content[:200]}...")
            except Exception as e:
                print(f"  读取失败: {e}")
    
    if not found_configs:
        print("✗ 未找到bitsrun配置文件")
        print("  可能的配置路径:")
        for path in config_paths:
            print(f"    - {path}")
    
    return found_configs

def main():
    print("=" * 60)
    print("网络连接诊断工具")
    print("=" * 60)
    
    # 检查bitsrun配置
    print("\n[1] 检查bitsrun配置")
    print("-" * 60)
    check_bitsrun_config()
    
    # 检查常见认证服务器
    print("\n[2] 检查常见认证服务器")
    print("-" * 60)
    test_hosts = [
        ("10.0.0.55", 80),
        ("10.0.0.55", 443),
        ("auth.bupt.edu.cn", 80),
        ("auth.bupt.edu.cn", 443),
    ]
    
    for host, port in test_hosts:
        print(f"\n检查 {host}:{port}")
        ping_ok = check_ping(host)
        port_ok = check_port(host, port)
        
        print(f"  Ping: {'✓ 可达' if ping_ok else '✗ 不可达'}")
        print(f"  端口: {'✓ 开放' if port_ok else '✗ 关闭/超时'}")
    
    # 检查网络路由
    print("\n[3] 检查网络路由")
    print("-" * 60)
    try:
        result = subprocess.run(
            ['ip', 'route', 'get', '8.8.8.8'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            print("默认路由:")
            print(result.stdout)
        else:
            print("无法获取路由信息")
    except Exception as e:
        print(f"检查路由时出错: {e}")
    
    # 建议
    print("\n" + "=" * 60)
    print("建议")
    print("=" * 60)
    print("1. 如果所有服务器都不可达，可能是网络未连接或需要认证")
    print("2. 尝试在浏览器中访问校园网认证页面")
    print("3. 如果bitsrun持续失败，可以使用网页认证")
    print("4. 联系网络管理员确认网络状态")
    print("\n注意: 网络问题不会影响本地模型训练和推理")

if __name__ == "__main__":
    main()

