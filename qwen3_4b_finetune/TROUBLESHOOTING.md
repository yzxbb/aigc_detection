# 网络连接问题排查指南

## 问题1：bitsrun连接超时

### 症状
```
httpx.ConnectTimeout: timed out
httpcore.ConnectTimeout: timed out
```

### 诊断
- bitsrun是北京理工大学校园网认证工具
- 在尝试连接认证服务器时超时
- 可能是网络配置或服务器不可达

### 解决方案

#### 方案1: 检查bitsrun配置
```bash
# 查看配置路径
bitsrun config-paths

# 查看配置内容
cat ~/.config/bitsrun/config.json
# 或
cat ~/.bitsrun/config.json
```

#### 方案2: 检查网络连接
```bash
# 检查是否能访问认证服务器
# 通常bitsrun会尝试连接校园网认证网关
ping 10.0.0.55  # 或其他认证服务器地址
```

#### 方案3: 使用网页认证
如果bitsrun无法连接，可以尝试：
1. 在浏览器中访问校园网认证页面
2. 完成网页认证后，网络即可使用
3. 然后再尝试使用bitsrun

#### 方案4: 检查是否需要VPN
某些网络环境需要先连接VPN才能访问认证服务器。

#### 方案5: 增加超时时间（如果支持）
检查bitsrun是否有超时配置选项：
```bash
bitsrun login --help
```

#### 方案6: 使用其他认证方式
如果bitsrun持续失败，可以：
- 使用网页认证
- 使用其他校园网认证工具
- 联系网络管理员

---

## 问题2：srun登录超时

### 症状
```
ERRO[2025-12-04 21:04:41] dial tcp 10.0.0.55:80: i/o timeout
```

### 诊断结果
- ✅ 网络层正常（ping可通）
- ❌ 应用层连接超时（80/443端口无法连接）

## 解决方案

### 方案1: 检查防火墙规则

```bash
# 检查iptables规则
sudo iptables -L -n | grep 10.0.0.55

# 检查是否有防火墙阻止
sudo ufw status  # Ubuntu
sudo firewall-cmd --list-all  # CentOS/RHEL
```

### 方案2: 检查服务是否运行

```bash
# 检查目标服务是否在运行
telnet 10.0.0.55 80
# 或
nc -zv 10.0.0.55 80
```

### 方案3: 使用代理或VPN

如果是企业/校园网络，可能需要：
1. 先连接VPN
2. 配置代理设置
3. 通过认证网关

### 方案4: 检查srun配置

```bash
# 查看srun配置
cat ~/.srun/config
# 或
cat ~/.config/srun/config

# 检查srun可执行文件
which srun
file $(which srun)
```

### 方案5: 增加超时时间

如果srun支持配置超时，可以尝试增加超时时间：

```bash
# 查看srun帮助
srun --help
# 或
srun login --help
```

### 方案6: 使用其他端口

某些服务可能使用非标准端口：

```bash
# 扫描常用端口
nmap -p 80,443,8080,8443 10.0.0.55
```

### 方案7: 检查DNS和路由

```bash
# 检查路由
ip route get 10.0.0.55

# 检查ARP表
arp -a | grep 10.0.0.55
```

## 临时解决方案

如果这是网络认证问题，可以尝试：

1. **使用浏览器访问**
   ```bash
   # 尝试在浏览器中访问
   http://10.0.0.55
   # 或
   https://10.0.0.55
   ```

2. **检查是否需要先认证**
   - 某些网络需要先通过网页认证
   - 完成认证后再使用srun

3. **联系网络管理员**
   - 确认10.0.0.55服务的状态
   - 确认是否需要特殊配置

## 与AIGC检测项目的关系

如果这个网络问题不影响模型下载和训练，可以：

1. **使用本地模型**：如果模型已下载，直接使用本地路径
2. **使用镜像源**：下载模型时使用镜像（如hf-mirror.com）
3. **离线训练**：所有训练都可以在本地完成，不需要网络连接

## 继续项目工作

即使网络有问题，也可以继续AIGC检测项目：

```bash
# 1. 如果模型已下载，直接配置路径
# 编辑 config.py 和 train_config.yaml

# 2. 使用本地数据进行训练
cd /home/zym/aigc_detection/qwen3_4b_finetune
./run_all.sh  # 数据处理（使用本地数据，不需要网络）

# 3. 开始训练（完全本地，不需要网络）
./train.sh
```

网络问题不会影响本地训练和推理流程。

