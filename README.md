# turing-unlock-setup

一键启用 [nlzy/turing-unlock](https://github.com/nlzy/turing-unlock)，解锁 RTX 2080 Ti (TU102) 的 FP16 Tensor Core 厂商限速；附 30 秒生效验证脚本与训练吞吐前后对比工具。

> **English** — One-shot setup script for [nlzy/turing-unlock](https://github.com/nlzy/turing-unlock) (Turing port of Zagitta's gpu-unlock): unlocks the FP16 Tensor Core speed cap on RTX 2080 Ti (TU102). Includes a ~30s verification benchmark (`bench_fp16.py`) and a before/after training-throughput comparison tool (`compare_throughput.py`). **Use at your own risk** — running Tensor Cores beyond vendor limits can damage hardware.

## 仓库结构

```
turing-unlock-setup/
├── turing_unlock_setup.sh   # 一键安装脚本（唯一需要 sudo 的入口）
├── clean_up.sh              # 还原操作备忘清单（逐步复制粘贴执行）
├── bench_fp16.py            # ~30 秒验证解锁是否生效
├── bench_fp16_acc.py        # FP16 GEMM 两种累加模式（FP32acc vs FP16acc）对比
├── compare_throughput.py    # 对比解锁前后训练日志的 tok/s 稳态吞吐
├── bench_logs/              # 本机实测日志（已脱敏）：
│   ├── turing_baseline.log  #   未启用：FP32acc 32.5 / 41.5 / 44.4 TFLOPS
│   └── turing_unlock.log    #   启用后：FP32acc 74.7 TFLOPS（2048^3，约 +80%）
└── requirements.txt         # 仅 bench 脚本需要 torch
```

## 这是什么

[nlzy/turing-unlock](https://github.com/nlzy/turing-unlock)（Zagitta gpu-unlock 的 Turing 移植）一键安装脚本。

- **解锁内容**：RTX 2080 Ti（TU102）的 FP16 Tensor Core 厂商限速。内核模块在 initramfs 阶段、`nvidia` 驱动绑定显卡之前写入速度选择器（SS0=`0x88888888`、SS1=`0x8`）并释放 ACR mutex 3。
- **实测收益**（原仓库，cuBLAS FP16 matmul，M.N.K=2048）：55.7 → 91.4 TFLOPS（约 1.6 倍）。Transformer 训练的主力算子就是这个。
- **实测环境**：Dell RTX 2080 Ti 22GB（`10de:1e07`）+ NVIDIA 驱动 575.51.03，专有/开源驱动均可；其他卡与其他驱动版本未测试。

## ⚠️ 风险 / Risk

解锁后 Tensor Core 满血运行、功耗大幅升高，**可能损坏硬件**（原仓库原话：use at your own risk）。请确保供电与散热充足。

本仓库作者与贡献者不对任何硬件损坏、数据丢失或其他后果负责。

## 使用步骤

```sh
# 1. 一键安装（自动装依赖、克隆仓库、make install、写内核参数）
sudo bash turing_unlock_setup.sh

# 2. 重启
sudo reboot

# 3. 重启后确认解锁生效（有输出且无 error 即成功）
journalctl -k -b | grep turing_unlock
```

脚本做了什么（可读可改，约 150 行）：

1. root / 风险确认 / GPU 检测（`10de:1e07`）/ 记录驱动版本
2. apt 安装 git、make、gcc、build-essential、dpkg-dev（已有则跳过）
3. 克隆仓库到 `~/turing-unlock`（已存在则更新）
4. `make install` —— 自动识别发行版（Ubuntu 走 deb），构建并安装，模块+固件自动进 initramfs
5. 幂等写入内核参数 `turing_unlock.targets=all`（备份原文件为 `/etc/default/grub.bak.turing-unlock`）并 `update-grub`
6. 打印重启、验证、对比的后续指引

> 显存解锁（22GB 识别）属于 vBIOS/硬件层面，与本模块无关；本模块只解锁 Tensor Core 速度。

## 30 秒验证解锁是否生效

```sh
pip install -r requirements.txt   # bench 脚本需要 torch；compare_throughput.py 无依赖
python3 bench_fp16.py
```

判读（对照原仓库数据）：`1024^3`≈50 或 `4096^3`≈54 TFLOPS → **未生效**；`1024^3`≈68 / `2048^3`≈91 / `4096^3`≈82 → **生效**。

`bench_fp16_acc.py` 进一步对比 FP16 GEMM 的两种累加模式（FP32acc / FP16acc），可确认 cuBLAS 是否真的选择了 FP16 累加 kernel（Turing 上 FP16 累加吞吐翻倍），详见脚本头部注释。

注意：小模型训练吃不满这个峰值。turing-unlock 解锁的是 Tensor Core **计算峰值**，收益只在大矩阵（M.N.K≥1024）场景；17M 级模型、seq512/batch32 的训练实际只跑到 ~4 TFLOPS（峰值的 5~7%），瓶颈在显存带宽和 kernel 开销，端到端提升约 1.5%（已实测）。想看到明显收益：换更大档位（small/74M 级）或加大 batch（22GB 显存大量空闲），把矩阵撑大。

## 训练前后吞吐对比

| | 优化前 | 优化后 |
|---|---|---|
| 基准数据 | 已有训练产出的 `train_log.txt`（各档位目录，如 `ckpt/pico/`） | 重跑训练新生成 |
| 训练命令 | — | **必须与优化前完全相同**（同档位、同数据、同超参、同 seed） |

```sh
# 重启后重跑训练（示例，用你原来的命令）
python train.py --out_dir ckpt/pico_unlock   # 具体参数按你优化前那条命令原样保留

# 一键对比（目录对目录，自动匹配同名档位）
python3 compare_throughput.py ckpt ckpt_unlock
# 或单文件
python3 compare_throughput.py ckpt/pico/train_log.txt ckpt/pico_unlock/train_log.txt
```

日志格式：训练日志逐行含 `36,854 tok/s`、`峰值显存 3,251 MB`、`总用时 0.33 h` 之类字段即可被解析，完整格式样例见 `compare_throughput.py` 头部注释。

对比口径：每份日志丢弃前 10% 的采样（启动 warmup 波动），取其余 `tok/s` 的**中位数**为稳态吞吐，另给均值、单条峰值、峰值显存与总用时。提升幅度可对照原仓库的 1.3~1.6 倍区间（训练里数据加载、小矩阵层等有瓶颈，端到端增益通常低于纯 matmul 增益，属正常）。

`bench_logs/` 里留有一份本机实测前后日志，可直接拿来自测对比工具：

```sh
python3 compare_throughput.py bench_logs/turing_baseline.log bench_logs/turing_unlock.log
```

## 还原

```sh
bash clean_up.sh     # 还原操作备忘清单，逐步复制粘贴执行
```

要点：

```sh
sudo apt remove turing-unlock          # 包管理器卸载，initramfs 自动重建
# 恢复 /etc/default/grub：把 turing_unlock.targets=all 删掉（或直接用 .bak.turing-unlock 覆盖）
sudo update-grub && sudo reboot
```

## 致谢与许可

- [nlzy/turing-unlock](https://github.com/nlzy/turing-unlock) —— 核心内核模块与固件，GPL-2.0。本仓库只是它的安装包装与评测工具，不含其代码。
- [Zagitta/gpu-unlock](https://github.com/Zagitta/gpu-unlock) —— Volta/Ampere 原始方案。

本仓库脚本部分以 [MIT License](LICENSE) 发布。
