#!/usr/bin/env python3
# ============================================================
# bench_fp16_acc.py — FP16 GEMM 两种累加模式对比（约 1 分钟）
#
# 用法：python3 bench_fp16_acc.py
#
# 两种模式（同一份输入数据，只切一个开关）：
#   FP16+FP32acc  allow_fp16_reduced_precision_reduction = False
#                 （强制 FP16 GEMM 用 FP32 累加，即原 bench_fp16.py 的模式）
#   FP16+FP16acc  allow_fp16_reduced_precision_reduction = True
#                 （允许 cuBLAS 用 FP16 做分块/分次累加；是否真用取决于 kernel 选择，
#                  看每档“误差”列：FP16acc 误差明显大于 FP32acc 才说明真走了 FP16 累加）
#
# 判读（RTX 2080 Ti @250W，turing-unlock 生效时）：
#   FP32acc ≈ 68 / 91 / 82 TFLOPS（对照原脚本）；FP16acc 档若明显更高
#   （Turing Tensor Core 的 FP16 累加吞吐是 FP32 累加的 2 倍）说明降精度累加被选中；
#   两档几乎相同且误差相同 → cuBLAS 没走 FP16 累加 kernel。
#   注：FP16 累加档只有 Volta/Turing 的 Tensor Core 才有；Ampere 及之后累加固定 FP32，
#   在那些卡上两档数字会基本一样，本对比主要用于 20 系卡。
# ============================================================
import time

import torch

SIZES = (1024, 2048, 4096)
WARMUP = 10
ITERS = 50


def bench_mm(a, b):
    """计时 torch.mm：返回 (每次耗时 ms, 最后一次结果)"""
    for _ in range(WARMUP):
        torch.mm(a, b)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        c = torch.mm(a, b)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return dt / ITERS * 1e3, c


assert torch.cuda.is_available(), '未检测到 CUDA'
print(f'GPU: {torch.cuda.get_device_name(0)}')
torch.manual_seed(0)
for n in SIZES:
    a = torch.randn(n, n, device='cuda', dtype=torch.float16)
    b = torch.randn(n, n, device='cuda', dtype=torch.float16)

    # —— FP16 + FP32 累加 ——
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    ms32, c32 = bench_mm(a, b)
    t32 = 2 * n**3 / (ms32 / 1e3) / 1e12

    # —— FP16 + FP16 累加（允许降精度累加）——
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    ms16, c16 = bench_mm(a, b)
    t16 = 2 * n**3 / (ms16 / 1e3) / 1e12

    # —— 精度对比：以 FP32 计算（FP16 输入精确上转 + FP32 累加）为基准 ——
    ref = a.float() @ b.float()
    scale = ref.abs().max().item()
    err32 = (c32.float() - ref).abs().max().item()
    err16 = (c16.float() - ref).abs().max().item()

    print(f'{n:>5}^3 | FP16+FP32acc: {t32:6.1f} TFLOPS ({ms32:7.3f} ms/次) | 最大误差 {err32:.3e} (相对 {err32 / scale:.2e})')
    print(f'{n:>5}^3 | FP16+FP16acc: {t16:6.1f} TFLOPS ({ms16:7.3f} ms/次) | 最大误差 {err16:.3e} (相对 {err16 / scale:.2e}) | 提速 {t16 / t32:4.2f}x')

print('判读：')
print('  FP32acc ≈68/91/82 TFLOPS → turing-unlock 生效；≈50/56/54 → 未生效')
print('  FP16acc 档明显更快且误差明显更大 → FP16 累加 kernel 被选中；两档几乎一样 → 没走 FP16 累加')
