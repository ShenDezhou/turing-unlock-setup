#!/usr/bin/env python3
# ============================================================
# bench_fp16.py — 验证 turing-unlock 是否生效（约 30 秒）
#
# 用法：python3 bench_fp16.py
#
# 判读（对照 turing-unlock README，RTX 2080 Ti @250W，FP16+FP32acc）：
#   1024^3:  ~50 TFLOPS = 未生效    ~68 TFLOPS = 生效
#   2048^3:  ~56 TFLOPS = 未生效    ~91 TFLOPS = 生效
#   4096^3:  ~54 TFLOPS = 未生效    ~82 TFLOPS = 生效
# ============================================================
import time

import torch
# 强制 FP16 GEMM 使用 FP32 进行累加
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
assert torch.cuda.is_available(), '未检测到 CUDA'
print(f'GPU: {torch.cuda.get_device_name(0)}')
for n in (1024, 2048, 4096):
    a = torch.randn(n, n, device='cuda', dtype=torch.float16)
    b = torch.randn(n, n, device='cuda', dtype=torch.float16)
    for _ in range(10):          # warmup
        torch.mm(a, b)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    iters = 50
    for _ in range(iters):
        torch.mm(a, b)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    tflops = 2 * n**3 * iters / dt / 1e12
    print(f'{n:>5}^3 FP16 matmul: {tflops:6.1f} TFLOPS')
print('判读：1024^3≈50 / 2048^3≈56 / 4096^3≈54 → 未生效；≈68 / ≈91 / ≈82 → 解锁生效')
