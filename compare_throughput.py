#!/usr/bin/env python3
# ============================================================
# compare_throughput.py — 对比 turing-unlock 启用前后的训练吞吐量
#
# 数据来源：train.py 生成的 ckpt/<档位>/train_log.txt
#   逐行形如：iter  2450/2451 | loss 5.2605 (ema 5.2008) | lr 6.00e-05 | 36,854 tok/s | 峰值显存 3,251 MB
#
# 用法：
#   目录对目录（自动匹配同名档位，推荐）：
#     python3 compare_throughput.py <优化前ckpt目录> <优化后ckpt目录>
#     例：python3 compare_throughput.py ckpt_before ckpt_after
#   单文件对单文件：
#     python3 compare_throughput.py a/train_log.txt b/train_log.txt
#
# 口径：丢弃每份日志前 10% 的样本（启动 warmup 波动），取其余样本
#       tok/s 的中位数作为稳态吞吐；同时给出均值、峰值与显存峰值。
# ============================================================
import re
import statistics
import sys
from pathlib import Path

TOK_RE = re.compile(r'([\d,]+)\s*tok/s')
MEM_RE = re.compile(r'峰值显存\s*([\d,]+)\s*MB')
TIME_RE = re.compile(r'总用时\s*([\d.]+)\s*h')


def parse_log(path: Path) -> dict:
    """解析一份 train_log.txt，返回吞吐统计。"""
    tok, mem, total_h = [], [], None
    for line in path.read_text(encoding='utf-8', errors='ignore').splitlines():
        m = TOK_RE.search(line)
        if m:
            tok.append(int(m.group(1).replace(',', '')))
        m = MEM_RE.search(line)
        if m:
            mem.append(int(m.group(1).replace(',', '')))
        m = TIME_RE.search(line)
        if m:
            total_h = float(m.group(1))
    if not tok:
        return {}
    k = max(1, len(tok) // 10)          # 丢前 10% warmup 样本
    steady = tok[k:] or tok
    return {
        'samples': len(tok),
        'steady_median': statistics.median(steady),
        'steady_mean': statistics.mean(steady),
        'peak': max(tok),
        'mem_max_mb': max(mem) if mem else 0,
        'total_h': total_h,
    }


def fmt(x: float) -> str:
    return f'{x:,.0f}'


def compare_pair(name: str, before: Path, after: Path) -> None:
    a, b = parse_log(before), parse_log(after)
    if not a or not b:
        print(f'[跳过] {name}：日志解析失败或无 tok/s 记录（{before} / {after}）')
        return
    gain = (b['steady_median'] / a['steady_median'] - 1) * 100 if a['steady_median'] else 0
    print(f'\n【{name}】')
    print(f'  样本数        {a["samples"]:>10}     {b["samples"]:>10}')
    print(f'  稳态中位数    {fmt(a["steady_median"]):>10}     {fmt(b["steady_median"]):>10}     {gain:+.1f}%')
    print(f'  稳态均值      {fmt(a["steady_mean"]):>10}     {fmt(b["steady_mean"]):>10}')
    print(f'  单条峰值      {fmt(a["peak"]):>10}     {fmt(b["peak"]):>10}')
    print(f'  峰值显存 MB   {a["mem_max_mb"]:>10,}     {b["mem_max_mb"]:>10,}')
    if a['total_h'] and b['total_h']:
        t_gain = (1 - b['total_h'] / a['total_h']) * 100
        print(f'  总用时 h      {a["total_h"]:>10.2f}     {b["total_h"]:>10.2f}     节省 {t_gain:.1f}%')
    print('  （对比有效前提：两次训练档位、数据、超参完全一致）')


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    a, b = Path(sys.argv[1]), Path(sys.argv[2])
    if a.is_dir() and b.is_dir():
        pairs = []
        for f in sorted(a.glob('*/train_log.txt')):
            g = b / f.parent.name / 'train_log.txt'
            if g.exists():
                pairs.append((f.parent.name, f, g))
        if not pairs:
            sys.exit(f'在 {a} 与 {b} 下没有找到同名档位的 train_log.txt')
        print(f'共 {len(pairs)} 组：' + ', '.join(p[0] for p in pairs))
        for name, f, g in pairs:
            compare_pair(name, f, g)
    elif a.is_file() and b.is_file():
        compare_pair(a.parent.name or 'train', a, b)
    else:
        sys.exit('两个参数需同为目录或同为文件')


if __name__ == '__main__':
    main()
