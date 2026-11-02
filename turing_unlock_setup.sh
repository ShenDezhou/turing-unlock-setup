#!/usr/bin/env bash
# ============================================================
# turing_unlock_setup.sh — 一键安装并启用 turing-unlock
#
# 作用：解锁 RTX 2080 Ti (TU102) 的 FP16 Tensor Core 全速运行
# 原理：内核模块在 initramfs 阶段、nvidia 驱动绑定显卡之前，
#       写入速度选择器（SS0=0x88888888, SS1=0x8）并释放 ACR mutex 3
# 实测（原仓库）：FP16 matmul 从 ~50 TFLOPS 提升到 ~68-91 TFLOPS
# 仓库：https://github.com/nlzy/turing-unlock
#       （Zagitta/gpu-unlock 的 Turing 移植，GPL-2.0）
#
# 用法：sudo bash turing_unlock_setup.sh
# 可选：TURING_UNLOCK_TARGETS=0000:21:00.0 sudo bash turing_unlock_setup.sh
#       （单卡默认 all，无需指定）
# ============================================================
set -euo pipefail

REPO_URL="https://github.com/nlzy/turing-unlock"
CLONE_DIR="${TURING_UNLOCK_DIR:-$HOME/turing-unlock}"
TARGETS="${TURING_UNLOCK_TARGETS:-all}"

log()  { printf '\033[1;32m[✓]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[!]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[✗]\033[0m %s\n' "$*" >&2; exit 1; }

# ---------- 0. root 检查 ----------
[[ $EUID -eq 0 ]] || die "请用 sudo 运行：sudo bash turing_unlock_setup.sh"

# ---------- 1. 风险确认 ----------
if [[ "${TURING_UNLOCK_ASSUME_YES:-0}" != "1" ]]; then
  cat <<'BANNER'
==========================================================
 ⚠ 风险提示（来自 turing-unlock 原仓库 README）
 解锁后 Tensor Core 满血运行，功耗大幅升高，
 可能损坏硬件。请确保供电与散热充足，后果自负。
==========================================================
BANNER
  read -r -p "继续？[y/N] " ans
  [[ "$ans" == "y" || "$ans" == "Y" ]] || die "已取消"
fi

# ---------- 2. 检查 GPU（TU102 = 10de:1e07） ----------
command -v lspci >/dev/null 2>&1 || apt-get install -y -qq pciutils
GPU=$(lspci -D -n 2>/dev/null | grep -i '10de:1e07' | head -1 | awk '{print $1}')
if [[ -n "$GPU" ]]; then
  log "检测到 RTX 2080 Ti (TU102, 10de:1e07)：$GPU"
else
  warn "未检测到 10de:1e07 (TU102)。原仓库仅在 Dell RTX 2080 Ti 22GB 上测试过。"
  read -r -p "其他卡未测试，仍要继续？[y/N] " ans2
  [[ "$ans2" == y* || "$ans2" == Y* ]] || die "已取消"
fi

# ---------- 3. 驱动版本（仅供记录，不阻塞） ----------
if command -v nvidia-smi >/dev/null 2>&1; then
  DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
  log "NVIDIA 驱动版本：${DRV:-未知}（原仓库测试版本 575.51.03，专有/开源驱动均可）"
else
  warn "未找到 nvidia-smi，请确认 NVIDIA 驱动已安装后再训练"
fi

# ---------- 4. 构建依赖（Debian/Ubuntu） ----------
if command -v apt-get >/dev/null 2>&1; then
  NEED=()
  command -v git  >/dev/null 2>&1 || NEED+=(git)
  command -v make >/dev/null 2>&1 || NEED+=(make)
  command -v gcc  >/dev/null 2>&1 || NEED+=(gcc)
  dpkg -s build-essential >/dev/null 2>&1 || NEED+=(build-essential)
  dpkg -s dpkg-dev        >/dev/null 2>&1 || NEED+=(dpkg-dev)
  if (( ${#NEED[@]} )); then
    log "安装构建依赖：${NEED[*]}"
    apt-get update -qq
    apt-get install -y -qq "${NEED[@]}"
  else
    log "构建依赖齐全"
  fi
else
  warn "非 Debian/Ubuntu 系，请自行确认 git / make / gcc 已安装"
fi

# ---------- 5. 克隆仓库 ----------
if [[ -d "$CLONE_DIR/.git" ]]; then
  git -C "$CLONE_DIR" pull --ff-only 2>/dev/null && log "仓库已更新：$CLONE_DIR" \
    || warn "git pull 失败，使用现有副本：$CLONE_DIR"
else
  rm -rf "$CLONE_DIR.tmp"
  git clone --depth 1 "$REPO_URL" "$CLONE_DIR.tmp"
  rm -rf "$CLONE_DIR"
  mv "$CLONE_DIR.tmp" "$CLONE_DIR"
  log "仓库已克隆：$CLONE_DIR"
fi

# ---------- 6. 安装（make install 自动识别发行版；Ubuntu 走 deb 路线） ----------
# 构建会下载两个固定版本的 NVIDIA 文件并校验 SHA-256（缓存于 ~/.cache/turing-unlock/inputs）
cd "$CLONE_DIR"
make install
log "turing-unlock 模块与固件已安装，并已配置进 initramfs（先于 nvidia 驱动加载）"

# ---------- 7. 内核命令行参数（幂等） ----------
GRUB_FILE=/etc/default/grub
if [[ -f "$GRUB_FILE" ]]; then
  if grep -q 'turing_unlock\.targets=' "$GRUB_FILE"; then
    log "内核参数已存在，跳过修改"
  else
    cp "$GRUB_FILE" "${GRUB_FILE}.bak.turing-unlock"
    # 在 GRUB_CMDLINE_LINUX_DEFAULT 的引号内追加参数：\1 捕获引号内的原值，
    # 替换时把原值放回引号内再追加新参数，保证引号始终成对（空值时同样适用）
    sed -i 's|^GRUB_CMDLINE_LINUX_DEFAULT="\(.*\)"$|GRUB_CMDLINE_LINUX_DEFAULT="\1 turing_unlock.targets='"${TARGETS}"'"|' "$GRUB_FILE"
    if grep -q 'turing_unlock\.targets=' "$GRUB_FILE" && sh -n "$GRUB_FILE" 2>/dev/null; then
      log "已写入内核参数 turing_unlock.targets=${TARGETS}（原文件备份为 ${GRUB_FILE}.bak.turing-unlock）"
    elif grep -q 'turing_unlock\.targets=' "$GRUB_FILE"; then
      warn "参数已写入，但 /etc/default/grub 语法检查未通过（可能是原文件就有问题），继续尝试 update-grub"
    else
      warn "自动写入 GRUB 失败，请手动在 /etc/default/grub 的 GRUB_CMDLINE_LINUX_DEFAULT 里加上："
      warn "  turing_unlock.targets=${TARGETS}"
    fi
    if command -v update-grub >/dev/null 2>&1; then
      update-grub || die "update-grub 失败：/etc/default/grub 语法有误。恢复方法：sudo cp ${GRUB_FILE}.bak.turing-unlock ${GRUB_FILE} 后重跑本脚本；若仍失败，把 'sed -n '30,45p' ${GRUB_FILE}' 的输出发回排查"
      log "GRUB 已更新"
    else
      warn "未找到 update-grub，请手动执行 grub-mkconfig 重建引导配置"
    fi
  fi
else
  warn "未找到 /etc/default/grub（可能是 systemd-boot/UKI 引导），请按原仓库 README 手动添加内核参数："
  warn "  turing_unlock.targets=${TARGETS}"
fi

# ---------- 8. 完成 ----------
cat <<'DONE'

==========================================================
 安装完成。接下来：

 1) 重启电脑：
      sudo reboot

 2) 重启后验证解锁是否生效：
      journalctl -k -b | grep turing_unlock
    有输出（无 error）即生效；无输出说明参数未生效，检查上面第 7 步。

 3) 用与优化前【完全相同】的命令重跑训练
    （同一档位、同一数据、同一超参，例：pico 档，2451 步约 20 分钟）

 4) 训练完成后对比吞吐量：
      python3 compare_throughput.py <优化前ckpt目录> <优化后ckpt目录>
    或单文件：python3 compare_throughput.py a/train_log.txt b/train_log.txt

 如需还原：sudo apt remove turing-unlock   （并从 /etc/default/grub
 删除 turing_unlock.targets=... 后 update-grub；备份在同目录 .bak.turing-unlock）
==========================================================
DONE
