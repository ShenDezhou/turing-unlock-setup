#!/usr/bin/env bash
# clean_up.sh — 还原（卸载 turing-unlock、恢复 GRUB）操作备忘清单。
# 非自动执行脚本：逐条复制粘贴到终端执行，按行内注释确认每步结果。

# 1) 卸载包（purge 连 initramfs hook 配置一起删；卸载过程会自动重建 initramfs）
sudo apt purge -y turing-unlock

# 2) 确认 initramfs 已清干净（应无任何输出）
lsinitramfs /boot/initrd.img-$(uname -r) | grep -i turing

# 3) 还原 GRUB 内核参数，二选一：
# 3a) 装完后没再改过 /etc/default/grub → 先 diff 确认差异只有那一个参数，再用脚本留的备份还原
diff /etc/default/grub.bak.turing-unlock /etc/default/grub
sudo cp /etc/default/grub.bak.turing-unlock /etc/default/grub

# 3b) 装完后还改过 grub → 手动删参数（备份别覆盖）
sudo sed -i 's/ turing_unlock\.targets=[0-9a-zA-Z:.,]*//g' /etc/default/grub

# 确认参数已不在（应看不到 turing_unlock）
grep GRUB_CMDLINE_LINUX_DEFAULT /etc/default/grub

# 4) 重建 GRUB
sudo update-grub

# 5) 重启
sudo reboot