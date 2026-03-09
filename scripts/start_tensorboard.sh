#!/bin/bash
# TensorBoard 启动脚本（DreamRec）
# 用法: bash scripts/start_tensorboard.sh
# 本地 Mac 访问: ssh -L 6006:localhost:6006 sjj@lab-3 → http://localhost:6006

LOGDIR="/home/sjj/wenhao/DreamRec/tensorboard"

echo "启动 TensorBoard..."
echo "日志目录: $LOGDIR"
echo ""
echo "如需远程访问，请在本地 Mac 运行:"
echo "  ssh -L 6006:localhost:6006 sjj@lab-3"
echo "  然后浏览器打开: http://localhost:6006"
echo ""

tensorboard --logdir="$LOGDIR" --port=6006 --bind_all
