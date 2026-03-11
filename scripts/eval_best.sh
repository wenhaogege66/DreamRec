#!/bin/bash
# =============================================================================
# DreamRec 最优模型评估脚本
# 用法: bash scripts/eval_best.sh [best_model.pt 路径]
# 示例: bash scripts/eval_best.sh outputs/yelp/dreamrec-yelp-20260308-224619/best_model.pt
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

CKPT="${1:-$PROJECT_DIR/outputs/yelp/dreamrec-yelp-20260308-224619/best_model.pt}"

# 检查 checkpoint 是否存在
if [ ! -f "$CKPT" ]; then
    echo "ERROR: checkpoint not found: $CKPT"
    echo "用法: bash scripts/eval_best.sh <best_model.pt 路径>"
    exit 1
fi

echo "=================================================="
echo "  DreamRec Best Model Evaluation"
echo "  Checkpoint: $CKPT"
echo "=================================================="

cd "$PROJECT_DIR"
python -u eval_best.py \
    --ckpt               "$CKPT"     \
    --data               yelp        \
    --cuda               0           \
    --hidden_factor      64          \
    --diffuser_type      mlp1        \
    --dropout_rate       0.15        \
    --timesteps          500         \
    --beta_sche          exp         \
    --beta_start         0.0001      \
    --beta_end           0.02        \
    --w                  2           \
    --predict_nums       3,5         \
    --candidate_multipliers 9,19,49,99 \
    --seed               1           \
    --predict_mode       ar
