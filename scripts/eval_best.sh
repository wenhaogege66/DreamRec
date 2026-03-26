#!/bin/bash
# =============================================================================
# DreamRec 最优模型评估脚本 — 串行对比多个 checkpoint
# 用法: bash scripts/eval_best.sh
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# =============================================================================
# 待评估的 checkpoint 列表（按学习率标注）
# =============================================================================
CKPTS=(
    "/home/sjj/wenhao/DreamRec/outputs/yelp/dreamrec-yelp-lr0.01-20260326-004308/best_model.pt"
    "/home/sjj/wenhao/DreamRec/outputs/yelp/dreamrec-yelp-lr0.0005-20260326-042717/best_model.pt"
    "/home/sjj/wenhao/DreamRec/outputs/yelp/dreamrec-yelp-lr0.00005-20260326-081323/best_model.pt"
)
LABELS=("lr=0.01" "lr=0.0005" "lr=0.00005")

# =============================================================================
# 固定评估参数
# =============================================================================
DATA="yelp"
HIDDEN_FACTOR=64
DIFFUSER_TYPE="mlp1"
DROPOUT_RATE=0.15
TIMESTEPS=500
BETA_SCHE="exp"
BETA_START=0.0001
BETA_END=0.02
W=10
PREDICT_NUMS=3
CANDIDATE_MULTIPLIERS=19
SEED=1
PREDICT_MODE="ar"

# =============================================================================
# 串行评估
# =============================================================================
cd "$PROJECT_DIR"

TOTAL=${#CKPTS[@]}
SUMMARY_LINES=()

for i in "${!CKPTS[@]}"; do
    CKPT="${CKPTS[$i]}"
    LABEL="${LABELS[$i]}"
    RUN_IDX=$((i + 1))

    echo ""
    echo "=================================================="
    echo "  DreamRec Eval  [$RUN_IDX/$TOTAL]  $LABEL"
    echo "  Checkpoint: $CKPT"
    echo "=================================================="

    if [ ! -f "$CKPT" ]; then
        echo "  ERROR: checkpoint not found, skipping."
        SUMMARY_LINES+=("  [$LABEL]  SKIPPED — file not found: $CKPT")
        continue
    fi

    python -u eval_best.py \
        --ckpt                  "$CKPT"              \
        --data                  "$DATA"              \
        --cuda                  0                    \
        --hidden_factor         $HIDDEN_FACTOR       \
        --diffuser_type         "$DIFFUSER_TYPE"     \
        --dropout_rate          $DROPOUT_RATE        \
        --timesteps             $TIMESTEPS           \
        --beta_sche             "$BETA_SCHE"         \
        --beta_start            $BETA_START          \
        --beta_end              $BETA_END            \
        --w                     $W                   \
        --predict_nums          $PREDICT_NUMS        \
        --candidate_multipliers $CANDIDATE_MULTIPLIERS \
        --seed                  $SEED                \
        --predict_mode          "$PREDICT_MODE"

    EXIT_CODE=$?
    STATUS=$( [ $EXIT_CODE -eq 0 ] && echo "OK" || echo "FAILED(exit=$EXIT_CODE)" )
    SUMMARY_LINES+=("  [$LABEL]  $STATUS  —  $CKPT")
done

echo ""
echo "=================================================="
echo "  所有评估完成，汇总如下："
for line in "${SUMMARY_LINES[@]}"; do
    echo "$line"
done
echo "=================================================="
