#!/bin/bash
# =============================================================================
# DreamRec Yelp 训练脚本
# 数据: /home/sjj/wenhao/DreamRec/data/yelp/
# 环境: conda activate DDBC
# 用法: bash scripts/train_yelp.sh
# =============================================================================

# ------------- 环境 -------------
export CUDA_VISIBLE_DEVICES=0
# CONDA_ENV="DDBC"           # 请在运行前手动 conda activate DDBC
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# ------------- 运行名（用于日志区分） -------------
RUN_NAME="dreamrec-yelp-$(date +%Y%m%d-%H%M%S)"
OUTPUT_DIR="$PROJECT_DIR/outputs/yelp/$RUN_NAME"
mkdir -p "$OUTPUT_DIR"

echo "=================================================="
echo "  DreamRec Yelp Training"
echo "  Run: $RUN_NAME"
echo "  Output: $OUTPUT_DIR"
echo "=================================================="

# =============================================================================
# 超参设置（所有可调参数集中在此处）
# =============================================================================

# --- 数据 ---
DATA="yelp"                   # 数据集名称，对应 data/{DATA}/ 目录

# --- 训练流程 ---
EPOCH=75                    # 最大训练轮数
BATCH_SIZE=256                # mini-batch 大小
RANDOM_SEED=100               # 随机种子（影响参数初始化、batch采样等）

# --- 模型结构 ---
HIDDEN_FACTOR=64              # item embedding 维度 = Transformer 隐层维度 = 扩散空间维度
                              # 必须和 DDBC 的 model.hidden_size 保持一致以便公平对比
DIFFUSER_TYPE="mlp1"          # 去噪网络结构:
                              #   mlp1: 单层线性 Linear(3H → H)，参数量最少、最快
                              #   mlp2: 两层 MLP Linear(3H→2H) + GELU + Linear(2H→H)，表达更强
# LAYERS=1                    # 定义了但代码未使用（Transformer固定单层），无需设置

# --- 正则与dropout ---
DROPOUT_RATE=0.15              # Transformer 中的 dropout 概率（attention + FFN + embedding）
L2_DECAY=1e-4                    # 优化器的 weight decay（L2正则系数），0表示不正则化

# --- 优化器 ---
OPTIMIZER="adamw"             # 优化器类型: adam / adamw / adagrad / rmsprop
LR=0.001                      # 学习率

# --- 扩散过程 ---
TIMESTEPS=500                 # 扩散步数 T: 越大去噪越精细但推理越慢
                              #   yc 论文用 500，ks 用 2000，zhihu 用 500
BETA_SCHE="exp"               # 噪声调度策略:
                              #   exp:    指数增长（yc/zhihu 默认，收敛稳定）
                              #   cosine: 余弦衰减（ks 默认，适合更多步数）
                              #   linear: 线性，需手动设置 beta_start/beta_end
                              #   sqrt:   平方根，激进噪声增长
BETA_START=0.0001             # 仅 linear 调度时生效，beta 起始值
BETA_END=0.02                 # 仅 linear 调度时生效，beta 终止值

# --- Classifier-Free Guidance ---
W=2                           # 引导强度 w
P=0.1                         # 训练时 context dropout 概率

# --- DDBC 兼容评估 ---
PREDICT_NUMS="3,5"          # 每次预测的 item 数，对应 DDBC 的 predict_num_items
CANDIDATE_MULTIPLIERS="19"    # 训练期间 valid 只用 x19（单一倍数，避免重复 diffusion）
                              # 完整 4 倍数评估（9/19/49/99）由 eval_best.sh 在训练结束后执行
EVAL_FREQ=10                  # 每隔多少 epoch 做一次 DDBC 评估
PREDICT_MODE="ar"         # 预测模式:
                              #   single: 单次扩散推理 → top-k（原始 DreamRec 行为）
                              #   ar:     自回归模式 — 每步扩散推理取 top-1，追加到历史，重复 k 次
                              #           理论上更准确，但推理耗时约为 single 的 k 倍

# =============================================================================
# 开始训练
# =============================================================================

LOG_FILE="$OUTPUT_DIR/train.log"
TB_LOG_DIR="$PROJECT_DIR/tensorboard/$DATA/$RUN_NAME"
SAVE_DIR="$OUTPUT_DIR"

echo "Logging to: $LOG_FILE"
echo "TensorBoard: $TB_LOG_DIR"
echo ""

python -u "$PROJECT_DIR/DreamRec.py" \
    --data         "$DATA"          \
    --epoch        $EPOCH           \
    --batch_size   $BATCH_SIZE      \
    --random_seed  $RANDOM_SEED     \
    --hidden_factor $HIDDEN_FACTOR  \
    --diffuser_type "$DIFFUSER_TYPE" \
    --dropout_rate  $DROPOUT_RATE   \
    --l2_decay      $L2_DECAY       \
    --optimizer     "$OPTIMIZER"    \
    --lr            $LR             \
    --timesteps     $TIMESTEPS      \
    --beta_sche     "$BETA_SCHE"    \
    --beta_start    $BETA_START     \
    --beta_end      $BETA_END       \
    --w             $W              \
    --p             $P              \
    --predict_nums  "$PREDICT_NUMS"          \
    --candidate_multipliers "$CANDIDATE_MULTIPLIERS" \
    --eval_freq     $EVAL_FREQ               \
    --predict_mode  "$PREDICT_MODE"          \
    --tb_log_dir    "$TB_LOG_DIR"            \
    --save_dir      "$SAVE_DIR"              \
    --cuda          $CUDA_VISIBLE_DEVICES    \
    --descri        "$RUN_NAME"              \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "=================================================="
echo "  Training finished. Log saved to:"
echo "  $LOG_FILE"
echo "=================================================="
