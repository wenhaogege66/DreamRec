#!/usr/bin/env python3
"""
DreamRec 最优模型评估脚本

用法:
    python eval_best.py --ckpt <path/to/best_model.pt> [options]

或通过 shell 包装:
    bash scripts/eval_best.sh [path/to/best_model.pt]
"""
import argparse
import os
import sys
import torch
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description='Evaluate DreamRec best checkpoint')
    p.add_argument('--ckpt', type=str, required=True,
                   help='best_model.pt 路径')
    p.add_argument('--data', type=str, default='yelp',
                   help='数据集名称 (yc / ks / yelp)')
    p.add_argument('--cuda', type=int, default=0)

    # 模型结构（必须与训练时完全一致）
    p.add_argument('--hidden_factor',  type=int,   default=64)
    p.add_argument('--diffuser_type',  type=str,   default='mlp1')
    p.add_argument('--dropout_rate',   type=float, default=0.15)
    p.add_argument('--timesteps',      type=int,   default=500)
    p.add_argument('--beta_sche',      type=str,   default='exp')
    p.add_argument('--beta_start',     type=float, default=0.0001)
    p.add_argument('--beta_end',       type=float, default=0.02)
    p.add_argument('--w',              type=float, default=2.0)

    # 评估参数
    p.add_argument('--predict_nums',          type=str, default='3,5',
                   help='逗号分隔的预测数量列表，如 "3,5"')
    p.add_argument('--candidate_multipliers', type=str, default='9,19,49,99',
                   help='逗号分隔的候选集倍数列表')
    p.add_argument('--seed',                  type=int, default=1,
                   help='随机种子（test 候选集沿用 seed=1 与 DDBC 对齐）')
    p.add_argument('--predict_mode',          type=str, default='single',
                   choices=['single', 'ar'],
                   help='single=one-shot top-k; ar=autoregressive top-1×k')
    return p.parse_args()


eval_args = parse_args()

# CUDA 设备需在 torch CUDA 初始化之前设置
os.environ['CUDA_VISIBLE_DEVICES'] = str(eval_args.cuda)

# 以最简 argv 导入 DreamRec，避免其模块级 parse_args() 与本脚本冲突
_saved_argv = sys.argv[:]
sys.argv = ['DreamRec.py']
import DreamRec as DR
sys.argv = _saved_argv


# ---- 配置 DreamRec 全局变量 ----
DR.data_directory = f'./data/{eval_args.data}'
data_statis = pd.read_pickle(os.path.join(DR.data_directory, 'data_statis.df'))
DR.seq_size = data_statis['seq_size'][0]
DR.item_num = data_statis['item_num'][0]

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# ---- 构建模型（结构必须与训练时完全一致）----
model = DR.Tenc(eval_args.hidden_factor, DR.item_num, DR.seq_size,
                eval_args.dropout_rate, eval_args.diffuser_type, device)
diff  = DR.diffusion(eval_args.timesteps, eval_args.beta_start,
                     eval_args.beta_end, eval_args.w)
model.to(device)

# ---- 加载 checkpoint ----
ckpt = torch.load(eval_args.ckpt, map_location=device)
model.load_state_dict(ckpt['model_state_dict'])
print(f'Loaded: {eval_args.ckpt}')
print(f'  best epoch = {ckpt["epoch"]},  val_recall = {ckpt["val_recall"]:.4f}')
print(f'  item_num={DR.item_num},  seq_size={DR.seq_size},  device={device}')

# ---- 评估 ----
predict_nums = [int(x) for x in eval_args.predict_nums.split(',')]
multipliers  = [int(x) for x in eval_args.candidate_multipliers.split(',')]

print('\n========== TEST RESULTS (best model) ==========')
DR.evaluate_ddbc(model, diff, device,
                 predict_nums, multipliers, eval_args.seed,
                 writer=None, epoch=None, split='test',
                 predict_mode=eval_args.predict_mode)
print('=' * 48)
