import numpy as np
import pandas as pd
import math
import random
import argparse
import pickle
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
import os
import logging
import time as Time
from utility import pad_history,calculate_hit,extract_axis_1
from collections import Counter
from Modules_ori import *

logging.getLogger().setLevel(logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser(description="Run supervised GRU.")

    parser.add_argument('--epoch', type=int, default=1000,
                        help='Number of max epochs.')
    parser.add_argument('--data', nargs='?', default='yc',
                        help='yc, ks, zhihu')
    parser.add_argument('--random_seed', type=int, default=100,
                        help='random seed')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--layers', type=int, default=1,
                        help='gru_layers')
    parser.add_argument('--hidden_factor', type=int, default=64,
                        help='Number of hidden factors, i.e., embedding size.')
    parser.add_argument('--timesteps', type=int, default=200,
                        help='timesteps for diffusion')
    parser.add_argument('--beta_end', type=float, default=0.02,
                        help='beta end of diffusion')
    parser.add_argument('--beta_start', type=float, default=0.0001,
                        help='beta start of diffusion')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='Learning rate.')
    parser.add_argument('--l2_decay', type=float, default=0,
                        help='l2 loss reg coef.')
    parser.add_argument('--cuda', type=int, default=0,
                        help='cuda device.')
    parser.add_argument('--dropout_rate', type=float, default=0.1,
                        help='dropout ')
    parser.add_argument('--w', type=float, default=2.0,
                        help='dropout ')
    parser.add_argument('--p', type=float, default=0.1,
                        help='dropout ')
    parser.add_argument('--report_epoch', type=bool, default=True,
                        help='report frequency')
    parser.add_argument('--diffuser_type', type=str, default='mlp1',
                        help='type of diffuser.')
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='type of optimizer.')
    parser.add_argument('--beta_sche', nargs='?', default='exp',
                        help='')
    parser.add_argument('--descri', type=str, default='',
                        help='description of the work.')
    # --- DDBC-compatible evaluation ---
    parser.add_argument('--predict_nums', type=str, default='1,3,5',
                        help='Comma-separated list of top-k items to predict, e.g. "1,3,5"')
    parser.add_argument('--candidate_multipliers', type=str, default='9,19,49,99',
                        help='Comma-separated candidate multipliers; pool_size = 1 + multiplier')
    parser.add_argument('--eval_freq', type=int, default=10,
                        help='Run DDBC-style evaluation every N epochs')
    parser.add_argument('--predict_mode', type=str, default='single',
                        choices=['single', 'ar'],
                        help='Prediction mode: '
                             'single=one-shot diffusion → top-k; '
                             'ar=autoregressive diffusion top-1 × k (update history each step)')
    parser.add_argument('--tb_log_dir', type=str, default='',
                        help='TensorBoard log directory (default: ./tensorboard/{data}/{descri})')
    parser.add_argument('--save_dir', type=str, default='',
                        help='Directory to save best model (default: ./outputs/{data}/)')
    return parser.parse_args()

args = parse_args()

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

setup_seed(args.random_seed)


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def linear_beta_schedule(timesteps, beta_start, beta_end):
    beta_start = beta_start
    beta_end = beta_end
    return torch.linspace(beta_start, beta_end, timesteps)

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def exp_beta_schedule(timesteps, beta_min=0.1, beta_max=10):
    x = torch.linspace(1, 2 * timesteps + 1, timesteps)
    betas = 1 - torch.exp(- beta_min / timesteps - x * 0.5 * (beta_max - beta_min) / (timesteps * timesteps))
    return betas

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

class diffusion():
    def __init__(self, timesteps, beta_start, beta_end, w):
        self.timesteps = timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.w = w

        if args.beta_sche == 'linear':
            self.betas = linear_beta_schedule(timesteps=self.timesteps, beta_start=self.beta_start, beta_end=self.beta_end)
        elif args.beta_sche == 'exp':
            self.betas = exp_beta_schedule(timesteps=self.timesteps)
        elif args.beta_sche =='cosine':
            self.betas = cosine_beta_schedule(timesteps=self.timesteps)
        elif args.beta_sche =='sqrt':
            self.betas = torch.tensor(betas_for_alpha_bar(self.timesteps, lambda t: 1-np.sqrt(t + 0.0001),)).float()

        # define alphas 
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod - 1)


        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
    
    def q_sample(self, x_start, t, noise=None):
        # print(self.betas)
        if noise is None:
            noise = torch.randn_like(x_start)
            # noise = torch.randn_like(x_start) / 100
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(self, denoise_model, x_start, h, t, noise=None, loss_type="l2"):
        # 
        if noise is None:
            noise = torch.randn_like(x_start) 
            # noise = torch.randn_like(x_start) / 100
        
        # 
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)


        predicted_x = denoise_model(x_noisy, h, t)

        
        # 
        if loss_type == 'l1':
            loss = F.l1_loss(x_start, predicted_x)
        elif loss_type == 'l2':
            loss = F.mse_loss(x_start, predicted_x)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(x_start, predicted_x)
        else:
            raise NotImplementedError()

        return loss, predicted_x

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )
    
    @torch.no_grad()
    def p_sample(self, model_forward, model_forward_uncon, x, h, t, t_index):

        x_start = (1 + self.w) * model_forward(x, h, t) - self.w * model_forward_uncon(x, t)
        x_t = x 
        model_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)

            return model_mean + torch.sqrt(posterior_variance_t) * noise 
        
    @torch.no_grad()
    def sample(self, model_forward, model_forward_uncon, h):
        x = torch.randn_like(h)
        # x = torch.randn_like(h) / 100

        for n in reversed(range(0, self.timesteps)):
            x = self.p_sample(model_forward, model_forward_uncon, x, h, torch.full((h.shape[0], ), n, device=h.device, dtype=torch.long), n)

        return x



class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
    
        
class Tenc(nn.Module):
    def __init__(self, hidden_size, item_num, state_size, dropout, diffuser_type, device, num_heads=1):
        super(Tenc, self).__init__()
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.item_num = int(item_num)
        self.dropout = nn.Dropout(dropout)
        self.diffuser_type = diffuser_type
        self.device = device
        self.item_embeddings = nn.Embedding(
            num_embeddings=item_num + 1,
            embedding_dim=hidden_size,
        )
        nn.init.normal_(self.item_embeddings.weight, 0, 1)
        self.none_embedding = nn.Embedding(
            num_embeddings=1,
            embedding_dim=self.hidden_size,
        )
        nn.init.normal_(self.none_embedding.weight, 0, 1)
        self.positional_embeddings = nn.Embedding(
            num_embeddings=state_size,
            embedding_dim=hidden_size
        )
        # emb_dropout is added
        self.emb_dropout = nn.Dropout(dropout)
        self.ln_1 = nn.LayerNorm(hidden_size)
        self.ln_2 = nn.LayerNorm(hidden_size)
        self.ln_3 = nn.LayerNorm(hidden_size)
        self.mh_attn = MultiHeadAttention(hidden_size, hidden_size, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(hidden_size, hidden_size, dropout)
        self.s_fc = nn.Linear(hidden_size, item_num)
        # self.ac_func = nn.ReLU()

        # self.step_embeddings = nn.Embedding(
        #     num_embeddings=50,
        #     embedding_dim=hidden_size
        # )

        self.step_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size*2),
            nn.GELU(),
            nn.Linear(self.hidden_size*2, self.hidden_size),
        )

        self.emb_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.hidden_size*2)
        )

        self.diff_mlp = nn.Sequential(
            nn.Linear(self.hidden_size * 3, self.hidden_size*2),
            nn.GELU(),
            nn.Linear(self.hidden_size*2, self.hidden_size),
        )


        if self.diffuser_type =='mlp1':
            self.diffuser = nn.Sequential(
                nn.Linear(self.hidden_size*3, self.hidden_size)
        )
        elif self.diffuser_type =='mlp2':
            self.diffuser = nn.Sequential(
            nn.Linear(self.hidden_size * 3, self.hidden_size*2),
            nn.GELU(),
            nn.Linear(self.hidden_size*2, self.hidden_size)
        )


    def forward(self, x, h, step):

        t = self.step_mlp(step)


        if self.diffuser_type == 'mlp1':
            res = self.diffuser(torch.cat((x, h, t), dim=1))
        elif self.diffuser_type == 'mlp2':
            res = self.diffuser(torch.cat((x, h, t), dim=1))
        return res

    def forward_uncon(self, x, step):
        h = self.none_embedding(torch.tensor([0]).to(self.device))
        h = torch.cat([h.view(1, 64)]*x.shape[0], dim=0)

        t = self.step_mlp(step)

        if self.diffuser_type == 'mlp1':
            res = self.diffuser(torch.cat((x, h, t), dim=1))
        elif self.diffuser_type == 'mlp2':
            res = self.diffuser(torch.cat((x, h, t), dim=1))
            
        return res

        # return x

    def cacu_x(self, x):
        x = self.item_embeddings(x)

        return x

    def cacu_h(self, states, len_states, p):
        #hidden
        inputs_emb = self.item_embeddings(states)
        inputs_emb += self.positional_embeddings(torch.arange(self.state_size).to(self.device))
        seq = self.emb_dropout(inputs_emb)
        mask = torch.ne(states, self.item_num).float().unsqueeze(-1).to(self.device)
        seq *= mask
        seq_normalized = self.ln_1(seq)
        mh_attn_out = self.mh_attn(seq_normalized, seq)
        ff_out = self.feed_forward(self.ln_2(mh_attn_out))
        ff_out *= mask
        ff_out = self.ln_3(ff_out)
        state_hidden = extract_axis_1(ff_out, len_states - 1)
        h = state_hidden.squeeze()

        B, D = h.shape[0], h.shape[1]
        mask1d = (torch.sign(torch.rand(B) - p) + 1) / 2
        maske1d = mask1d.view(B, 1)
        mask = torch.cat([maske1d] * D, dim=1)
        mask = mask.to(self.device)

        # print(h.device, self.none_embedding(torch.tensor([0]).to(self.device)).device, mask.device)
        h = h * mask + self.none_embedding(torch.tensor([0]).to(self.device)) * (1-mask)


        return h  
    
    def predict(self, states, len_states, diff, candidate_ids=None):
        """
        Args:
            states:        [B, seq_size] LongTensor
            len_states:    [B] numpy array
            diff:          diffusion object
            candidate_ids: None → score full item pool (original behavior)
                           [B, n_cand] LongTensor → score only these candidates per sample
        Returns:
            scores: [B, n_items] or [B, n_cand]
        """
        inputs_emb = self.item_embeddings(states)
        inputs_emb += self.positional_embeddings(torch.arange(self.state_size).to(self.device))
        seq = self.emb_dropout(inputs_emb)
        mask = torch.ne(states, self.item_num).float().unsqueeze(-1).to(self.device)
        seq *= mask
        seq_normalized = self.ln_1(seq)
        mh_attn_out = self.mh_attn(seq_normalized, seq)
        ff_out = self.feed_forward(self.ln_2(mh_attn_out))
        ff_out *= mask
        ff_out = self.ln_3(ff_out)
        state_hidden = extract_axis_1(ff_out, len_states - 1)
        h = state_hidden.squeeze()

        x = diff.sample(self.forward, self.forward_uncon, h)

        if candidate_ids is not None:
            # candidate_ids: [B, n_cand]
            # cand_emb:      [B, n_cand, H]
            cand_emb = self.item_embeddings(candidate_ids)
            scores = torch.bmm(cand_emb, x.unsqueeze(-1)).squeeze(-1)  # [B, n_cand]
        else:
            test_item_emb = self.item_embeddings.weight
            scores = torch.matmul(x, test_item_emb.transpose(0, 1))

        return scores



def evaluate(model, test_data, diff, device):
    """原始 HR/NDCG 评估（对全量item pool打分），仅用于快速监控，不作为最终指标。"""
    eval_data=pd.read_pickle(os.path.join(data_directory, test_data))

    batch_size = 100
    total_purchase = 0.0
    hit_purchase=[0,0,0,0]
    ndcg_purchase=[0,0,0,0]

    seq, len_seq, target = list(eval_data['seq'].values), list(eval_data['len_seq'].values), list(eval_data['next'].values)
    num_total = len(seq)

    for i in range(num_total // batch_size):
        seq_b, len_seq_b, target_b = seq[i * batch_size: (i + 1)* batch_size], len_seq[i * batch_size: (i + 1)* batch_size], target[i * batch_size: (i + 1)* batch_size]
        states = torch.LongTensor(np.array(seq_b)).to(device)
        prediction = model.predict(states, np.array(len_seq_b), diff)
        _, topK = prediction.topk(100, dim=1, largest=True, sorted=True)
        topK = topK.cpu().detach().numpy()
        sorted_list2 = np.flip(topK, axis=1)
        calculate_hit(sorted_list2, topk, target_b, hit_purchase, ndcg_purchase)
        total_purchase += batch_size

    hr_list = []
    ndcg_list = []
    print('{:<10s} {:<10s} {:<10s} {:<10s} {:<10s} {:<10s}'.format('HR@'+str(topk[0]), 'NDCG@'+str(topk[0]), 'HR@'+str(topk[1]), 'NDCG@'+str(topk[1]), 'HR@'+str(topk[2]), 'NDCG@'+str(topk[2])))
    for i in range(len(topk)):
        hr_purchase = hit_purchase[i] / total_purchase
        ng_purchase = ndcg_purchase[i] / total_purchase
        hr_list.append(hr_purchase)
        ndcg_list.append(ng_purchase[0,0])
        if i == 1:
            hr_20 = hr_purchase
    print('{:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f}'.format(hr_list[0], ndcg_list[0], hr_list[1], ndcg_list[1], hr_list[2], ndcg_list[2]))
    return hr_20


DDBC_CAND_DIR = "/home/sjj/wenhao/DDBC_f-main/datasets/Yelp"


def _load_or_build_candidate_pool(labels_list, item_num, multiplier, predict_n, seed, cache_path):
    """
    为每个评估样本构建候选集，与 DDBC 逻辑完全一致。

    候选集 = unique(labels) 的 item IDs + multiplier×predict_n 个随机负样本
    总大小 = |unique_labels| + multiplier×predict_n
    （因 Yelp 10-item window 中 label 几乎不重复，实际 = predict_n + predict_n×multiplier）

    注意：item ID 空间两边均为 0-based，无需转换。
    """
    if os.path.exists(cache_path):
        print(f'[Candidate] Loading from {cache_path}')
        with open(cache_path, 'rb') as f:
            return pickle.load(f)['candidates']

    print(f'[Candidate] Building valid pool: predict_n={predict_n}, multiplier={multiplier}, '
          f'n_samples={len(labels_list)}')
    rng     = np.random.RandomState(seed)
    all_ids = np.arange(item_num)
    candidate_pool = []
    for label_list in labels_list:
        unique_labels = list(set(label_list))       # 与 DDBC 一致：去重后放入候选集
        mask          = np.ones(item_num, dtype=bool)
        for lid in unique_labels:
            mask[lid] = False
        n_random     = predict_n * multiplier       # 与 DDBC 一致：random 数量 = predict_n × mult
        random_items = rng.choice(all_ids[mask], size=n_random, replace=False).tolist()
        candidate_pool.append(unique_labels + random_items)

    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump({'metadata': {'seed': seed, 'multiplier': multiplier,
                                  'predict_n': predict_n, 'num_samples': len(labels_list)},
                     'candidates': candidate_pool}, f)
    print(f'[Candidate] Saved to {cache_path}')
    return candidate_pool


def _seq_mode_metrics(pred_items, label_list):
    """
    Sequence mode 指标（allow_duplicate_items=True），与 DDBC 评估逻辑完全一致。
    pred_items : list of predict_n item IDs（DreamRec 预测，无重复）
    label_list : list of predict_n item IDs（可能含重复）
    返回 dict: recall, precision, hit_1~5, jaccard
    """
    pred_counter  = Counter(pred_items)
    label_counter = Counter(label_list)
    all_keys      = set(pred_counter) | set(label_counter)

    # 交集（multiset）：sum(min(p, l))
    intersection = sum(min(pred_counter[k], label_counter[k]) for k in label_counter)
    # 并集（multiset）：sum(max(p, l))
    union        = sum(max(pred_counter[k], label_counter[k]) for k in all_keys)

    total_label = sum(label_counter.values())   # = predict_n（含重复）
    total_pred  = sum(pred_counter.values())    # = predict_n（无重复）

    recall    = intersection / total_label if total_label > 0 else 0.0
    precision = intersection / total_pred  if total_pred  > 0 else 0.0
    jaccard   = intersection / union       if union       > 0 else 0.0

    # hit_n：是否命中 ≥ n 个 label
    hits = {f'hit_{n}': (1 if intersection >= n else 0) for n in range(1, 6)}
    return {'recall': recall, 'precision': precision, **hits, 'jaccard': jaccard}


def evaluate_ddbc(model, diff, device,
                  predict_nums, multipliers, seed,
                  writer=None, epoch=None, split='test', predict_mode='single'):
    """
    DDBC 兼容评估（与 DDBC evaluator.py 逻辑对齐）。

    数据文件：{split}_data_items{predict_n}.df
      - seq     : [SEQ_SIZE] 历史序列（PAD=item_num）
      - len_seq : 有效历史长度
      - labels  : [predict_n] 个真实 label item（可能含重复）

    候选集：
      - test : 直接使用 DDBC 已生成的 test_candidates_seed1_x{mult}_items{n}.pkl
      - valid : 自动生成并缓存到 data/yelp/valid_candidates_seed{seed}_x{mult}_items{n}.pkl

    指标：recall, precision, hit_1~5, jaccard（sequence mode，Counter 计算，与 DDBC 一致）
    """
    batch_size  = 100
    all_results = {}

    for predict_n in predict_nums:
        # --- 加载评估数据 ---
        data_path = os.path.join(data_directory, f'{split}_data_items{predict_n}.df')
        eval_data = pd.read_pickle(data_path)
        seq       = list(eval_data['seq'].values)
        len_seq   = list(eval_data['len_seq'].values)
        labels    = list(eval_data['labels'].values)
        num_total = len(seq)

        for multiplier in multipliers:
            # --- 确定候选集文件路径 ---
            if split == 'test':
                # 直接复用 DDBC 已生成的测试候选集（seed=1，与 DDBC 完全一致）
                cand_path = os.path.join(
                    DDBC_CAND_DIR,
                    f'test_candidates_seed1_x{multiplier}_items{predict_n}.pkl'
                )
            else:
                # valid 候选集：自动生成并缓存
                cand_path = os.path.join(
                    data_directory,
                    f'valid_candidates_seed{seed}_x{multiplier}_items{predict_n}.pkl'
                )

            candidate_pool = _load_or_build_candidate_pool(
                labels, item_num, multiplier, predict_n, seed, cand_path
            )

            # --- 批量打分 ---
            metric_accum = {'recall': 0., 'precision': 0.,
                            'hit_1': 0., 'hit_2': 0., 'hit_3': 0.,
                            'hit_4': 0., 'hit_5': 0., 'jaccard': 0.}
            n_valid = 0

            model.eval()
            with torch.no_grad():
                n_batches = (num_total + batch_size - 1) // batch_size
                for bi in range(n_batches):
                    s          = slice(bi * batch_size, min((bi + 1) * batch_size, num_total))
                    states     = torch.LongTensor(np.array(seq[s])).to(device)
                    len_states = np.array(len_seq[s])
                    cands_b    = candidate_pool[s.start:s.stop]   # list of lists (variable length)

                    # Pad to uniform length within batch to handle variable-length candidate lists
                    max_cand_len = max(len(c) for c in cands_b)
                    padded_cands = [c + [0] * (max_cand_len - len(c)) for c in cands_b]
                    cand_ids   = torch.LongTensor(np.array(padded_cands)).to(device)

                    if predict_mode == 'ar':
                        # ── 自回归模式 ──────────────────────────────────────────
                        # 每步：扩散推理 → top-1 → 追加到历史 → 从候选集移除 → 下一步
                        batch_actual = s.stop - s.start
                        curr_seqs = np.array(seq[s.start:s.stop], dtype=np.int64).copy()   # [B, seq_size]
                        curr_lens = np.array(len_seq[s.start:s.stop], dtype=np.int64).copy()  # [B]
                        remaining_cands = [list(c) for c in cands_b]   # mutable per-sample lists
                        pred_items_batch = [[] for _ in range(batch_actual)]

                        for step in range(predict_n):
                            # 构造当前候选集 tensor（每步候选数比上一步少1）
                            max_cand_step = max(len(c) for c in remaining_cands)
                            padded_step = [c + [0] * (max_cand_step - len(c)) for c in remaining_cands]
                            cand_ids_step = torch.LongTensor(np.array(padded_step)).to(device)
                            states_step   = torch.LongTensor(curr_seqs).to(device)

                            scores_step = model.predict(
                                states_step, curr_lens, diff,
                                candidate_ids=cand_ids_step
                            ).detach().cpu().numpy()

                            # Mask padding
                            for j in range(batch_actual):
                                actual_len = len(remaining_cands[j])
                                if actual_len < max_cand_step:
                                    scores_step[j, actual_len:] = -np.inf

                            # top-1 per sample → update history & candidate pool
                            for j in range(batch_actual):
                                best_idx  = int(np.argmax(scores_step[j]))
                                best_item = remaining_cands[j][best_idx]
                                pred_items_batch[j].append(best_item)

                                # 从候选集移除（避免重复预测）
                                remaining_cands[j].pop(best_idx)

                                # 追加到历史序列
                                pos = int(curr_lens[j])
                                if pos < seq_size:
                                    curr_seqs[j, pos] = best_item
                                    curr_lens[j] += 1
                                else:
                                    # 历史已满：左移一位，末尾追加
                                    curr_seqs[j, :-1] = curr_seqs[j, 1:]
                                    curr_seqs[j, -1]  = best_item
                                    # curr_lens 保持 seq_size 不变

                        for j in range(batch_actual):
                            label_items = list(labels[s.start + j])
                            m = _seq_mode_metrics(pred_items_batch[j], label_items)
                            for k in metric_accum:
                                metric_accum[k] += m[k]
                            n_valid += 1

                    else:
                        # ── 单次推理模式（原始行为） ────────────────────────────
                        scores_np = model.predict(states, len_states, diff,
                                                  candidate_ids=cand_ids).detach().cpu().numpy()

                        # Mask padded positions so they won't be selected
                        for j in range(len(cands_b)):
                            actual_len = len(cands_b[j])
                            if actual_len < max_cand_len:
                                scores_np[j, actual_len:] = -np.inf

                        for j in range(len(cands_b)):
                            top_indices = np.argsort(scores_np[j])[::-1][:predict_n]
                            pred_items  = [cands_b[j][idx] for idx in top_indices]
                            label_items = list(labels[s.start + j])

                            m = _seq_mode_metrics(pred_items, label_items)
                            for k in metric_accum:
                                metric_accum[k] += m[k]
                            n_valid += 1
            model.train()

            # 均值
            result = {k: v / n_valid for k, v in metric_accum.items()}
            all_results[(predict_n, multiplier)] = result

    # --- 打印（与 DDBC output_results 格式对齐）---
    print(f'\n[{split.upper()} DDBC metrics]')
    for (predict_n, mult), m in sorted(all_results.items()):
        tag = f'@{predict_n}_x{mult}'
        print(f"  recall{tag}={m['recall']:.4f}  precision{tag}={m['precision']:.4f}  "
              f"hit_1{tag}={m['hit_1']:.4f}  hit_2{tag}={m['hit_2']:.4f}  "
              f"hit_3{tag}={m['hit_3']:.4f}  jaccard{tag}={m['jaccard']:.4f}")

    # --- TensorBoard ---
    if writer is not None and epoch is not None:
        for (predict_n, mult), m in all_results.items():
            prefix = f'{split}/x{mult}/top{predict_n}'
            for metric_name, val in m.items():
                writer.add_scalar(f'{prefix}/{metric_name}', val, epoch)

    # 主指标：smallest multiplier 下 predict_n=3 的 recall（用于保存最优模型）
    main_key = (predict_nums[0], multipliers[0])
    return all_results.get(main_key, {}).get('recall', 0.0)




if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)

    data_directory = './data/' + args.data
    data_statis = pd.read_pickle(
        os.path.join(data_directory, 'data_statis.df'))
    seq_size = data_statis['seq_size'][0]
    item_num = data_statis['item_num'][0]
    topk = [10, 20, 50]

    # --- 解析 DDBC 评估参数 ---
    predict_nums  = [int(x) for x in args.predict_nums.split(',')]
    multipliers   = [int(x) for x in args.candidate_multipliers.split(',')]
    eval_seed     = args.random_seed

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Tenc(args.hidden_factor, item_num, seq_size, args.dropout_rate, args.diffuser_type, device)
    diff  = diffusion(args.timesteps, args.beta_start, args.beta_end, args.w)

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-8, weight_decay=args.l2_decay)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, eps=1e-8, weight_decay=args.l2_decay)
    elif args.optimizer == 'adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr, eps=1e-8, weight_decay=args.l2_decay)
    elif args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, eps=1e-8, weight_decay=args.l2_decay)

    model.to(device)

    # --- TensorBoard ---
    descri = args.descri if args.descri else f'{args.data}-t{args.timesteps}-lr{args.lr}-w{args.w}'
    tb_log_dir = args.tb_log_dir if args.tb_log_dir else f'./tensorboard/{args.data}/{descri}'
    writer = SummaryWriter(log_dir=tb_log_dir)
    print(f'TensorBoard log dir: {tb_log_dir}')

    # --- 模型保存目录 ---
    save_dir = args.save_dir if args.save_dir else f'./outputs/{args.data}/{descri}'
    os.makedirs(save_dir, exist_ok=True)

    train_data = pd.read_pickle(os.path.join(data_directory, 'train_data.df'))

    best_val_recall = 0.0
    best_epoch = 0
    num_rows    = train_data.shape[0]
    num_batches = int(num_rows / args.batch_size)

    for i in range(args.epoch):
        start_time = Time.time()
        model.train()
        epoch_loss = 0.0
        for j in range(num_batches):
            batch  = train_data.sample(n=args.batch_size).to_dict()
            seq    = list(batch['seq'].values())
            len_seq = list(batch['len_seq'].values())
            target = list(batch['next'].values())

            optimizer.zero_grad()
            seq     = torch.LongTensor(seq).to(device)
            len_seq = torch.LongTensor(len_seq).to(device)
            target  = torch.LongTensor(target).to(device)

            x_start = model.cacu_x(target)
            h = model.cacu_h(seq, len_seq, args.p)
            n = torch.randint(0, args.timesteps, (args.batch_size,), device=device).long()
            loss, predicted_x = diff.p_losses(model, x_start, h, n, loss_type='l2')

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / num_batches
        writer.add_scalar('train/loss', avg_loss, i)

        if args.report_epoch:
            print("Epoch {:03d}; ".format(i) + 'Train loss: {:.4f}; '.format(avg_loss) +
                  "Time cost: " + Time.strftime("%H: %M: %S", Time.gmtime(Time.time() - start_time)))

        if (i + 1) % args.eval_freq == 0:
            eval_start = Time.time()

            print('-------------------------- VAL PHRASE --------------------------')
            val_recall = evaluate_ddbc(
                model, diff, device,
                predict_nums, multipliers, eval_seed,
                writer=writer, epoch=i, split='valid',
                predict_mode=args.predict_mode
            )

            print('-------------------------- TEST PHRASE -------------------------')
            evaluate_ddbc(
                model, diff, device,
                predict_nums, multipliers, eval_seed,
                writer=writer, epoch=i, split='test',
                predict_mode=args.predict_mode
            )

            print("Evaluation cost: " + Time.strftime("%H: %M: %S", Time.gmtime(Time.time() - eval_start)))
            print('----------------------------------------------------------------')

            # 保存最优模型
            if val_recall > best_val_recall:
                best_val_recall = val_recall
                best_epoch = i
                ckpt_path = os.path.join(save_dir, 'best_model.pt')
                torch.save({'epoch': i, 'model_state_dict': model.state_dict(),
                            'val_recall': val_recall}, ckpt_path)
                print(f'[Saved] Best model at epoch {i}, val_recall={val_recall:.4f} → {ckpt_path}')

    writer.close()
    print(f'\nTraining done. Best epoch={best_epoch}, val_recall@{predict_nums[0]}_x{multipliers[0]}={best_val_recall:.4f}')

    # 用最优 checkpoint 在 test 上跑完整 8 组结果
    ckpt_path = os.path.join(save_dir, 'best_model.pt')
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f'\n========== BEST MODEL TEST (epoch={ckpt["epoch"]}, '
              f'val_recall={ckpt["val_recall"]:.4f}) ==========')
        evaluate_ddbc(model, diff, device,
                      predict_nums, multipliers, eval_seed,
                      writer=None, epoch=None, split='test',
                      predict_mode=args.predict_mode)
        print('=' * 60)





                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     

