"""
AMP Classification with SOTA Model Comparison
==============================================
验证 SequenceMotifEncoder 特征提取的有效性，
对比近年 AMP 识别领域的代表性深度学习模型。

对比模型:
  ours           — SequenceMotifEncoder (多尺度卷积 + Transformer + VAE)
  deep_ampep30   — Deep-AmPEP30 (Li et al., 2020) 多通道 CNN
  amp_scanner    — AMP Scanner v2 (Veltri et al., 2018) CNN→LSTM
  amplify        — AMPlify (Li et al., 2022) BiLSTM + 多头注意力
  iamp_ca2l      — iAMP-CA2L (Xu et al., 2023) CNN-Attention 双通道融合
  amp_bert       — AMP-BERT (Lee et al., 2023) BERT-style Transformer
  sampred_gat    — sAMPpred-GAT (Yan et al., 2023) 图注意力网络

Usage:
  python compare_experiment.py --model ours --epochs 50
  python compare_experiment.py --model amp_bert --epochs 50
  python compare_experiment.py --model all --epochs 50
"""
import matplotlib
matplotlib.use('TkAgg')
import os, sys, argparse, time, json, warnings
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, matthews_corrcoef,
    confusion_matrix, classification_report, roc_curve,
)
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ════════════════════════════════════════════════════════════
# 0. 项目模块导入
# ════════════════════════════════════════════════════════════
try:
    from models.multi_scale_diffusion import MultiScaleConditionalDiffusion
    from models.layers import MultiScaleConv1D, TransformerBlock, SinusoidalTimeEmbedding
except ImportError:
    print("⚠️  无法导入项目模块，将使用内联定义")


# ════════════════════════════════════════════════════════════
# 1. 数据集
# ════════════════════════════════════════════════════════════
AA_VOCAB = {
    '<PAD>': 0, '<UNK>': 1,
    'A': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6,
    'G': 7, 'H': 8, 'I': 9, 'K': 10, 'L': 11,
    'M': 12, 'N': 13, 'P': 14, 'Q': 15, 'R': 16,
    'S': 17, 'T': 18, 'V': 19, 'W': 20, 'Y': 21,
}
VOCAB_SIZE = len(AA_VOCAB)


def encode_sequence(seq: str, max_len: int = 50) -> torch.LongTensor:
    tokens = [AA_VOCAB.get(aa, AA_VOCAB['<UNK>']) for aa in seq.upper()]
    tokens = tokens[:max_len]
    tokens += [AA_VOCAB['<PAD>']] * (max_len - len(tokens))
    return torch.LongTensor(tokens)


class AMPClassificationDataset(Dataset):
    def __init__(self, csv_path: str, max_len: int = 50):
        self.max_len = max_len
        self.df = pd.read_csv(csv_path)
        seq_col = [c for c in self.df.columns if c.lower() in ('seq', 'sequence', 'sequences')][0]
        label_col = [c for c in self.df.columns if c.lower() in ('label', 'labels', 'target', 'class')][0]
        self.sequences = self.df[seq_col].astype(str).tolist()
        self.labels = self.df[label_col].astype(int).tolist()
        valid = [(s, l) for s, l in zip(self.sequences, self.labels) if 0 < len(s) <= max_len * 2]
        self.sequences, self.labels = zip(*valid) if valid else ([], [])
        print(f"  📊 Loaded {len(self.sequences)} samples from {csv_path}")
        print(f"     Positive: {sum(self.labels)}, Negative: {len(self.labels) - sum(self.labels)}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]
        tokens = encode_sequence(seq, self.max_len)
        mask = (tokens != AA_VOCAB['<PAD>']).float()
        return {'tokens': tokens, 'mask': mask, 'label': torch.FloatTensor([label]), 'seq_str': seq}


# ════════════════════════════════════════════════════════════
# 2. 子模块（SequenceMotifEncoder 依赖）
# ════════════════════════════════════════════════════════════
class _MultiScaleConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[3, 5, 7]):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, k, padding=k // 2),
                nn.BatchNorm1d(out_channels), nn.GELU(),
            ) for k in kernel_sizes
        ])
        self.merge = nn.Conv1d(out_channels * len(kernel_sizes), out_channels, 1)

    def forward(self, x):
        return self.merge(torch.cat([c(x) for c in self.convs], dim=1))


class _TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim), nn.Dropout(dropout),
        )
        self.cond_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, cond=None, mask=None):
        if cond is not None and cond.dim() == 2:
            x = x + self.cond_proj(cond).unsqueeze(1)
        kpm = (mask == 0) if mask is not None else None
        attn_out, _ = self.attn(x, x, x, key_padding_mask=kpm)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x


# ════════════════════════════════════════════════════════════
# 3. SequenceMotifEncoder（我们的模型）
# ════════════════════════════════════════════════════════════
class _SequenceMotifEncoder(nn.Module):
    def __init__(self, vocab_size=22, max_len=50, latent_dim=128,
                 hidden_dim=256, num_heads=8, num_layers=4,
                 kernel_sizes=[3, 5, 7], dropout=0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.aa_embed = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.pos_embed = nn.Embedding(max_len, hidden_dim)
        self.motif_conv = _MultiScaleConv1D(hidden_dim, hidden_dim, kernel_sizes)
        self.transformer_blocks = nn.ModuleList([
            _TransformerBlock(hidden_dim, num_heads, hidden_dim, dropout) for _ in range(num_layers)
        ])
        self.to_latent = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, latent_dim * 2))

    def forward(self, seq_tokens, mask=None):
        B, L = seq_tokens.shape
        positions = torch.arange(L, device=seq_tokens.device).unsqueeze(0)
        x = self.aa_embed(seq_tokens) + self.pos_embed(positions)
        x = x + self.motif_conv(x.permute(0, 2, 1)).permute(0, 2, 1)
        cond = torch.zeros(B, x.shape[-1], device=x.device)
        for block in self.transformer_blocks:
            x = block(x, cond, mask)
        stats = self.to_latent(x)
        z_mean, z_logvar = stats.chunk(2, dim=-1)
        return z_mean, z_logvar

try:
    from models.encoders import SequenceMotifEncoder as _ProjectEncoder
    SequenceMotifEncoder = _ProjectEncoder
except ImportError:
    SequenceMotifEncoder = _SequenceMotifEncoder


# ════════════════════════════════════════════════════════════
# 4. SOTA AMP 识别模型编码器
# ════════════════════════════════════════════════════════════

# ──────── 4.1 Deep-AmPEP30 (Li et al., 2020) ────────
class DeepAmPEP30Encoder(nn.Module):
    """
    Deep-AmPEP30 — 多通道 CNN
    ──────────────────────────────────────────────
    Ref: Li et al. "Deep-AmPEP30: Improve Short Antimicrobial Peptides
         Prediction with Deep Learning", Mol. Ther. Nucleic Acids, 2020.
    核心思想: 使用不同大小的卷积核（3, 5, 9, 15）并行提取多尺度局部模式，
    然后拼接融合。原文使用二进制编码特征，此处改用可学习 Embedding。
    """

    def __init__(self, vocab_size, max_len, hidden_dim=256, dropout=0.1):
        super().__init__()
        embed_dim = 128
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        kernel_sizes = [3, 5, 9, 15]
        n_per = hidden_dim // len(kernel_sizes)  # 每个通道的滤波器数
        self.conv_channels = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(embed_dim, n_per, k, padding=k // 2),
                nn.BatchNorm1d(n_per), nn.ReLU(),
                nn.Conv1d(n_per, n_per, k, padding=k // 2),
                nn.BatchNorm1d(n_per), nn.ReLU(),
            ) for k in kernel_sizes
        ])
        cnn_total = n_per * len(kernel_sizes)
        self.merge = nn.Sequential(
            nn.Conv1d(cnn_total, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim), nn.ReLU(),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens, mask=None):
        L = tokens.size(1)
        x = self.embed(tokens).permute(0, 2, 1)          # (B, C, L)
        outs = [conv(x)[:, :, :L] for conv in self.conv_channels]
        x = self.merge(torch.cat(outs, dim=1))            # (B, D, L)
        return self.dropout(x.permute(0, 2, 1))           # (B, L, D)


# ──────── 4.2 AMP Scanner v2 (Veltri et al., 2018) ────────
class AMPScannerEncoder(nn.Module):
    """
    AMP Scanner v2 — CNN → BiLSTM 串行架构
    ──────────────────────────────────────────────
    Ref: Veltri et al. "Deep learning improves antimicrobial peptide
         recognition", Bioinformatics, 2018.
    核心思想: 先用一组卷积层提取局部 motif 特征，再用 BiLSTM 建模序列依赖。
    经典的 AMP 预测 baseline。
    """

    def __init__(self, vocab_size, max_len, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.conv_block = nn.Sequential(
            nn.Conv1d(hidden_dim, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, hidden_dim, 7, padding=3), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            hidden_dim, hidden_dim // 2, num_layers=2,
            batch_first=True, bidirectional=True, dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens, mask=None):
        x = self.embed(tokens).permute(0, 2, 1)
        x = self.conv_block(x).permute(0, 2, 1)          # (B, L, D)
        if mask is not None:
            lengths = mask.sum(1).long().cpu().clamp(min=1)
            packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
            output, _ = self.lstm(packed)
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True, total_length=tokens.size(1))
        else:
            output, _ = self.lstm(x)
        return self.dropout(output)                       # (B, L, D)


# ──────── 4.3 AMPlify (Li et al., 2022) ────────
class AMPlifyEncoder(nn.Module):
    """
    AMPlify — BiLSTM + 多头自注意力
    ──────────────────────────────────────────────
    Ref: Li et al. "AMPlify: attentive deep learning model for discovery
         of novel antimicrobial peptides", BMC Genomics, 2022.
    核心思想: 3 层 BiLSTM 捕获双向上下文，再用多头自注意力加权关键位点，
    是多个 benchmark 上的 top-performing 模型之一。
    """

    def __init__(self, vocab_size, max_len, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            hidden_dim, hidden_dim, num_layers=3,
            batch_first=True, bidirectional=True, dropout=dropout,
        )
        lstm_out = hidden_dim * 2
        self.attention = nn.MultiheadAttention(
            lstm_out, num_heads=8, dropout=dropout, batch_first=True,
        )
        self.norm = nn.LayerNorm(lstm_out)
        self.ff = nn.Sequential(
            nn.Linear(lstm_out, lstm_out), nn.GELU(), nn.Dropout(dropout),
        )
        self.ff_norm = nn.LayerNorm(lstm_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens, mask=None):
        x = self.embed(tokens)
        if mask is not None:
            lengths = mask.sum(1).long().cpu().clamp(min=1)
            packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
            lstm_out, _ = self.lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True, total_length=tokens.size(1))
        else:
            lstm_out, _ = self.lstm(x)
        kpm = (mask == 0) if mask is not None else None
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out, key_padding_mask=kpm)
        x = self.norm(lstm_out + attn_out)
        x = self.ff_norm(x + self.ff(x))
        return self.dropout(x)                            # (B, L, hidden_dim*2)


# ──────── 4.4 iAMP-CA2L (Xu et al., 2023) ────────
class iAMPCA2LEncoder(nn.Module):
    """
    iAMP-CA2L — CNN-Attention 双通道 + 交叉注意力融合
    ──────────────────────────────────────────────
    Ref: Xu et al. "iAMP-CA2L: a new CNN-attention-based method for
         identifying antimicrobial peptides", 2023.
    核心思想:
      Channel 1: 多尺度 CNN 提取局部 motif
      Channel 2: 多头自注意力捕获全局依赖
      交叉注意力 + 拼接融合两通道特征
    """

    def __init__(self, vocab_size, max_len, hidden_dim=256, num_heads=8, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.pos_embed = nn.Embedding(max_len, hidden_dim)

        # ── Channel 1: Multi-scale CNN ──
        n_per = hidden_dim // 3
        cnn_total = n_per * 3
        self.cnn_branch = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(hidden_dim, n_per, k, padding=k // 2),
                nn.BatchNorm1d(n_per), nn.GELU(),
            ) for k in [3, 5, 7]
        ])
        self.cnn_proj = nn.Sequential(
            nn.Conv1d(cnn_total, hidden_dim, 1), nn.BatchNorm1d(hidden_dim), nn.GELU(),
        )

        # ── Channel 2: Self-Attention ──
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.attn_ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.ff_norm = nn.LayerNorm(hidden_dim)

        # ── Cross-channel Attention ──
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads // 2, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(hidden_dim)

        # ── Fusion ──
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.GELU(), nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens, mask=None):
        B, L = tokens.shape
        pos = torch.arange(L, device=tokens.device).unsqueeze(0)
        x = self.embed(tokens) + self.pos_embed(pos)
        kpm = (mask == 0) if mask is not None else None

        # CNN branch
        xc = x.permute(0, 2, 1)
        xc = torch.cat([conv(xc)[:, :, :L] for conv in self.cnn_branch], dim=1)
        x_cnn = self.cnn_proj(xc).permute(0, 2, 1)       # (B, L, D)

        # Attention branch
        ao, _ = self.self_attn(x, x, x, key_padding_mask=kpm)
        x_attn = self.attn_norm(x + ao)
        x_attn = self.ff_norm(x_attn + self.attn_ff(x_attn))

        # Cross-channel
        co, _ = self.cross_attn(x_cnn, x_attn, x_attn, key_padding_mask=kpm)
        x_cnn = self.cross_norm(x_cnn + co)

        return self.dropout(self.fusion(torch.cat([x_cnn, x_attn], -1)))  # (B, L, D)


# ──────── 4.5 AMP-BERT (Lee et al., 2023) ────────
class AMPBERTEncoder(nn.Module):
    """
    AMP-BERT — BERT-style 深层 Transformer
    ──────────────────────────────────────────────
    Ref: Lee et al. "AMP-BERT: Prediction of antimicrobial peptide
         function based on a BERT model", IJMS, 2023.
    核心思想: 使用 6 层 Pre-Norm Transformer Encoder，
    配合 LayerNorm Embedding 和 GELU 激活（BERT 范式）。
    注: 原文使用蛋白质语言模型预训练权重；此处从零训练以公平对比。
    """

    def __init__(self, vocab_size, max_len, hidden_dim=256,
                 num_heads=8, num_layers=6, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.pos_embed = nn.Embedding(max_len, hidden_dim)
        self.embed_norm = nn.LayerNorm(hidden_dim)
        self.embed_drop = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4, dropout=dropout,
            batch_first=True, activation='gelu', norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, norm=nn.LayerNorm(hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens, mask=None):
        B, L = tokens.shape
        pos = torch.arange(L, device=tokens.device).unsqueeze(0)
        x = self.embed_drop(self.embed_norm(self.embed(tokens) + self.pos_embed(pos)))
        kpm = (mask == 0) if mask is not None else None
        return self.dropout(self.transformer(x, src_key_padding_mask=kpm))  # (B, L, D)


# ──────── 4.6 sAMPpred-GAT (Yan et al., 2023) ────────

class _GATHead(nn.Module):
    """单头图注意力（Veličković et al., ICLR 2018 风格）"""

    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a_src = nn.Linear(out_dim, 1, bias=False)
        self.a_dst = nn.Linear(out_dim, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """x: (B, N, in_dim), mask: (B, N)"""
        B, N, _ = x.shape
        h = self.W(x)                                    # (B, N, out_dim)
        e = self.leaky_relu(
            self.a_src(h) + self.a_dst(h).transpose(1, 2) # (B, N, N)
        )
        if mask is not None:
            e = e.masked_fill(mask.unsqueeze(1).expand(-1, N, -1) == 0, float('-inf'))
        alpha = self.dropout(F.softmax(e, dim=-1))
        alpha = alpha.masked_fill(torch.isnan(alpha), 0.0)
        return torch.bmm(alpha, h)                        # (B, N, out_dim)


class _MultiHeadGAT(nn.Module):
    """多头 GAT（拼接模式）"""

    def __init__(self, in_dim, out_dim, num_heads=4, dropout=0.1):
        super().__init__()
        assert out_dim % num_heads == 0
        self.heads = nn.ModuleList([
            _GATHead(in_dim, out_dim // num_heads, dropout)
            for _ in range(num_heads)
        ])

    def forward(self, x, mask=None):
        return torch.cat([h(x, mask) for h in self.heads], dim=-1)


class sAMPpredGATEncoder(nn.Module):
    """
    sAMPpred-GAT — 图注意力网络
    ──────────────────────────────────────────────
    Ref: Yan et al. "sAMPpred-GAT: prediction of antimicrobial peptide
         by graph attention network", J. Chem. Inf. Model., 2023.
    核心思想: 将肽段建模为全连接图（每个残基为节点），
    利用多头图注意力机制学习残基间的重要交互关系。
    使用 ELU 激活 + 残差连接 + LayerNorm。
    """

    def __init__(self, vocab_size, max_len, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.pos_embed = nn.Embedding(max_len, hidden_dim)

        num_heads, n_layers = 4, 3
        self.gat_layers = nn.ModuleList([
            _MultiHeadGAT(hidden_dim, hidden_dim, num_heads, dropout)
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_layers)])
        self.ffs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2), nn.ELU(),
                nn.Dropout(dropout), nn.Linear(hidden_dim * 2, hidden_dim),
            ) for _ in range(n_layers)
        ])
        self.ff_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens, mask=None):
        B, L = tokens.shape
        pos = torch.arange(L, device=tokens.device).unsqueeze(0)
        x = self.embed(tokens) + self.pos_embed(pos)
        for gat, norm, ff, fn in zip(self.gat_layers, self.norms, self.ffs, self.ff_norms):
            x = norm(x + gat(x, mask))
            x = fn(x + ff(x))
        return self.dropout(x)                            # (B, L, D)


# ════════════════════════════════════════════════════════════
# 5. 注意力池化
# ════════════════════════════════════════════════════════════
class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x, mask=None):
        scores = self.attention(x).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        weights = F.softmax(scores, dim=-1).unsqueeze(-1)
        return (x * weights).sum(dim=1)


# ════════════════════════════════════════════════════════════
# 6. 统一分类器框架
# ════════════════════════════════════════════════════════════
class ClassifierHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        return self.head(x)


class AMPClassifier(nn.Module):
    """
    统一 AMP 分类器: encoder_name 切换不同特征提取器。
    """

    def __init__(self, encoder_name="ours", vocab_size=VOCAB_SIZE,
                 max_len=50, latent_dim=128, hidden_dim=256,
                 num_heads=8, num_layers=4, dropout=0.1,
                 pretrained_encoder_path=None, freeze_encoder=False):
        super().__init__()
        self.encoder_name = encoder_name
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.encoder, self.encoder_output_dim = self._build_encoder(
            encoder_name, vocab_size, max_len, latent_dim,
            hidden_dim, num_heads, num_layers, dropout,
        )
        self.pool = AttentionPooling(self.encoder_output_dim)
        self.classifier = ClassifierHead(self.encoder_output_dim, 128, dropout)

        if pretrained_encoder_path:
            self._load_pretrained_encoder(pretrained_encoder_path)
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
            print("  ❄️  Encoder frozen")

    # ──────────────────────────────────────────
    def _build_encoder(self, name, vs, ml, ld, hd, nh, nl, dp):
        """根据名称构建编码器并返回 (encoder, output_dim)"""

        if name in ("ours", "ours_frozen"):
            enc = SequenceMotifEncoder(
                vocab_size=vs, max_len=ml, latent_dim=ld,
                hidden_dim=hd, num_heads=nh, num_layers=nl,
                kernel_sizes=[3, 5, 7], dropout=dp,
            )
            return enc, ld

        elif name == "deep_ampep30":
            return DeepAmPEP30Encoder(vs, ml, hd, dp), hd

        elif name == "amp_scanner":
            return AMPScannerEncoder(vs, ml, hd, dp), hd

        elif name == "amplify":
            return AMPlifyEncoder(vs, ml, hd, dp), hd * 2

        elif name == "iamp_ca2l":
            return iAMPCA2LEncoder(vs, ml, hd, nh, dp), hd

        elif name == "amp_bert":
            return AMPBERTEncoder(vs, ml, hd, nh, num_layers=6, dropout=dp), hd

        elif name == "sampred_gat":
            return sAMPpredGATEncoder(vs, ml, hd, dp), hd

        else:
            raise ValueError(f"Unknown encoder: {name}")

    # ──────────────────────────────────────────
    def _load_pretrained_encoder(self, path):
        print(f"  📥 Loading pretrained encoder from {path}")
        ckpt = torch.load(path, map_location="cpu")
        sd = ckpt.get("ema_shadow", ckpt.get("model_state_dict", ckpt))
        enc_sd = {k.replace("encoder_l1.", ""): v for k, v in sd.items() if k.startswith("encoder_l1.")}
        if enc_sd:
            m, u = self.encoder.load_state_dict(enc_sd, strict=False)
            print(f"    Loaded {len(enc_sd)} params  missing={len(m)} unexpected={len(u)}")
        else:
            print("    ⚠️ No encoder_l1 weights found")

    # ──────────────────────────────────────────
    def forward(self, tokens, mask=None):
        if self.encoder_name in ("ours", "ours_frozen"):
            z_mean, _ = self.encoder(tokens, mask)
            features = z_mean
        else:
            features = self.encoder(tokens, mask)
        return self.classifier(self.pool(features, mask))

    def extract_features(self, tokens, mask=None):
        with torch.no_grad():
            if self.encoder_name in ("ours", "ours_frozen"):
                z_mean, _ = self.encoder(tokens, mask)
                features = z_mean
            else:
                features = self.encoder(tokens, mask)
            return self.pool(features, mask)


# ════════════════════════════════════════════════════════════
# 7. 训练与评估引擎
# ════════════════════════════════════════════════════════════
class Trainer:
    def __init__(self, model, train_loader, val_loader, device,
                 lr=1e-4, weight_decay=1e-5, epochs=50,
                 patience=10, output_dir="results", model_name="model"):
        self.model = model.to(device)
        self.device = device
        self.epochs = epochs
        self.patience = patience
        self.model_name = model_name
        self.output_dir = Path(output_dir) / model_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        n_pos = sum(1 for b in train_loader for l in b['label'] if l.item() > .5)
        n_neg = len(train_loader.dataset) - n_pos
        pw = torch.FloatTensor([n_neg / max(n_pos, 1)]).to(device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
        print(f"  ⚖️ Pos weight: {pw.item():.2f} (pos={n_pos}, neg={n_neg})")

        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, weight_decay=weight_decay,
        )
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=lr * 0.01)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.history = {k: [] for k in ('train_loss', 'val_loss', 'val_acc', 'val_f1', 'val_auc', 'val_mcc')}
        self.best_val_f1 = 0.0
        self.patience_counter = 0

    def train_epoch(self):
        self.model.train()
        total, n = 0.0, 0
        for batch in self.train_loader:
            tok = batch['tokens'].to(self.device)
            msk = batch['mask'].to(self.device)
            lbl = batch['label'].to(self.device)
            logits = self.model(tok, msk)
            loss = self.criterion(logits, lbl)
            self.optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total += loss.item(); n += 1
        return total / max(n, 1)

    @torch.no_grad()
    def evaluate(self, loader=None):
        loader = loader or self.val_loader
        self.model.eval()
        all_lg, all_lb = [], []
        total, n = 0.0, 0
        for batch in loader:
            tok = batch['tokens'].to(self.device)
            msk = batch['mask'].to(self.device)
            lbl = batch['label'].to(self.device)
            lg = self.model(tok, msk)
            total += self.criterion(lg, lbl).item(); n += 1
            all_lg.append(lg.cpu()); all_lb.append(lbl.cpu())
        all_lg = torch.cat(all_lg).squeeze(-1).numpy()
        all_lb = torch.cat(all_lb).squeeze(-1).numpy()
        probs = 1 / (1 + np.exp(-all_lg))
        preds = (probs >= .5).astype(int)
        uniq = len(np.unique(all_lb)) > 1
        return {
            'loss': total / max(n, 1),
            'accuracy':  accuracy_score(all_lb, preds),
            'precision': precision_score(all_lb, preds, zero_division=0),
            'recall':    recall_score(all_lb, preds, zero_division=0),
            'f1':        f1_score(all_lb, preds, zero_division=0),
            'mcc':       matthews_corrcoef(all_lb, preds),
            'auc_roc':   roc_auc_score(all_lb, probs) if uniq else 0,
            'auc_pr':    average_precision_score(all_lb, probs) if uniq else 0,
        }, all_lb, probs, preds

    def train(self):
        total_p  = sum(p.numel() for p in self.model.parameters())
        train_p  = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"\n{'='*60}\n  🚀 {self.model_name}  |  Encoder: {self.model.encoder_name}")
        print(f"  🔢 Params: {total_p:,}  Trainable: {train_p:,}\n{'='*60}\n")
        t0 = time.time()

        for ep in range(1, self.epochs + 1):
            tl = self.train_epoch()
            vm, _, _, _ = self.evaluate()
            self.scheduler.step()

            self.history['train_loss'].append(tl)
            self.history['val_loss'].append(vm['loss'])
            self.history['val_acc'].append(vm['accuracy'])
            self.history['val_f1'].append(vm['f1'])
            self.history['val_auc'].append(vm['auc_roc'])
            self.history['val_mcc'].append(vm['mcc'])

            if ep % 5 == 0 or ep == 1:
                print(f"  Ep {ep:3d}/{self.epochs} | TrL {tl:.4f} | "
                      f"VL {vm['loss']:.4f} Acc {vm['accuracy']:.4f} "
                      f"F1 {vm['f1']:.4f} AUC {vm['auc_roc']:.4f} MCC {vm['mcc']:.4f}")

            if vm['f1'] > self.best_val_f1:
                self.best_val_f1 = vm['f1']; self.patience_counter = 0
                torch.save({'model_state_dict': self.model.state_dict(),
                            'encoder_name': self.model.encoder_name,
                            'best_f1': self.best_val_f1, 'epoch': ep,
                            'history': self.history},
                           self.output_dir / 'best_model.pt')
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f"\n  ⏹ Early stopping at epoch {ep}"); break

        elapsed = time.time() - t0
        print(f"\n  ⏱ Training time: {elapsed:.1f}s")
        best = torch.load(self.output_dir / 'best_model.pt', map_location=self.device)
        self.model.load_state_dict(best['model_state_dict'])
        fm, lb, pr, pd = self.evaluate()

        print(f"\n  🏆 Best Validation ({self.model_name}):")
        for k, v in fm.items():
            print(f"     {k}: {v:.4f}")

        fm['n_params'] = total_p
        fm['n_trainable'] = train_p
        fm['training_time'] = elapsed
        self._save_results(fm, lb, pr, pd, elapsed)
        return fm

    def _save_results(self, metrics, labels, probs, preds, elapsed):
        # JSON
        res = {'model': self.model_name, 'encoder': self.model.encoder_name,
               'metrics': {k: float(v) for k, v in metrics.items()}}
        with open(self.output_dir / 'results.json', 'w') as f:
            json.dump(res, f, indent=2)

        # 训练曲线
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        er = range(1, len(self.history['train_loss']) + 1)
        axes[0].plot(er, self.history['train_loss'], label='Train', c='#E74C3C')
        axes[0].plot(er, self.history['val_loss'], label='Val', c='#3498DB')
        axes[0].set_title('Loss'); axes[0].legend()
        axes[1].plot(er, self.history['val_acc'], label='Acc', c='#2ECC71')
        axes[1].plot(er, self.history['val_f1'], label='F1', c='#E67E22')
        axes[1].set_title('Metrics'); axes[1].legend()
        axes[2].plot(er, self.history['val_auc'], label='AUC', c='#9B59B6')
        axes[2].plot(er, self.history['val_mcc'], label='MCC', c='#1ABC9C')
        axes[2].set_title('AUC & MCC'); axes[2].legend()
        fig.suptitle(f'{self.model_name}', fontsize=14)
        plt.tight_layout(); plt.savefig(self.output_dir / 'curves.png', dpi=150); plt.close()

        # ROC
        if len(np.unique(labels)) > 1:
            fpr, tpr, _ = roc_curve(labels, probs)
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.plot(fpr, tpr, c='#E74C3C', lw=2, label=f'AUC={metrics["auc_roc"]:.4f}')
            ax.plot([0, 1], [0, 1], 'k--'); ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
            ax.set_title(f'{self.model_name} ROC'); ax.legend()
            plt.savefig(self.output_dir / 'roc.png', dpi=150); plt.close()

        # 混淆矩阵
        cm = confusion_matrix(labels, preds)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Non-AMP', 'AMP'], yticklabels=['Non-AMP', 'AMP'])
        ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
        plt.savefig(self.output_dir / 'cm.png', dpi=150); plt.close()

        with open(self.output_dir / 'report.txt', 'w') as f:
            f.write(classification_report(labels, preds, target_names=['Non-AMP', 'AMP']))


# ════════════════════════════════════════════════════════════
# 8. 对比实验汇总
# ════════════════════════════════════════════════════════════
ALL_MODELS = [
    'ours', 'deep_ampep30', 'amp_scanner',
    'amplify', 'iamp_ca2l', 'amp_bert', 'sampred_gat',
]

MODEL_DISPLAY = {
    'ours':          'Ours (MotifEncoder)',
    'ours_frozen':   'Ours (frozen)',
    'deep_ampep30':  'Deep-AmPEP30',
    'amp_scanner':   'AMP Scanner v2',
    'amplify':       'AMPlify',
    'iamp_ca2l':     'iAMP-CA2L',
    'amp_bert':      'AMP-BERT',
    'sampred_gat':   'sAMPpred-GAT',
}


def run_comparison(train_path, val_path, models_to_run, pretrained_path=None,
                   max_len=50, batch_size=64, epochs=50, lr=1e-4,
                   device="cuda", output_dir="results/classification"):
    train_ds = AMPClassificationDataset(train_path, max_len)
    val_ds   = AMPClassificationDataset(val_path, max_len)
    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_ld   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    all_results = {}

    for model_name in models_to_run:
        print(f"\n{'#'*60}\n#  Model: {MODEL_DISPLAY.get(model_name, model_name)}\n{'#'*60}")

        # 处理 frozen 变体
        if model_name == "ours_frozen":
            enc_name, pre_path, freeze = "ours", pretrained_path, True
        else:
            enc_name = model_name
            pre_path = pretrained_path if model_name == "ours" else None
            freeze = False

        model = AMPClassifier(
            encoder_name=enc_name, vocab_size=VOCAB_SIZE, max_len=max_len,
            latent_dim=128, hidden_dim=256, num_heads=8, num_layers=4,
            dropout=0.1, pretrained_encoder_path=pre_path, freeze_encoder=freeze,
        )
        trainer = Trainer(model, train_ld, val_ld, device, lr=lr, epochs=epochs,
                          patience=15, output_dir=output_dir, model_name=model_name)
        all_results[model_name] = trainer.train()

    _generate_comparison_table(all_results, output_dir)
    _generate_comparison_plots(all_results, output_dir)
    return all_results


def _generate_comparison_table(all_results, output_dir):
    rows = []
    for name, m in all_results.items():
        rows.append({
            'Model': MODEL_DISPLAY.get(name, name),
            'Params': f"{m.get('n_params', 0):,}",
            'Acc': f"{m['accuracy']:.4f}",
            'Prec': f"{m['precision']:.4f}",
            'Rec': f"{m['recall']:.4f}",
            'F1': f"{m['f1']:.4f}",
            'AUC': f"{m['auc_roc']:.4f}",
            'AP': f"{m['auc_pr']:.4f}",
            'MCC': f"{m['mcc']:.4f}",
            'Time(s)': f"{m.get('training_time', 0):.0f}",
        })
    df = pd.DataFrame(rows)
    print(f"\n{'='*100}\n📊 COMPARISON RESULTS\n{'='*100}")
    print(df.to_string(index=False))
    print(f"{'='*100}\n")
    out = Path(output_dir)
    df.to_csv(out / "comparison.csv", index=False)
    with open(out / "comparison.tex", 'w') as f:
        f.write(df.to_latex(index=False, caption="AMP Classification — SOTA Comparison",
                            label="tab:amp_sota"))
    print(f"  💾 Saved → {out / 'comparison.csv'}")


def _generate_comparison_plots(all_results, output_dir):
    mkeys  = ['accuracy', 'f1', 'auc_roc', 'mcc']
    mlabel = ['Accuracy', 'F1 Score', 'AUC-ROC', 'MCC']
    models = list(all_results.keys())
    disp   = [MODEL_DISPLAY.get(m, m) for m in models]
    colors = sns.color_palette("Set2", len(models))

    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    for ax, mk, ml in zip(axes, mkeys, mlabel):
        vals = [all_results[m][mk] for m in models]
        bars = ax.bar(range(len(models)), vals, color=colors, edgecolor='white', width=0.6)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + .005,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(disp, rotation=35, ha='right', fontsize=8)
        ax.set_title(ml, fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1.08); ax.grid(axis='y', alpha=.3)
    fig.suptitle('AMP Classification — SOTA Model Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "comparison_bar.png", dpi=200, bbox_inches='tight'); plt.close()

    # 雷达图
    rkeys = ['accuracy', 'precision', 'recall', 'f1', 'auc_roc', 'mcc']
    rlabel = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC-ROC', 'MCC']
    angles = np.linspace(0, 2 * np.pi, len(rkeys), endpoint=False).tolist() + [0]
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for (name, met), c in zip(all_results.items(), sns.color_palette("husl", len(all_results))):
        vals = [met[k] for k in rkeys] + [met[rkeys[0]]]
        ax.plot(angles, vals, 'o-', lw=2, label=MODEL_DISPLAY.get(name, name), color=c)
        ax.fill(angles, vals, alpha=.08, color=c)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(rlabel, fontsize=10)
    ax.set_ylim(0, 1); ax.set_title('Radar Chart', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=8)
    plt.savefig(Path(output_dir) / "comparison_radar.png", dpi=200, bbox_inches='tight'); plt.close()
    print(f"  📊 Plots saved → {output_dir}/")


# ════════════════════════════════════════════════════════════
# 9. 主入口
# ════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description='AMP Classification — SOTA Comparison')
    parser.add_argument('--train_data', type=str, default='data/classification_training_data.csv')
    parser.add_argument('--val_data',   type=str, default='data/classification_val_data.csv')
    parser.add_argument('--max_len',    type=int, default=50)
    parser.add_argument('--model', type=str, default='all',
                        choices=['ours', 'ours_frozen',
                                 'deep_ampep30', 'amp_scanner', 'amplify',
                                 'iamp_ca2l', 'amp_bert', 'sampred_gat',
                                 'all'],
                        help='Model to train, or "all" for full comparison.')
    parser.add_argument('--pretrained', type=str, default=None)
    parser.add_argument('--epochs',     type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr',         type=float, default=1e-4)
    parser.add_argument('--output_dir', type=str, default='results/classification')
    parser.add_argument('--device',     type=str, default='cuda')
    args = parser.parse_args()

    if args.model == 'all':
        models_to_run = [
            'ours',  # 我们的方法
            'deep_ampep30',
            'amp_scanner',
            'amp_scanner',
            'iamp_ca2l',
            'amp_bert',
            'sampred_gat',

        ]
        if args.pretrained:
            models_to_run.insert(1, 'ours_frozen')
    else:
        models_to_run = [args.model]

    print(f"""
    ╔══════════════════════════════════════════════════════════╗
    ║     AMP Classification — SOTA Model Benchmark           ║
    ╠══════════════════════════════════════════════════════════╣
    ║  Models:                                                ║""")
    for m in models_to_run:
        print(f"║    • {m:<50s} ║")
    print(f"""║  Train data:  {args.train_data:<40s} ║
    ║  Val data:    {args.val_data:<40s} ║
    ║  Max length:  {args.max_len:<40d} ║
    ║  Epochs:      {args.epochs:<40d} ║
    ║  Batch size:  {args.batch_size:<40d} ║
    ║  Device:      {args.device:<40s} ║
    ╚══════════════════════════════════════════════════════════╝
        """)

    all_results = run_comparison(
        train_path=args.train_data, val_path=args.val_data,
        models_to_run=models_to_run, pretrained_path=args.pretrained,
        max_len=args.max_len, batch_size=args.batch_size,
        epochs=args.epochs, lr=args.lr,
        device=args.device, output_dir=args.output_dir,
    )

    print("\n🎉 All experiments completed!")
    print(f"   Results saved to: {args.output_dir}/")


if __name__ == '__main__':
    main()
