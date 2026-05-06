"""

对比实验组：
  1. Seq-Only:        只用序列层 (Layer 1)
  2. Seq+Struct:      序列 + 二级结构层 (Layer 1 + 2)
  3. Seq+Property:    序列 + 理化性质层 (Layer 1 + 3)
  4. Full-Model:      完整三层模型 (Layer 1 + 2 + 3)

评估指标：
  - Sequence Reconstruction Loss (内部验证)
  - Novelty (多样性验证)
  - AMP Probability (功能验证，使用SOTA分类器)

Usage:
  python ablation_study_sota.py \
      --checkpoint checkpoints/best_model.pt \
      --data_path data/amp_extended_dataset_v2.csv \
      --output_dir results/ablation \
      --num_samples 500 \
      --batch_size 64
"""

import os
import argparse
import json
import warnings
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import matplotlib

matplotlib.use('TkAgg')
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from config import load_config, FullConfig
from data.dataset import (
    AminoAcidTokenizer,
    SecondaryStructureEncoder,
    PhysicochemicalCalculator,
    AMPDataset,
)
from models.multi_scale_diffusion import MultiScaleConditionalDiffusion

warnings.filterwarnings("ignore", category=UserWarning)

# ======================== 美学设置 ========================
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})
sns.set_style("whitegrid")


# ======================== SOTA AMP分类器模块 ========================

class ProteinEmbedding(nn.Module):
    """
    多尺度蛋白质嵌入
    结合: 位置编码 + 氨基酸理化性质 + 学习嵌入
    """

    def __init__(self, vocab_size=25, embed_dim=128, max_len=100):
        super().__init__()

        # 学习的嵌入
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # 位置编码 (Sinusoidal)
        self.register_buffer(
            'position_encoding',
            self._get_sinusoidal_encoding(max_len, embed_dim)
        )

        # 氨基酸理化性质嵌入
        self.physchem_embedding = nn.Embedding(vocab_size, 64, padding_idx=0)

        # 融合层
        self.fusion = nn.Linear(embed_dim + 64, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def _get_sinusoidal_encoding(self, max_len, d_model):
        """正弦位置编码"""
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len]
        Returns:
            embeddings: [batch_size, seq_len, embed_dim]
        """
        B, L = x.shape

        # Token embedding
        token_emb = self.token_embedding(x)  # [B, L, embed_dim]

        # 添加位置编码
        pos_enc = self.position_encoding[:L, :].unsqueeze(0)  # [1, L, embed_dim]
        token_emb = token_emb + pos_enc

        # 理化性质嵌入
        physchem_emb = self.physchem_embedding(x)  # [B, L, 64]

        # 融合
        combined = torch.cat([token_emb, physchem_emb], dim=-1)
        fused = self.fusion(combined)

        return self.layer_norm(fused)


class MultiHeadSelfAttention(nn.Module):
    """多头自注意力机制"""

    def __init__(self, embed_dim=128, num_heads=8, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x: [B, L, embed_dim]
            mask: [B, L] (1 for valid, 0 for padding)
        """
        B, L, C = x.shape

        # QKV projection
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # 应用mask
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # 加权求和
        out = (attn @ v).transpose(1, 2).reshape(B, L, C)
        out = self.proj(out)

        return out


class TransformerEncoderLayer(nn.Module):
    """Transformer编码器层"""

    def __init__(self, embed_dim=128, num_heads=8, ff_dim=512, dropout=0.1):
        super().__init__()

        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, mask=None):
        # Self-attention
        attn_out = self.attn(x, mask)
        x = self.norm1(x + attn_out)

        # Feed-forward
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x


class AttentionPooling(nn.Module):
    """注意力池化"""

    def __init__(self, embed_dim=128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )

    def forward(self, x, mask=None):
        """
        Args:
            x: [B, L, embed_dim]
            mask: [B, L]
        Returns:
            pooled: [B, embed_dim]
        """
        scores = self.attention(x).squeeze(-1)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        weights = F.softmax(scores, dim=1).unsqueeze(-1)
        pooled = (x * weights).sum(dim=1)

        return pooled


class AMPTransformerClassifier(nn.Module):
    """
    基于Transformer的SOTA AMP分类器

    参考文献：
    - AMPlify (Nature Communications 2022)
    - AMP-BERT (Briefings in Bioinformatics 2023)
    """

    def __init__(
            self,
            vocab_size: int = 25,
            embed_dim: int = 128,
            num_layers: int = 4,
            num_heads: int = 8,
            ff_dim: int = 512,
            dropout: float = 0.2,
            max_len: int = 100,
            use_physicochemical: bool = True
    ):
        super().__init__()

        self.use_physicochemical = use_physicochemical

        # 嵌入层
        self.embedding = ProteinEmbedding(vocab_size, embed_dim, max_len)

        # Transformer编码器
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        # 注意力池化
        self.attention_pooling = AttentionPooling(embed_dim)

        # 物化性质编码器
        if use_physicochemical:
            self.physchem_encoder = nn.Sequential(
                nn.Linear(8, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            fusion_dim = embed_dim + 256
        else:
            fusion_dim = embed_dim

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, 1)
        )

    def forward(self, seq_tokens, physicochemical_features=None, mask=None):
        """
        Args:
            seq_tokens: [B, L]
            physicochemical_features: [B, 8]
            mask: [B, L]
        Returns:
            logits: [B, 1]
        """
        # 嵌入
        x = self.embedding(seq_tokens)

        # Transformer编码
        for layer in self.transformer_layers:
            x = layer(x, mask)

        # 注意力池化
        pooled = self.attention_pooling(x, mask)

        # 融合物化性质
        if self.use_physicochemical and physicochemical_features is not None:
            physchem_emb = self.physchem_encoder(physicochemical_features)
            pooled = torch.cat([pooled, physchem_emb], dim=-1)

        # 分类
        logits = self.classifier(pooled)

        return logits

    def predict_proba(self, seq_tokens, physicochemical_features=None, mask=None):
        """返回AMP概率"""
        logits = self.forward(seq_tokens, physicochemical_features, mask)
        probs = torch.sigmoid(logits)
        return probs


class CNNBiLSTMClassifier(nn.Module):
    """
    增强版CNN-BiLSTM分类器（快速备选方案）
    """

    def __init__(
            self,
            vocab_size: int = 25,
            embedding_dim: int = 128,
            hidden_dim: int = 256,
            num_filters: int = 128,
            kernel_sizes: List[int] = [3, 5, 7, 9],
            lstm_layers: int = 2,
            dropout: float = 0.3
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # 多尺度CNN
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(embedding_dim, num_filters, k, padding=k // 2),
                nn.BatchNorm1d(num_filters),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            for k in kernel_sizes
        ])

        # BiLSTM
        self.lstm = nn.LSTM(
            num_filters * len(kernel_sizes),
            hidden_dim,
            num_layers=lstm_layers,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0,
            batch_first=True
        )

        # 注意力池化
        self.attention = AttentionPooling(hidden_dim * 2)

        # 物化性质编码器
        self.physchem_encoder = nn.Sequential(
            nn.Linear(8, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 256),
            nn.ReLU()
        )

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 256, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(256, 1)
        )

    def forward(self, seq_tokens, physicochemical_features, mask=None):
        # Embedding
        x = self.embedding(seq_tokens)
        x = x.transpose(1, 2)

        # 多尺度CNN
        conv_outputs = [conv(x) for conv in self.convs]
        x = torch.cat(conv_outputs, dim=1)
        x = x.transpose(1, 2)

        # BiLSTM
        lstm_out, _ = self.lstm(x)

        # 注意力池化
        pooled = self.attention(lstm_out, mask)

        # 融合物化性质
        physchem_emb = self.physchem_encoder(physicochemical_features)
        fused = torch.cat([pooled, physchem_emb], dim=-1)

        # 分类
        logits = self.classifier(fused)
        return logits

    def predict_proba(self, seq_tokens, physicochemical_features, mask=None):
        logits = self.forward(seq_tokens, physicochemical_features, mask)
        return torch.sigmoid(logits)


# ======================== SOTA分类器训练函数 ========================

def train_sota_amp_classifier(
        dataset,
        model_type: str = 'transformer',
        device: str = 'cuda',
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        early_stopping_patience: int = 10
):
    """
    训练SOTA AMP分类器

    Args:
        model_type: 'transformer' (最佳性能) 或 'cnn_lstm' (快速)
    """
    print(f"🎓 Training SOTA AMP Classifier ({model_type.upper()})...")

    physchem = PhysicochemicalCalculator()
    tokenizer = AminoAcidTokenizer(max_len=50)

    # 准备数据
    print("   Preparing dataset...")
    X_seq = []
    X_props = []
    y = []

    sample_size = min(len(dataset), 10000)

    for i in tqdm(range(sample_size), desc="Extracting positive samples"):
        sample = dataset[i]
        X_seq.append(sample['seq_tokens'].numpy())
        X_props.append(sample['global_properties'].numpy())
        y.append(1)

    # 生成高质量负样本
    print("   Generating negative samples...")
    neg_strategies = ['shuffle', 'reverse', 'random']

    for i in tqdm(range(sample_size), desc="Generating negatives"):
        strategy = neg_strategies[i % len(neg_strategies)]
        orig_tokens = X_seq[i].copy()

        if strategy == 'shuffle':
            neg_tokens = orig_tokens.copy()
            non_pad = neg_tokens[neg_tokens != 0]
            np.random.shuffle(non_pad)
            neg_tokens[neg_tokens != 0] = non_pad

        elif strategy == 'reverse':
            neg_tokens = orig_tokens.copy()
            non_pad_idx = np.where(neg_tokens != 0)[0]
            if len(non_pad_idx) > 0:
                neg_tokens[non_pad_idx] = neg_tokens[non_pad_idx][::-1]

        else:  # random
            length = (orig_tokens != 0).sum()
            neg_tokens = np.zeros_like(orig_tokens)
            neg_tokens[:length] = np.random.randint(1, 21, size=length)

        X_seq.append(neg_tokens)

        # 重新计算物化性质
        neg_seq_str = tokenizer.decode(torch.tensor(neg_tokens))
        if len(neg_seq_str) > 0:
            neg_props = physchem.compute_properties(neg_seq_str)
        else:
            neg_props = np.zeros(8)
        X_props.append(neg_props)

        y.append(0)

    X_seq = np.array(X_seq)
    X_props = np.array(X_props)
    y = np.array(y)

    # 数据划分
    indices = np.arange(len(y))
    train_idx, val_idx = train_test_split(
        indices, test_size=0.15, stratify=y, random_state=42
    )

    print(f"   Dataset: {len(y)} samples (Train: {len(train_idx)}, Val: {len(val_idx)})")
    print(f"   Positive: {np.sum(y)}, Negative: {len(y) - np.sum(y)}")

    # 初始化模型
    if model_type == 'transformer':
        model = AMPTransformerClassifier(
            vocab_size=25,
            embed_dim=128,
            num_layers=4,
            num_heads=8,
            ff_dim=512,
            dropout=0.2,
            max_len=100,
            use_physicochemical=True
        ).to(device)
    else:  # cnn_lstm
        model = CNNBiLSTMClassifier(
            vocab_size=25,
            embedding_dim=128,
            hidden_dim=256,
            num_filters=128,
            kernel_sizes=[3, 5, 7, 9],
            lstm_layers=2,
            dropout=0.3
        ).to(device)

    # 训练配置
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=False
    )

    # 类别权重
    pos_weight = torch.tensor([len(y) / (2 * np.sum(y))]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # 训练循环
    best_val_auc = 0
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        # ===== 训练阶段 =====
        model.train()
        train_losses = []
        train_preds = []
        train_labels = []

        # 打乱训练数据
        np.random.shuffle(train_idx)

        for i in range(0, len(train_idx), batch_size):
            batch_idx = train_idx[i:i + batch_size]

            batch_seq = torch.LongTensor(X_seq[batch_idx]).to(device)
            batch_props = torch.FloatTensor(X_props[batch_idx]).to(device)
            batch_labels = torch.FloatTensor(y[batch_idx]).unsqueeze(1).to(device)
            batch_mask = (batch_seq != 0).long()

            optimizer.zero_grad()
            logits = model(batch_seq, batch_props, batch_mask)
            loss = criterion(logits, batch_labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(loss.item())
            train_preds.extend(torch.sigmoid(logits).detach().cpu().numpy())
            train_labels.extend(batch_labels.cpu().numpy())

        # ===== 验证阶段 =====
        model.eval()
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for i in range(0, len(val_idx), batch_size):
                batch_idx = val_idx[i:i + batch_size]

                batch_seq = torch.LongTensor(X_seq[batch_idx]).to(device)
                batch_props = torch.FloatTensor(X_props[batch_idx]).to(device)
                batch_labels = torch.FloatTensor(y[batch_idx]).unsqueeze(1).to(device)
                batch_mask = (batch_seq != 0).long()

                logits = model(batch_seq, batch_props, batch_mask)
                val_preds.extend(torch.sigmoid(logits).cpu().numpy())
                val_labels.extend(batch_labels.cpu().numpy())

        # 计算指标
        train_loss = np.mean(train_losses)
        train_auc = roc_auc_score(train_labels, train_preds)
        val_auc = roc_auc_score(val_labels, val_preds)

        val_acc = accuracy_score(
            val_labels,
            (np.array(val_preds) > 0.5).astype(int)
        )
        val_f1 = f1_score(
            val_labels,
            (np.array(val_preds) > 0.5).astype(int)
        )

        scheduler.step(val_auc)

        # Early stopping
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1}/{epochs} | "
                  f"Loss: {train_loss:.4f} | "
                  f"Train AUC: {train_auc:.4f} | "
                  f"Val AUC: {val_auc:.4f} | "
                  f"Val Acc: {val_acc:.4f} | "
                  f"Val F1: {val_f1:.4f}")

        if patience_counter >= early_stopping_patience:
            print(f"  ⚠️  Early stopping at epoch {epoch + 1}")
            break

    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    model.to(device)
    model.eval()

    print(f"  ✓ Training completed! Best Val AUC: {best_val_auc:.4f}\n")

    return model


# ======================== 消融模型变体 ========================

class AblationModelWrapper:
    """
    包装原始模型，实现不同的消融变体
    """

    def __init__(self, model: MultiScaleConditionalDiffusion, mode: str):
        """
        mode: 'seq_only', 'seq_struct', 'seq_property', 'full'
        """
        self.model = model
        self.mode = mode
        self.model.eval()

    @torch.no_grad()
    def generate(self, num_samples, seq_len, guidance_scale, device):
        """根据消融模式生成序列"""
        if self.mode == 'seq_only':
            return self._generate_seq_only(num_samples, seq_len, guidance_scale, device)
        elif self.mode == 'seq_struct':
            return self._generate_seq_struct(num_samples, seq_len, guidance_scale, device)
        elif self.mode == 'seq_property':
            return self._generate_seq_property(num_samples, seq_len, guidance_scale, device)
        elif self.mode == 'full':
            return self.model.generate(
                num_samples=num_samples,
                seq_len=seq_len,
                guidance_scale=guidance_scale,
                device=device
            )
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _generate_seq_only(self, num_samples, seq_len, guidance_scale, device):
        """只用Layer 1 (序列层)"""
        L = seq_len

        # 只生成 Layer 1
        shape1 = (num_samples, L, self.model.config.layer1.latent_dim)
        z1 = self.model.diffusion_l1.sample(
            self.model.denoise_l1, shape1, guidance_scale=guidance_scale
        )

        # Layer 2 和 3 置零
        z2 = torch.zeros(num_samples, L, self.model.config.layer2.latent_dim, device=device)
        z3 = torch.zeros(num_samples, L, self.model.config.layer3.latent_dim, device=device)

        return self._decode_all_layers(z1, z2, z3, num_samples, seq_len, device)

    def _generate_seq_struct(self, num_samples, seq_len, guidance_scale, device):
        """Layer 1 + Layer 2 (序列 + 结构)"""
        L = seq_len

        shape1 = (num_samples, L, self.model.config.layer1.latent_dim)
        shape2 = (num_samples, L, self.model.config.layer2.latent_dim)

        z1 = self.model.diffusion_l1.sample(
            self.model.denoise_l1, shape1, guidance_scale=guidance_scale
        )
        z2 = self.model.diffusion_l2.sample(
            self.model.denoise_l2, shape2, guidance_scale=guidance_scale
        )

        z3 = torch.zeros(num_samples, L, self.model.config.layer3.latent_dim, device=device)

        # 只应用 L1 <-> L2 跨层注意力
        mask = torch.ones(num_samples, L, device=device)
        z1, z2 = self.model.cross_attn_12(z1, z2, mask, mask)

        return self._decode_all_layers(z1, z2, z3, num_samples, seq_len, device)

    def _generate_seq_property(self, num_samples, seq_len, guidance_scale, device):
        """Layer 1 + Layer 3 (序列 + 性质)"""
        L = seq_len

        shape1 = (num_samples, L, self.model.config.layer1.latent_dim)
        shape3 = (num_samples, L, self.model.config.layer3.latent_dim)

        z1 = self.model.diffusion_l1.sample(
            self.model.denoise_l1, shape1, guidance_scale=guidance_scale
        )
        z3 = self.model.diffusion_l3.sample(
            self.model.denoise_l3, shape3, guidance_scale=guidance_scale
        )

        z2 = torch.zeros(num_samples, L, self.model.config.layer2.latent_dim, device=device)

        # 只应用 L1 <-> L3 跨层注意力
        mask = torch.ones(num_samples, L, device=device)
        z1, z3 = self.model.cross_attn_13(z1, z3, mask, mask)

        return self._decode_all_layers(z1, z2, z3, num_samples, seq_len, device)

    def _decode_all_layers(self, z1, z2, z3, num_samples, seq_len, device):

        """统一的解码逻辑"""
        t_zero = torch.zeros(num_samples, device=device, dtype=torch.long)
        mask = torch.ones(num_samples, seq_len, device=device)

        '''# ========== 解码序列（使用温度采样）==========
        seq_logits = self.model.decoder_l1(z1, t_zero, mask)  # [B, L, vocab_size]

        # ✅ 方法1：温度采样（推荐）
        temperature = 0.2 # 可调参数：1.0=原始分布, >1=更随机, <1=更确定

        # 应用温度缩放
        scaled_logits = seq_logits / temperature

        # 对每个位置独立采样
        batch_size, seq_length, vocab_size = seq_logits.shape
        probs = F.softmax(scaled_logits, dim=-1)  # [B, L, V]

        # 重塑为2D进行采样
        probs_2d = probs.view(-1, vocab_size)  # [B*L, V]
        sampled_tokens = torch.multinomial(probs_2d, num_samples=1)  # [B*L, 1]
        seq_tokens = sampled_tokens.view(batch_size, seq_length)  # [B, L]

        # ========== 解码二级结构（可以用argmax）==========
        ss_logits = self.model.decoder_l2(z2, t_zero, mask)
        ss_label = ss_logits.argmax(dim=-1)  # 结构预测不需要多样性

        # ========== 解码物化性质 ==========
        global_props, residue_props = self.model.decoder_l3(z3, t_zero, mask)
        '''

        # 解码序列
        seq_logits = self.model.decoder_l1(z1, t_zero, mask)
        seq_tokens = seq_logits.argmax(dim=-1)

        # 解码二级结构 (全局标签)
        ss_logits = self.model.decoder_l2(z2, t_zero, mask)
        ss_label = ss_logits.argmax(dim=-1)  # [B]

        # 解码物化性质
        global_props, residue_props = self.model.decoder_l3(z3, t_zero, mask)
        return {
            'seq_tokens': seq_tokens,
            'ss_label': ss_label,
            'seq_props': global_props,
            'residue_props': residue_props,
            'seq_mask': mask,
        }

# ======================== 评估指标 ========================

def compute_reconstruction_loss(model_wrapper, test_loader, device, mode='full'):
    """计算序列重建损失"""
    model = model_wrapper.model
    model.eval()

    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Computing recon loss ({mode})"):
            seq_tokens = batch['seq_tokens'].to(device)
            ss_label = batch['ss_label'].to(device)
            seq_props = batch['global_properties'].to(device)
            residue_props = batch['residue_properties'].to(device)
            mask = batch['seq_mask'].to(device)

            batch_size, seq_len = seq_tokens.shape

            # 编码
            z1_mean, z1_logvar = model.encoder_l1(seq_tokens, mask)
            z2_mean, z2_logvar = model.encoder_l2(ss_label, mask)
            z3_mean, z3_logvar = model.encoder_l3(seq_props, residue_props, mask)

            # 根据模式选择性使用层
            if mode == 'seq_only':
                z2_mean = torch.zeros_like(z2_mean)
                z3_mean = torch.zeros_like(z3_mean)
                z1_coupled, z2_coupled, z3_coupled = z1_mean, z2_mean, z3_mean

            elif mode == 'seq_struct':
                z3_mean = torch.zeros_like(z3_mean)
                z1_coupled, z2_coupled = model.cross_attn_12(z1_mean, z2_mean, mask, mask)
                z3_coupled = z3_mean

            elif mode == 'seq_property':
                z2_mean = torch.zeros_like(z2_mean)
                z1_coupled, z3_coupled = model.cross_attn_13(z1_mean, z3_mean, mask, mask)
                z2_coupled = z2_mean

            else:  # full
                z1_coupled, z2_coupled, z3_coupled = model.apply_cross_layer_attention(
                    z1_mean, z2_mean, z3_mean, mask
                )

            # 解码并计算损失
            t_zero = torch.zeros(batch_size, device=device, dtype=torch.long)

            seq_logits = model.decoder_l1(z1_coupled, t_zero, mask)
            loss = F.cross_entropy(
                seq_logits.reshape(-1, seq_logits.size(-1)),
                seq_tokens.reshape(-1),
                ignore_index=0,
                reduction='mean'
            )

            total_loss += loss.item()
            n_batches += 1

            if n_batches >= 50:  # 限制批次数
                break

    avg_loss = total_loss / max(n_batches, 1)
    return avg_loss


def compute_novelty(generated_seqs: List[str], training_seqs: List[str]) -> Dict:
    """计算新颖性指标"""
    training_set = set(training_seqs)
    generated_set = set(generated_seqs)

    # 新颖性
    novel_seqs = generated_set - training_set
    novelty_rate = len(novel_seqs) / max(len(generated_seqs), 1)

    # 唯一性
    unique_rate = len(generated_set) / max(len(generated_seqs), 1)

    # 平均编辑距离
    try:
        from Levenshtein import distance as levenshtein_distance

        edit_distances = []
        sample_gen = generated_seqs[:100]
        sample_train = training_seqs[:500]

        for gen_seq in sample_gen:
            if len(sample_train) == 0:
                edit_distances.append(0)
                continue
            min_dist = min(
                levenshtein_distance(gen_seq, train_seq)
                for train_seq in sample_train
            )
            edit_distances.append(min_dist)

        avg_edit_distance = np.mean(edit_distances) if edit_distances else 0.0

    except ImportError:
        print("  ⚠️  python-Levenshtein not installed, skipping edit distance")
        avg_edit_distance = 0.0

    return {
        'novelty_rate': novelty_rate,
        'unique_rate': unique_rate,
        'avg_edit_distance': avg_edit_distance,
    }


def compute_amp_probability_sota(
        generated_results: Dict,
        classifier: nn.Module,
        device: str = 'cuda'
) -> Dict:
    """使用SOTA分类器评估生成序列的AMP概率"""

    seq_tokens = generated_results['seq_tokens'].to(device)
    if 'seq_props' in generated_results:
        seq_props = generated_results['seq_props'].to(device)  # [N, 8]
    else:
        seq_props = generated_results['global_properties'].to(device)  # [N, 8]
    seq_mask = (seq_tokens != 0).long()

    classifier.eval()
    with torch.no_grad():
        amp_probs = classifier.predict_proba(
            seq_tokens, seq_props, seq_mask
        ).cpu().numpy().flatten()

    return {
        'mean_amp_prob': float(amp_probs.mean()),
        'std_amp_prob': float(amp_probs.std()),
        'median_amp_prob': float(np.median(amp_probs)),
        'high_conf_rate': float((amp_probs > 0.7).sum() / len(amp_probs)),
        'very_high_conf_rate': float((amp_probs > 0.9).sum() / len(amp_probs)),
        'min_amp_prob': float(amp_probs.min()),
        'max_amp_prob': float(amp_probs.max()),
    }


# ======================== 主消融实验 ========================

def run_ablation_study(
        checkpoint_path: str,
        data_path: str,
        output_dir: str,
        num_samples: int = 500,
        batch_size: int = 64,
        seq_len: int = 30,
        guidance_scale: float = 3.0,
        device: str = 'cuda',
        classifier_type: str = 'transformer',
):
    """运行完整的消融实验"""

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    print("=" * 70)
    print("  🧪 ABLATION STUDY: Multi-Scale Conditional Diffusion")
    print("=" * 70)

    # ---- 加载模型 ----
    print("\n📦 Loading model...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']

    base_model = MultiScaleConditionalDiffusion(config).to(device)

    if 'ema_shadow' in checkpoint:
        for name, param in base_model.named_parameters():
            if name in checkpoint['ema_shadow']:
                param.data = checkpoint['ema_shadow'][name].to(device)
        print("   ✓ Loaded EMA weights")
    else:
        base_model.load_state_dict(checkpoint['model_state_dict'])
        print("   ✓ Loaded model weights")

    base_model.eval()

    # ---- 加载数据集 ----
    print("\n📂 Loading dataset...")
    dataset = AMPDataset(data_path)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"   Train: {len(train_dataset)}, Test: {len(test_dataset)}")

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    # 提取训练序列
    tokenizer = AminoAcidTokenizer(max_len=seq_len)
    training_seqs = []
    for idx in range(min(1000, len(train_dataset))):
        sample = train_dataset[idx]
        seq = tokenizer.decode(sample['seq_tokens'])
        if len(seq) > 0:
            training_seqs.append(seq)

    print(f"   Extracted {len(training_seqs)} training sequences")

    # ---- 训练SOTA AMP分类器 ----
    print("\n" + "=" * 70)
    print("  🧬 Training SOTA AMP Classifier")
    print("=" * 70)

    amp_classifier = train_sota_amp_classifier(
        train_dataset,
        model_type=classifier_type,
        device=device,
        epochs=50,
        batch_size=32,
        learning_rate=1e-3,
        early_stopping_patience=10
    )

    # ---- 定义实验组 ----
    ablation_modes = {
        'Seq-Only': 'seq_only',
        'Seq+Struct': 'seq_struct',
        'Seq+Property': 'seq_property',
        'Full-Model': 'full',
    }

    results = {}

    # ---- 对每个实验组运行评估 ----
    for exp_name, mode in ablation_modes.items():

        print(f"\n{'=' * 70}")
        print(f"  🧬 Experiment: {exp_name} ({mode})")
        print(f"{'=' * 70}")

        model_wrapper = AblationModelWrapper(base_model, mode)

        # 1. 生成序列
        print(f"\n  ▸ Generating {num_samples} sequences...")
        generated_results = model_wrapper.generate(
            num_samples=num_samples,
            seq_len=seq_len,
            guidance_scale=guidance_scale,
            device=device
        )

        # 解码生成的序列
        generated_seqs = []
        for tokens in generated_results['seq_tokens']:
            seq = tokenizer.decode(tokens)
            if len(seq) > 0:
                generated_seqs.append(seq)

        print(f"     Generated {len(generated_seqs)} valid sequences")

        # 2. 计算重建损失
        print("  ▸ Computing reconstruction loss...")
        recon_loss = compute_reconstruction_loss(
            model_wrapper, test_loader, device, mode
        )

        # 3. 计算新颖性
        print("  ▸ Computing novelty metrics...")
        novelty_metrics = compute_novelty(generated_seqs, training_seqs)

        # 4. 计算AMP概率 (使用SOTA分类器)
        print("  ▸ Computing AMP probability (SOTA)...")
        amp_metrics = compute_amp_probability_sota(
            generated_results, amp_classifier, device
        )

        # 汇总结果
        results[exp_name] = {
            'mode': mode,
            'reconstruction_loss': recon_loss,
            'novelty': novelty_metrics,
            'amp_probability': amp_metrics,
            'generated_sequences': generated_seqs[:50],
        }

        # 打印结果
        print(f"\n  📊 Results for {exp_name}:")
        print(f"     Reconstruction Loss:     {recon_loss:.4f}")
        print(f"     Novelty Rate:            {novelty_metrics['novelty_rate']:.3f}")
        print(f"     Unique Rate:             {novelty_metrics['unique_rate']:.3f}")
        print(f"     Avg Edit Distance:       {novelty_metrics['avg_edit_distance']:.2f}")
        print(f"     Mean AMP Prob:           {amp_metrics['mean_amp_prob']:.3f}")
        print(f"     Median AMP Prob:         {amp_metrics['median_amp_prob']:.3f}")
        print(f"     High Conf Rate (>0.7):   {amp_metrics['high_conf_rate']:.3f}")
        print(f"     Very High Conf (>0.9):   {amp_metrics['very_high_conf_rate']:.3f}")

    # ---- 保存结果 ----
    print(f"\n💾 Saving results...")
    results_path = os.path.join(output_dir, 'ablation_results.json')

    results_serializable = {}
    for exp_name, res in results.items():
        results_serializable[exp_name] = {
            'mode': res['mode'],
            'reconstruction_loss': float(res['reconstruction_loss']),
            'novelty': {k: float(v) for k, v in res['novelty'].items()},
            'amp_probability': {k: float(v) for k, v in res['amp_probability'].items()},
            'sample_sequences': res['generated_sequences'][:10],
        }

    with open(results_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    print(f"   ✓ Saved to {results_path}")

    # ---- 可视化 ----
    print(f"\n🎨 Generating visualizations...")
    visualize_ablation_results(results, output_dir)

    print(f"\n{'=' * 70}")
    print("  ✅ Ablation study completed!")
    print(f"{'=' * 70}\n")
    print(results)
    return results


# ======================== 可视化 ========================

def visualize_ablation_results(results: Dict, output_dir: str):
    """可视化消融实验结果"""

    exp_names = list(results.keys())
    recon_losses = [results[name]['reconstruction_loss'] for name in exp_names]
    novelty_rates = [results[name]['novelty']['novelty_rate'] for name in exp_names]
    unique_rates = [results[name]['novelty']['unique_rate'] for name in exp_names]
    amp_probs = [results[name]['amp_probability']['mean_amp_prob'] for name in exp_names]
    high_conf_rates = [results[name]['amp_probability']['high_conf_rate'] for name in exp_names]

    # ---- 创建综合图表 ----
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Ablation Study Results: Multi-Scale Conditional Diffusion (SOTA Classifier)',
                 fontsize=16, fontweight='bold', y=0.98)

    colors = ['#3498DB', '#E74C3C', '#2ECC71', '#F39C12']

    # 1. 重建损失
    ax = axes[0, 0]
    bars = ax.bar(range(len(exp_names)), recon_losses, color=colors, alpha=0.7, edgecolor='white', linewidth=1.5)
    ax.set_xticks(range(len(exp_names)))
    ax.set_xticklabels(exp_names, rotation=15, ha='right')
    ax.set_ylabel('Reconstruction Loss', fontweight='bold')
    ax.set_title('(A) Sequence Reconstruction Loss\n(Lower is Better)', fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    for i, (bar, val) in enumerate(zip(bars, recon_losses)):
        ax.text(i, val, f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # 2. 新颖性率
    ax = axes[0, 1]
    bars = ax.bar(range(len(exp_names)), novelty_rates, color=colors, alpha=0.7, edgecolor='white', linewidth=1.5)
    ax.set_xticks(range(len(exp_names)))
    ax.set_xticklabels(exp_names, rotation=15, ha='right')
    ax.set_ylabel('Novelty Rate', fontweight='bold')
    ax.set_title('(B) Novelty Rate\n(Higher is Better)', fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    for i, (bar, val) in enumerate(zip(bars, novelty_rates)):
        ax.text(i, val + 0.02, f'{val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # 3. 唯一性率
    ax = axes[0, 2]
    bars = ax.bar(range(len(exp_names)), unique_rates, color=colors, alpha=0.7, edgecolor='white', linewidth=1.5)
    ax.set_xticks(range(len(exp_names)))
    ax.set_xticklabels(exp_names, rotation=15, ha='right')
    ax.set_ylabel('Unique Rate', fontweight='bold')
    ax.set_title('(C) Sequence Uniqueness\n(Higher is Better)', fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    for i, (bar, val) in enumerate(zip(bars, unique_rates)):
        ax.text(i, val + 0.02, f'{val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # 4. 平均AMP概率
    ax = axes[1, 0]
    bars = ax.bar(range(len(exp_names)), amp_probs, color=colors, alpha=0.7, edgecolor='white', linewidth=1.5)
    ax.set_xticks(range(len(exp_names)))
    ax.set_xticklabels(exp_names, rotation=15, ha='right')
    ax.set_ylabel('Mean AMP Probability', fontweight='bold')
    ax.set_title('(D) Average AMP Probability (SOTA)\n(Higher is Better)', fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    for i, (bar, val) in enumerate(zip(bars, amp_probs)):
        ax.text(i, val + 0.02, f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # 5. 高置信度率
    ax = axes[1, 1]
    bars = ax.bar(range(len(exp_names)), high_conf_rates, color=colors, alpha=0.7, edgecolor='white', linewidth=1.5)
    ax.set_xticks(range(len(exp_names)))
    ax.set_xticklabels(exp_names, rotation=15, ha='right')
    ax.set_ylabel('High Confidence Rate (>0.7)', fontweight='bold')
    ax.set_title('(E) High-Confidence AMP Rate\n(Higher is Better)', fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    for i, (bar, val) in enumerate(zip(bars, high_conf_rates)):
        ax.text(i, val + 0.02, f'{val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # 6. 综合雷达图
    ax = axes[1, 2]

    max_recon = max(recon_losses) if max(recon_losses) > 0 else 1
    metrics = np.array([
        [1 - recon_losses[i] / max_recon, novelty_rates[i], unique_rates[i],
         amp_probs[i], high_conf_rates[i]]
        for i in range(len(exp_names))
    ])

    categories = ['Recon\n(norm)', 'Novelty', 'Unique', 'AMP\nProb', 'High\nConf']
    N = len(categories)

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    ax = plt.subplot(2, 3, 6, projection='polar')

    for i, (name, color) in enumerate(zip(exp_names, colors)):
        values = metrics[i].tolist()
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=name, color=color, markersize=6)
        ax.fill(angles, values, alpha=0.15, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_title('(F) Comprehensive Comparison', fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.4)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'ablation_comparison_sota.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()

    print(f"   ✓ Visualization saved to {save_path}")

    # ---- 生成表格摘要 ----
    summary_df = pd.DataFrame({
        'Experiment': exp_names,
        'Recon Loss': [f'{x:.4f}' for x in recon_losses],
        'Novelty': [f'{x:.3f}' for x in novelty_rates],
        'Unique': [f'{x:.3f}' for x in unique_rates],
        'AMP Prob': [f'{x:.3f}' for x in amp_probs],
        'High Conf': [f'{x:.3f}' for x in high_conf_rates],
    })

    table_path = os.path.join(output_dir, 'ablation_summary_sota.csv')
    summary_df.to_csv(table_path, index=False)
    print(f"   ✓ Summary table saved to {table_path}")

    print("\n" + "=" * 70)
    print("  📊 SUMMARY TABLE (SOTA Classifier)")
    print("=" * 70)
    print(summary_df.to_string(index=False))
    print("=" * 70 + "\n")


# ======================== 主函数 ========================

def main():
    parser = argparse.ArgumentParser(
        description='Ablation Study with SOTA AMP Classifier'
    )
    parser.add_argument('--checkpoint', type=str,
                        default='checkpoints/best_model.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--data_path', type=str,
                        default='data/amp_extended_dataset_v2.csv',
                        help='Path to dataset CSV')
    parser.add_argument('--output_dir', type=str,
                        default='results/ablation_sota',
                        help='Output directory for results')
    parser.add_argument('--num_samples', type=int, default=200,
                        help='Number of sequences to generate per experiment')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seq_len', type=int, default=30)
    parser.add_argument('--guidance_scale', type=float, default=2.0)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--classifier_type', type=str, default='transformer',
                        choices=['transformer', 'cnn_lstm'],
                        help='SOTA classifier architecture')

    args = parser.parse_args()

    # 运行消融实验
    results = run_ablation_study(
        checkpoint_path=args.checkpoint,
        data_path=args.data_path,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        guidance_scale=args.guidance_scale,
        device=args.device,
        classifier_type=args.classifier_type,
    )


if __name__ == '__main__':
    main()
