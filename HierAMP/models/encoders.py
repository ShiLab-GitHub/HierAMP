"""
Encoders & Decoders for Three Latent Layers
Layer 1: Sequence Motif Encoder/Decoder
Layer 2: Secondary Structure Encoder/Decoder
Layer 3: Physicochemical Property Encoder/Decoder
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import (
    MultiScaleConv1D, TransformerBlock,
    SinusoidalTimeEmbedding, AdaptiveLayerNorm
)


class SequenceMotifEncoder(nn.Module):
    """
    Layer 1 编码器: 序列 Motif 层
    使用多尺度 CNN 捕获局部 motif 模式
    + Transformer 层建模全局上下文
    """

    def __init__(self, vocab_size=22, max_len=50, latent_dim=128,
                 hidden_dim=256, num_heads=8, num_layers=4,
                 kernel_sizes=[3, 5, 7], dropout=0.1):
        super().__init__()
        self.latent_dim = latent_dim

        # Amino acid embedding
        self.aa_embed = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.pos_embed = nn.Embedding(max_len, hidden_dim)

        # Multi-scale CNN for local motif detection
        self.motif_conv = MultiScaleConv1D(
            hidden_dim, hidden_dim, kernel_sizes
        )

        # Transformer layers for global context
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, hidden_dim, dropout)
            for _ in range(num_layers)
        ])

        # Project to latent space
        self.to_latent = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, latent_dim * 2)  # mean + logvar
        )

    def forward(self, seq_tokens, mask=None):
        """
        Args:
            seq_tokens: [B, L] 氨基酸 token IDs
            mask: [B, L] padding mask
        Returns:
            z_mean: [B, L, latent_dim]
            z_logvar: [B, L, latent_dim]
        """
        B, L = seq_tokens.shape

        # Embeddings
        positions = torch.arange(L, device=seq_tokens.device).unsqueeze(0)
        x = self.aa_embed(seq_tokens) + self.pos_embed(positions)

        # CNN motif detection
        x_conv = x.permute(0, 2, 1)     # [B, D, L]
        x_conv = self.motif_conv(x_conv)  # [B, D, L]
        x = x + x_conv.permute(0, 2, 1)  # Residual

        # Transformer (use zero cond for encoder)
        cond = torch.zeros(B, x.shape[-1], device=x.device)
        for block in self.transformer_blocks:
            x = block(x, cond, mask)

        # To latent
        stats = self.to_latent(x)  # [B, L, 2*latent_dim]
        z_mean, z_logvar = stats.chunk(2, dim=-1)

        return z_mean, z_logvar


class SequenceMotifDecoder(nn.Module):
    """Layer 1 解码器: 从潜在表示重建序列"""

    def __init__(self, vocab_size=22, max_len=50, latent_dim=128,
                 hidden_dim=256, num_heads=8, num_layers=4,
                 time_dim=256, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(latent_dim, hidden_dim)
        self.pos_embed = nn.Embedding(max_len, hidden_dim)
        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        self.time_proj = nn.Linear(time_dim, hidden_dim)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, hidden_dim, dropout)
            for _ in range(num_layers)
        ])

        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, vocab_size)
        )

    def forward(self, z, t, mask=None):
        """
        Args:
            z: [B, L, latent_dim] 潜在表示
            t: [B] 扩散时间步
        Returns:
            logits: [B, L, vocab_size]
        """
        B, L, _ = z.shape
        positions = torch.arange(L, device=z.device).unsqueeze(0)
        x = self.input_proj(z) + self.pos_embed(positions)

        # Time conditioning
        t_emb = self.time_embed(t)       # [B, time_dim]
        cond = self.time_proj(t_emb)     # [B, hidden_dim]

        for block in self.transformer_blocks:
            x = block(x, cond, mask)

        return self.output_proj(x)


class SecondaryStructureEncoder(nn.Module):
    """
    Layer 2 编码器: 二级结构层 —— 全局单标签版本
    输入: ss_label [B]，取值 0(H) 或 1(E)
    处理: 嵌入为全局向量 → MLP → 广播到序列维度
    输出: z_mean, z_logvar，shape 均为 [B, L, latent_dim]
    """

    def __init__(self, num_ss_types=2, max_len=50, latent_dim=64,
                 hidden_dim=128, num_heads=4, num_layers=3,
                 ss_embed_dim=32, dropout=0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.max_len = max_len

        # 全局标签嵌入：[B] -> [B, ss_embed_dim]
        self.ss_embed = nn.Embedding(num_ss_types, ss_embed_dim)

        # MLP 将全局嵌入映射到 hidden_dim
        self.global_proj = nn.Sequential(
            nn.Linear(ss_embed_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 位置编码（与序列层共享长度语义）
        self.pos_embed = nn.Embedding(max_len, hidden_dim)

        # Transformer 精炼（使全局信息在序列维度上分布）
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, hidden_dim, dropout)
            for _ in range(num_layers)
        ])

        self.to_latent = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, latent_dim * 2)
        )

    def forward(self, ss_label, mask=None):
        """
        Args:
            ss_label: [B] 全局二级结构标签 (H=0, E=1)
            mask:     [B, L] 序列掩码（可选）
        Returns:
            z_mean, z_logvar: [B, L, latent_dim]
        """
        B = ss_label.shape[0]
        L = self.max_len
        device = ss_label.device

        # 1. 全局标签 → 嵌入向量 [B, hidden_dim]
        g = self.global_proj(self.ss_embed(ss_label))   # [B, hidden_dim]

        # 2. 广播到序列维度 [B, L, hidden_dim]
        positions = torch.arange(L, device=device).unsqueeze(0)  # [1, L]
        x = g.unsqueeze(1) + self.pos_embed(positions)           # [B, L, hidden_dim]

        # 3. Transformer 精炼
        cond = torch.zeros(B, x.shape[-1], device=device)
        for block in self.transformer_blocks:
            x = block(x, cond, mask)

        # 4. 映射到潜在空间均值和方差
        stats = self.to_latent(x)                        # [B, L, latent_dim*2]
        z_mean, z_logvar = stats.chunk(2, dim=-1)
        return z_mean, z_logvar


class SecondaryStructureDecoder(nn.Module):
    """
    Layer 2 解码器 —— 全局单标签版本
    输入: z [B, L, latent_dim]
    输出: logits [B, num_ss_types]（全局分类，对序列做 mean-pool）
    """

    def __init__(self, num_ss_types=2, max_len=50, latent_dim=64,
                 hidden_dim=128, num_heads=4, num_layers=3,
                 time_dim=128, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(latent_dim, hidden_dim)
        self.pos_embed = nn.Embedding(max_len, hidden_dim)
        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        self.time_proj = nn.Linear(time_dim, hidden_dim)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, hidden_dim, dropout)
            for _ in range(num_layers)
        ])

        # 全局池化后分类
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_ss_types)
        )

    def forward(self, z, t, mask=None):
        """
        Args:
            z:    [B, L, latent_dim]
            t:    [B] 时间步
            mask: [B, L] 序列掩码（可选）
        Returns:
            logits: [B, num_ss_types]  全局分类 logits
        """
        B, L, _ = z.shape
        positions = torch.arange(L, device=z.device).unsqueeze(0)
        x = self.input_proj(z) + self.pos_embed(positions)   # [B, L, hidden_dim]

        t_emb = self.time_embed(t)
        cond = self.time_proj(t_emb)                          # [B, hidden_dim]

        for block in self.transformer_blocks:
            x = block(x, cond, mask)

        # 序列维度 mean-pool → 全局向量
        if mask is not None:
            # 只对有效位置平均
            m = mask.unsqueeze(-1)                            # [B, L, 1]
            x_pool = (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
        else:
            x_pool = x.mean(dim=1)                           # [B, hidden_dim]

        return self.output_proj(x_pool)                       # [B, num_ss_types]


class PhysicochemEncoder(nn.Module):
    """
    Layer 3 编码器: 物化性质层
    编码全局和 per-residue 物化性质
    """

    def __init__(self, num_properties=8, max_len=50, latent_dim=32,
                 hidden_dim=64, num_heads=4, num_layers=2,
                 dropout=0.1):
        super().__init__()
        self.latent_dim = latent_dim

        # Global property encoder
        self.global_encoder = nn.Sequential(
            nn.Linear(num_properties, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU()
        )

        # Per-residue property encoder
        self.residue_proj = nn.Linear(3, hidden_dim)  # charge, hydro, mw
        self.pos_embed = nn.Embedding(max_len, hidden_dim)

        # Combine global + local
        self.combine = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU()
        )

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, hidden_dim, dropout)
            for _ in range(num_layers)
        ])

        self.to_latent = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, latent_dim * 2)
        )

    def forward(self, global_props, residue_props, mask=None):
        """
        Args:
            global_props:  [B, 8] 全局物化性质
            residue_props: [B, L, 3] 每残基物化性质
        Returns:
            z_mean, z_logvar: [B, L, latent_dim]
        """
        B, L, _ = residue_props.shape
        positions = torch.arange(L, device=residue_props.device).unsqueeze(0)

        # Global features
        g = self.global_encoder(global_props)        # [B, H]
        g = g.unsqueeze(1).expand(-1, L, -1)         # [B, L, H]

        # Local features
        r = self.residue_proj(residue_props)          # [B, L, H]
        r = r + self.pos_embed(positions)

        # Combine
        x = self.combine(torch.cat([g, r], dim=-1))  # [B, L, H]

        cond = torch.zeros(B, x.shape[-1], device=x.device)
        for block in self.transformer_blocks:
            x = block(x, cond, mask)

        stats = self.to_latent(x)
        z_mean, z_logvar = stats.chunk(2, dim=-1)
        return z_mean, z_logvar


class PhysicochemDecoder(nn.Module):
    """Layer 3 解码器"""

    def __init__(self, num_properties=8, max_len=50, latent_dim=32,
                 hidden_dim=64, num_heads=4, num_layers=2,
                 time_dim=64, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(latent_dim, hidden_dim)
        self.pos_embed = nn.Embedding(max_len, hidden_dim)
        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        self.time_proj = nn.Linear(time_dim, hidden_dim)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, hidden_dim, dropout)
            for _ in range(num_layers)
        ])

        # Output: global props + per-residue props
        self.global_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_properties)
        )
        self.residue_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 3)
        )

    def forward(self, z, t, mask=None):
        B, L, _ = z.shape
        positions = torch.arange(L, device=z.device).unsqueeze(0)
        x = self.input_proj(z) + self.pos_embed(positions)

        t_emb = self.time_embed(t)
        cond = self.time_proj(t_emb)

        for block in self.transformer_blocks:
            x = block(x, cond, mask)

        # Global: average pool then predict
        global_pred = self.global_head(x.mean(dim=1))   # [B, 8]
        residue_pred = self.residue_head(x)              # [B, L, 3]

        return global_pred, residue_pred
