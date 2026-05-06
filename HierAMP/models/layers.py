"""
Custom Neural Network Layers
- Sinusoidal Time Embedding
- Cross-Layer Attention
- Adaptive Layer Normalization
- Multi-Scale Convolutional Block
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class SinusoidalTimeEmbedding(nn.Module):
    """正弦时间嵌入 (与 Transformer positional encoding 类似)"""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: [B] 时间步 (整数或浮点)
        Returns:
            [B, dim] 时间嵌入
        """
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(
            torch.arange(half_dim, device=device) * -embeddings
        )
        embeddings = t[:, None].float() * embeddings[None, :]
        embeddings = torch.cat([
            embeddings.sin(), embeddings.cos()
        ], dim=-1)

        return self.mlp(embeddings)


class AdaptiveLayerNorm(nn.Module):
    """自适应 Layer Normalization (条件归一化)
    通过时间嵌入动态调节 scale 和 shift
    """

    def __init__(self, hidden_dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, hidden_dim * 2)
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, D] or [B, D]
            cond: [B, cond_dim] 条件向量 (通常是时间嵌入)
        """
        x = self.norm(x)
        scale_shift = self.proj(cond)

        if x.dim() == 3:
            scale_shift = scale_shift.unsqueeze(1)  # [B, 1, 2D]

        scale, shift = scale_shift.chunk(2, dim=-1)
        return x * (1 + scale) + shift


class MultiScaleConv1D(nn.Module):
    """多尺度一维卷积块 (用于 motif 检测)"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: list = [3, 5, 7]
    ):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, k, padding=k // 2),
                nn.BatchNorm1d(out_channels),
                nn.GELU()
            )
            for k in kernel_sizes
        ])
        self.fusion = nn.Sequential(
            nn.Conv1d(out_channels * len(kernel_sizes), out_channels, 1),
            nn.GELU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, L]
        Returns:
            [B, out_channels, L]
        """
        conv_outs = [conv(x) for conv in self.convs]
        concat = torch.cat(conv_outs, dim=1)  # [B, C*num_kernels, L]
        return self.fusion(concat)


class CrossLayerAttention(nn.Module):
    """
    跨层 Attention 耦合模块
    实现不同层级潜在表示之间的信息交换

    支持:
    - Layer_i -> Layer_j 单向注意力
    - 双向注意力
    - 可调耦合强度
    """

    def __init__(
        self,
        query_dim: int,
        key_dim: int,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        coupling_strength: float = 0.5
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.coupling_strength = coupling_strength

        assert hidden_dim % num_heads == 0

        self.q_proj = nn.Linear(query_dim, hidden_dim)
        self.k_proj = nn.Linear(key_dim, hidden_dim)
        self.v_proj = nn.Linear(key_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, query_dim)

        self.norm_q = nn.LayerNorm(query_dim)
        self.norm_k = nn.LayerNorm(key_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

        # 可学习的耦合门控
        self.gate = nn.Sequential(
            nn.Linear(query_dim + key_dim, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        query_mask: torch.Tensor = None,
        kv_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            query:     [B, Lq, Dq] 查询层的表示
            key_value: [B, Lk, Dk] 被查询层的表示
            query_mask: [B, Lq] 查询 mask
            kv_mask:    [B, Lk] KV mask
        Returns:
            [B, Lq, Dq] 更新后的查询表示
        """
        B, Lq, Dq = query.shape
        Lk = key_value.shape[1]

        # Normalize
        q_normed = self.norm_q(query)
        kv_normed = self.norm_k(key_value)

        # Project
        Q = self.q_proj(q_normed)   # [B, Lq, H]
        K = self.k_proj(kv_normed)  # [B, Lk, H]
        V = self.v_proj(kv_normed)  # [B, Lk, H]

        # Reshape for multi-head
        Q = rearrange(Q, 'b l (h d) -> b h l d', h=self.num_heads)
        K = rearrange(K, 'b l (h d) -> b h l d', h=self.num_heads)
        V = rearrange(V, 'b l (h d) -> b h l d', h=self.num_heads)

        # Attention scores
        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # Apply mask
        if kv_mask is not None:
            kv_mask = kv_mask.unsqueeze(1).unsqueeze(2)  # [B,1,1,Lk]
            attn = attn.masked_fill(~kv_mask.bool(), float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Weighted sum
        out = torch.matmul(attn, V)
        out = rearrange(out, 'b h l d -> b l (h d)')
        out = self.out_proj(out)

        # Gate-controlled coupling (动态耦合强度)
        # 使用 query 和 key 的全局信息计算门控
        q_global = query.mean(dim=1)        # [B, Dq]
        kv_global = key_value.mean(dim=1)   # [B, Dk]
        gate_input = torch.cat([q_global, kv_global], dim=-1)
        gate_value = self.gate(gate_input).unsqueeze(1)  # [B, 1, 1]

        # 融合: 原始 + 耦合强度 * 门控 * 注意力输出
        return query + self.coupling_strength * gate_value * out


class BidirectionalCrossLayerAttention(nn.Module):
    """双向跨层注意力"""

    def __init__(self, dim_a: int, dim_b: int, hidden_dim: int,
                 num_heads: int = 8, dropout: float = 0.1,
                 coupling_strength: float = 0.5):
        super().__init__()
        self.attn_a2b = CrossLayerAttention(
            dim_a, dim_b, hidden_dim, num_heads, dropout, coupling_strength
        )
        self.attn_b2a = CrossLayerAttention(
            dim_b, dim_a, hidden_dim, num_heads, dropout, coupling_strength
        )

    def forward(self, feat_a, feat_b, mask_a=None, mask_b=None):
        """双向注意力: A 和 B 相互增强"""
        updated_a = self.attn_a2b(feat_a, feat_b, mask_a, mask_b)
        updated_b = self.attn_b2a(feat_b, feat_a, mask_b, mask_a)
        return updated_a, updated_b


class TransformerBlock(nn.Module):
    """标准 Transformer Block with Adaptive LayerNorm"""

    def __init__(self, hidden_dim: int, num_heads: int,
                 cond_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        self.norm1 = AdaptiveLayerNorm(hidden_dim, cond_dim)
        self.norm2 = AdaptiveLayerNorm(hidden_dim, cond_dim)

    def forward(self, x, cond, mask=None):
        """
        x: [B, L, D]
        cond: [B, cond_dim]
        """
        # Self-attention with adaptive norm
        h = self.norm1(x, cond)
        key_padding_mask = ~mask.bool() if mask is not None else None
        h, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask)
        x = x + h

        # Feedforward with adaptive norm
        h = self.norm2(x, cond)
        h = self.ff(h)
        x = x + h

        return x
