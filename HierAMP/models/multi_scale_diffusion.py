"""
Multi-Scale Conditional Diffusion Architecture
===============================================
Core architecture integrating three latent layers
with independent diffusion + cross-layer attention coupling.

Architecture:
    Layer 1 (Sequence Motif)  <--attention--> Layer 2 (Secondary Structure)
                                                    ^
    Layer 3 (Physicochemical) <--attention-----------+
                               <--attention--> Layer 1

Key Features:
    - Independent diffusion per layer
    - Bidirectional cross-layer attention coupling
    - Layer-specific control (fix any layer during generation)
    - Classifier-free guidance support
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from .layers import (
    SinusoidalTimeEmbedding,
    CrossLayerAttention,
    BidirectionalCrossLayerAttention,
    TransformerBlock,
    AdaptiveLayerNorm
)
from .encoders import (
    SequenceMotifEncoder, SequenceMotifDecoder,
    SecondaryStructureEncoder, SecondaryStructureDecoder,
    PhysicochemEncoder, PhysicochemDecoder
)
from .diffusion import GaussianDiffusion


# ============================================================
# Per-Layer Denoising Networks
# ============================================================

class Layer1DenoiseNet(nn.Module):
    """Layer 1 去噪网络 (序列 Motif)"""

    def __init__(self, latent_dim=128, hidden_dim=256,
                 num_heads=8, num_layers=4, time_dim=256,
                 max_len=50, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(latent_dim, hidden_dim)
        self.self_cond_proj = nn.Linear(latent_dim, hidden_dim)
        self.pos_embed = nn.Embedding(max_len, hidden_dim)
        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        self.time_proj = nn.Linear(time_dim, hidden_dim)

        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, hidden_dim, dropout)
            for _ in range(num_layers)
        ])

        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, z_noisy, t, condition=None, x_self_cond=None,
                cross_layer_feats=None, mask=None):
        B, L, _ = z_noisy.shape
        positions = torch.arange(L, device=z_noisy.device).unsqueeze(0)

        x = self.input_proj(z_noisy) + self.pos_embed(positions)

        # Self-conditioning
        if x_self_cond is not None:
            x = x + self.self_cond_proj(x_self_cond)

        # Time embedding
        t_emb = self.time_embed(t)
        cond = self.time_proj(t_emb)

        for block in self.blocks:
            x = block(x, cond, mask)

        return self.output_proj(x)


class Layer2DenoiseNet(nn.Module):
    """Layer 2 去噪网络 (二级结构)"""

    def __init__(self, latent_dim=64, hidden_dim=128,
                 num_heads=4, num_layers=3, time_dim=128,
                 max_len=50, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(latent_dim, hidden_dim)
        self.self_cond_proj = nn.Linear(latent_dim, hidden_dim)
        self.pos_embed = nn.Embedding(max_len, hidden_dim)
        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        self.time_proj = nn.Linear(time_dim, hidden_dim)

        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, hidden_dim, dropout)
            for _ in range(num_layers)
        ])

        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, z_noisy, t, condition=None, x_self_cond=None,
                cross_layer_feats=None, mask=None):
        B, L, _ = z_noisy.shape
        positions = torch.arange(L, device=z_noisy.device).unsqueeze(0)

        x = self.input_proj(z_noisy) + self.pos_embed(positions)
        if x_self_cond is not None:
            x = x + self.self_cond_proj(x_self_cond)

        t_emb = self.time_embed(t)
        cond = self.time_proj(t_emb)

        for block in self.blocks:
            x = block(x, cond, mask)

        return self.output_proj(x)


class Layer3DenoiseNet(nn.Module):
    """Layer 3 去噪网络 (物化性质)"""

    def __init__(self, latent_dim=32, hidden_dim=64,
                 num_heads=4, num_layers=2, time_dim=64,
                 max_len=50, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(latent_dim, hidden_dim)
        self.self_cond_proj = nn.Linear(latent_dim, hidden_dim)
        self.pos_embed = nn.Embedding(max_len, hidden_dim)
        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        self.time_proj = nn.Linear(time_dim, hidden_dim)

        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, hidden_dim, dropout)
            for _ in range(num_layers)
        ])

        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, z_noisy, t, condition=None, x_self_cond=None,
                cross_layer_feats=None, mask=None):
        B, L, _ = z_noisy.shape
        positions = torch.arange(L, device=z_noisy.device).unsqueeze(0)

        x = self.input_proj(z_noisy) + self.pos_embed(positions)
        if x_self_cond is not None:
            x = x + self.self_cond_proj(x_self_cond)

        t_emb = self.time_embed(t)
        cond = self.time_proj(t_emb)

        for block in self.blocks:
            x = block(x, cond, mask)

        return self.output_proj(x)


# ============================================================
# Multi-Scale Conditional Diffusion Model (主模型)
# ============================================================

class MultiScaleConditionalDiffusion(nn.Module):
    """
    多尺度条件扩散模型
    ===========================
    三层独立扩散 + 跨层 Attention 耦合

    层级结构:
        Layer 1: Sequence Motif      (latent_dim=128, 4 Transformer blocks)
        Layer 2: Secondary Structure (latent_dim=64,  3 Transformer blocks)
        Layer 3: Physicochemical     (latent_dim=32,  2 Transformer blocks)

    跨层耦合:
        Layer1 <--> Layer2 (双向 attention)
        Layer2 <--> Layer3 (双向 attention)
        Layer1 <--> Layer3 (双向 attention)
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # ---- Encoders (VAE-style, 用于训练时编码到潜在空间) ----
        self.encoder_l1 = SequenceMotifEncoder(
            vocab_size=config.data.vocab_size,
            max_len=config.data.max_seq_len,
            latent_dim=config.layer1.latent_dim,
            hidden_dim=config.layer1.hidden_dim,
            num_heads=config.layer1.num_heads,
            num_layers=config.layer1.num_layers,
            kernel_sizes=config.layer1.kernel_sizes,
            dropout=config.layer1.dropout
        )
        self.encoder_l2 = SecondaryStructureEncoder(
            num_ss_types=config.layer2.num_ss_types,
            max_len=config.data.max_seq_len,
            latent_dim=config.layer2.latent_dim,
            hidden_dim=config.layer2.hidden_dim,
            num_heads=config.layer2.num_heads,
            num_layers=config.layer2.num_layers,
            dropout=config.layer2.dropout
        )
        self.encoder_l3 = PhysicochemEncoder(
            num_properties=config.layer3.num_properties,
            max_len=config.data.max_seq_len,
            latent_dim=config.layer3.latent_dim,
            hidden_dim=config.layer3.hidden_dim,
            num_heads=config.layer3.num_heads,
            num_layers=config.layer3.num_layers,
            dropout=config.layer3.dropout
        )

        # ---- Denoising Networks (每层独立) ----
        self.denoise_l1 = Layer1DenoiseNet(
            latent_dim=config.layer1.latent_dim,
            hidden_dim=config.layer1.hidden_dim,
            num_heads=config.layer1.num_heads,
            num_layers=config.layer1.num_layers,
            max_len=config.data.max_seq_len,
            dropout=config.layer1.dropout
        )
        self.denoise_l2 = Layer2DenoiseNet(
            latent_dim=config.layer2.latent_dim,
            hidden_dim=config.layer2.hidden_dim,
            num_heads=config.layer2.num_heads,
            num_layers=config.layer2.num_layers,
            max_len=config.data.max_seq_len,
            dropout=config.layer2.dropout
        )
        self.denoise_l3 = Layer3DenoiseNet(
            latent_dim=config.layer3.latent_dim,
            hidden_dim=config.layer3.hidden_dim,
            num_heads=config.layer3.num_heads,
            num_layers=config.layer3.num_layers,
            max_len=config.data.max_seq_len,
            dropout=config.layer3.dropout
        )

        # ---- Decoders (从潜在空间解码回数据空间) ----
        self.decoder_l1 = SequenceMotifDecoder(
            vocab_size=config.data.vocab_size,
            max_len=config.data.max_seq_len,
            latent_dim=config.layer1.latent_dim,
            hidden_dim=config.layer1.hidden_dim,
            num_heads=config.layer1.num_heads,
            num_layers=config.layer1.num_layers,
            dropout=config.layer1.dropout
        )
        self.decoder_l2 = SecondaryStructureDecoder(
            num_ss_types=config.layer2.num_ss_types,
            max_len=config.data.max_seq_len,
            latent_dim=config.layer2.latent_dim,
            hidden_dim=config.layer2.hidden_dim,
            num_heads=config.layer2.num_heads,
            num_layers=config.layer2.num_layers,
            dropout=config.layer2.dropout
        )
        self.decoder_l3 = PhysicochemDecoder(
            num_properties=config.layer3.num_properties,
            max_len=config.data.max_seq_len,
            latent_dim=config.layer3.latent_dim,
            hidden_dim=config.layer3.hidden_dim,
            num_heads=config.layer3.num_heads,
            num_layers=config.layer3.num_layers,
            dropout=config.layer3.dropout
        )

        # ---- Cross-Layer Attention (跨层耦合) ----
        cl = config.cross_layer
        # Layer1 <--> Layer2
        self.cross_attn_12 = BidirectionalCrossLayerAttention(
            dim_a=config.layer1.latent_dim,
            dim_b=config.layer2.latent_dim,
            hidden_dim=cl.hidden_dim,
            num_heads=cl.num_heads,
            dropout=cl.dropout,
            coupling_strength=cl.coupling_strength
        )
        # Layer2 <--> Layer3
        self.cross_attn_23 = BidirectionalCrossLayerAttention(
            dim_a=config.layer2.latent_dim,
            dim_b=config.layer3.latent_dim,
            hidden_dim=cl.hidden_dim,
            num_heads=cl.num_heads,
            dropout=cl.dropout,
            coupling_strength=cl.coupling_strength
        )
        # Layer1 <--> Layer3
        self.cross_attn_13 = BidirectionalCrossLayerAttention(
            dim_a=config.layer1.latent_dim,
            dim_b=config.layer3.latent_dim,
            hidden_dim=cl.hidden_dim,
            num_heads=cl.num_heads,
            dropout=cl.dropout,
            coupling_strength=cl.coupling_strength
        )

        # ---- Diffusion Processes (每层独立) ----
        diff_cfg = config.diffusion
        self.diffusion_l1 = GaussianDiffusion(
            latent_dim=config.layer1.latent_dim,
            timesteps=diff_cfg.num_timesteps,
            beta_schedule=diff_cfg.beta_schedule,
            loss_type=diff_cfg.loss_type,
            objective=diff_cfg.objective,
            self_condition=diff_cfg.self_condition
        )
        self.diffusion_l2 = GaussianDiffusion(
            latent_dim=config.layer2.latent_dim,
            timesteps=diff_cfg.num_timesteps,
            beta_schedule=diff_cfg.beta_schedule,
            loss_type=diff_cfg.loss_type,
            objective=diff_cfg.objective,
            self_condition=diff_cfg.self_condition
        )
        self.diffusion_l3 = GaussianDiffusion(
            latent_dim=config.layer3.latent_dim,
            timesteps=diff_cfg.num_timesteps,
            beta_schedule=diff_cfg.beta_schedule,
            loss_type=diff_cfg.loss_type,
            objective=diff_cfg.objective,
            self_condition=diff_cfg.self_condition
        )

    def reparameterize(self, mean, logvar):
        """VAE 重参数化技巧"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def encode_all_layers(self, batch: Dict) -> Dict:
        """编码所有层到潜在空间"""
        mask = batch['seq_mask']

        # Layer 1: Sequence -> latent
        z1_mean, z1_logvar = self.encoder_l1(
            batch['seq_tokens'], mask)
        z1 = self.reparameterize(z1_mean, z1_logvar)

        # Layer 2: Secondary Structure -> latent（全局单标签 H=0/E=1）
        z2_mean, z2_logvar = self.encoder_l2(
            batch['ss_label'], mask)   # ss_label: [B] 标量
        z2 = self.reparameterize(z2_mean, z2_logvar)

        # Layer 3: Physicochemical -> latent
        z3_mean, z3_logvar = self.encoder_l3(
            batch['global_properties'],
            batch['residue_properties'], mask)
        z3 = self.reparameterize(z3_mean, z3_logvar)

        return {
            'z1': z1, 'z1_mean': z1_mean, 'z1_logvar': z1_logvar,
            'z2': z2, 'z2_mean': z2_mean, 'z2_logvar': z2_logvar,
            'z3': z3, 'z3_mean': z3_mean, 'z3_logvar': z3_logvar,
            'mask': mask
        }

    def apply_cross_layer_attention(self, z1, z2, z3, mask=None):
        """应用跨层 Attention 耦合"""
        # L1 <--> L2
        z1, z2 = self.cross_attn_12(z1, z2, mask, mask)
        # L2 <--> L3
        z2, z3 = self.cross_attn_23(z2, z3, mask, mask)
        # L1 <--> L3
        z1, z3 = self.cross_attn_13(z1, z3, mask, mask)

        return z1, z2, z3

    def training_step(self, batch: Dict) -> Dict:
        """
        训练步骤: 编码 -> 跨层耦合 -> 独立扩散损失
        """
        # 1. Encode to latent
        latents = self.encode_all_layers(batch)
        z1, z2, z3 = latents['z1'], latents['z2'], latents['z3']
        mask = latents['mask']

        # 2. Cross-layer attention coupling
        z1_coupled, z2_coupled, z3_coupled = \
            self.apply_cross_layer_attention(z1, z2, z3, mask)

        # 3. Independent diffusion loss per layer
        B = z1.shape[0]
        device = z1.device

        # 随机时间步 (每层可以独立采样不同的 t)
        t1 = torch.randint(0, self.config.diffusion.num_timesteps,
                           (B,), device=device)
        t2 = torch.randint(0, self.config.diffusion.num_timesteps,
                           (B,), device=device)
        t3 = torch.randint(0, self.config.diffusion.num_timesteps,
                           (B,), device=device)

        # 扩散损失
        loss_l1 = self.diffusion_l1.compute_loss(
            self.denoise_l1, z1_coupled, t1)
        loss_l2 = self.diffusion_l2.compute_loss(
            self.denoise_l2, z2_coupled, t2)
        loss_l3 = self.diffusion_l3.compute_loss(
            self.denoise_l3, z3_coupled, t3)

        # 4. 重建损失 (Decoder)
        t_zero = torch.zeros(B, device=device, dtype=torch.long)

        seq_logits = self.decoder_l1(z1_coupled, t_zero, mask)
        loss_recon_seq = F.cross_entropy(
            seq_logits.reshape(-1, seq_logits.size(-1)),
            batch['seq_tokens'].reshape(-1),
            ignore_index=0  # PAD
        )

        # ss_logits: [B, num_ss_types]（全局分类）
        # ss_label:  [B] 标量
        ss_logits = self.decoder_l2(z2_coupled, t_zero, mask)
        loss_recon_ss = F.cross_entropy(ss_logits, batch['ss_label'])

        global_pred, residue_pred = self.decoder_l3(
            z3_coupled, t_zero, mask)
        loss_recon_prop = (
            F.mse_loss(global_pred, batch['global_properties']) +
            F.mse_loss(residue_pred, batch['residue_properties'])
        )

        # 5. KL 散度
        kl_loss = 0
        for key in ['z1', 'z2', 'z3']:
            mean = latents[f'{key}_mean']
            logvar = latents[f'{key}_logvar']
            kl = -0.5 * torch.sum(
                1 + logvar - mean.pow(2) - logvar.exp()
            ) / B
            kl_loss = kl_loss + kl

        # 6. Total loss
        total_loss = (
            loss_l1 + loss_l2 + loss_l3 +             # 扩散损失
            loss_recon_seq + loss_recon_ss + loss_recon_prop +  # 重建
            0.001 * kl_loss                             # KL (β-VAE)
        )

        return {
            'total_loss': total_loss,
            'diffusion_l1': loss_l1.item(),
            'diffusion_l2': loss_l2.item(),
            'diffusion_l3': loss_l3.item(),
            'recon_seq': loss_recon_seq.item(),
            'recon_ss': loss_recon_ss.item(),
            'recon_prop': loss_recon_prop.item(),
            'kl_loss': kl_loss.item() if isinstance(kl_loss, torch.Tensor) else kl_loss
        }

    @torch.no_grad()
    def generate(
        self,
        num_samples: int = 10,
        seq_len: int = 30,
        guidance_scale: float = 3.0,
        fix_layer1: bool = False,
        fix_layer2: bool = False,
        fix_layer3: bool = False,
        fixed_z1: Optional[torch.Tensor] = None,
        fixed_z2: Optional[torch.Tensor] = None,
        fixed_z3: Optional[torch.Tensor] = None,
        device: str = 'cuda'
    ) -> Dict:
        """
        生成抗菌肽

        可单独控制某一层:
        - fix_layer1=True: 固定序列, 只优化结构和物化性质
        - fix_layer2=True: 固定结构, 只优化序列和物化性质
        - fix_layer3=True: 固定物化性质, 只优化序列和结构
        """
        self.eval()

        L = seq_len

        # 初始化或使用固定的潜在表示
        if fix_layer1 and fixed_z1 is not None:
            z1 = fixed_z1.to(device)
        else:
            shape1 = (num_samples, L, self.config.layer1.latent_dim)
            z1 = self.diffusion_l1.sample(
                self.denoise_l1, shape1,
                guidance_scale=guidance_scale
            )

        if fix_layer2 and fixed_z2 is not None:
            z2 = fixed_z2.to(device)
        else:
            shape2 = (num_samples, L, self.config.layer2.latent_dim)
            z2 = self.diffusion_l2.sample(
                self.denoise_l2, shape2,
                guidance_scale=guidance_scale
            )

        if fix_layer3 and fixed_z3 is not None:
            z3 = fixed_z3.to(device)
        else:
            shape3 = (num_samples, L, self.config.layer3.latent_dim)
            z3 = self.diffusion_l3.sample(
                self.denoise_l3, shape3,
                guidance_scale=guidance_scale
            )

        # 跨层耦合 (生成后融合)
        z1, z2, z3 = self.apply_cross_layer_attention(z1, z2, z3)

        # 解码
        t_zero = torch.zeros(num_samples, device=device, dtype=torch.long)
        mask = torch.ones(num_samples, L, device=device)

        seq_logits = self.decoder_l1(z1, t_zero, mask)
        seq_tokens = seq_logits.argmax(dim=-1)

        # decoder_l2 输出全局分类 logits [B, num_ss_types]
        ss_logits = self.decoder_l2(z2, t_zero, mask)
        ss_label = ss_logits.argmax(dim=-1)   # [B] 全局标签，0=H，1=E

        global_props, residue_props = self.decoder_l3(z3, t_zero, mask)


        '''# 解码
        t_zero = torch.zeros(num_samples, device=device, dtype=torch.long)
        mask = torch.ones(num_samples, seq_len, device=device)

        seq_logits = self.decoder_l1(z1, t_zero, mask)
        use_sampling= True
        temperature = 0.2
        # ✅ 修改解码逻辑
        if use_sampling and temperature > 0:
            # 温度采样
            scaled_logits = seq_logits / temperature
            probs = F.softmax(scaled_logits, dim=-1)
            seq_tokens = torch.multinomial(
                probs.view(-1, probs.size(-1)),
                num_samples=1
            ).view(num_samples, seq_len)
        else:
            # 原始argmax
            seq_tokens = seq_logits.argmax(dim=-1)

        # 其他解码保持不变
        ss_logits = self.decoder_l2(z2, t_zero, mask)
        ss_label = ss_logits.argmax(dim=-1)

        global_props, residue_props = self.decoder_l3(z3, t_zero, mask)
        '''
        return {
            'seq_tokens': seq_tokens,
            'ss_label': ss_label,             # [B] 全局二级结构标签
            'global_properties': global_props,
            'residue_properties': residue_props,
            'z1': z1, 'z2': z2, 'z3': z3
        }
