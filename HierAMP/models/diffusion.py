"""
Diffusion Process Core Module
- Noise Scheduling (Linear, Cosine, Quadratic)
- Forward Process (q sampling)
- Reverse Process (denoising)
- Loss Computation
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


def linear_beta_schedule(timesteps: int, beta_start=1e-4, beta_end=0.02):
    """线性 β 调度"""
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps: int, s=0.008):
    """余弦 β 调度 (Nichol & Dhariwal, 2021)"""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.9999)


def quadratic_beta_schedule(timesteps: int, beta_start=1e-4, beta_end=0.02):
    """二次方 β 调度"""
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2


class GaussianDiffusion(nn.Module):
    """
    高斯扩散过程
    支持 pred_noise / pred_x0 / pred_v 三种目标
    """

    def __init__(
        self,
        latent_dim: int,
        timesteps: int = 1000,
        beta_schedule: str = 'cosine',
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        loss_type: str = 'huber',
        objective: str = 'pred_x0',
        self_condition: bool = True
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.timesteps = timesteps
        self.objective = objective
        self.loss_type = loss_type
        self.self_condition = self_condition

        # β schedule
        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps, beta_start, beta_end)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        elif beta_schedule == 'quadratic':
            betas = quadratic_beta_schedule(timesteps, beta_start, beta_end)
        else:
            raise ValueError(f"Unknown schedule: {beta_schedule}")

        # 预计算扩散参数
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod',
                             torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             torch.sqrt(1.0 / alphas_cumprod - 1))

        # 后验分布参数 q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped',
                             torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
                             betas * torch.sqrt(alphas_cumprod_prev) /
                             (1.0 - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1.0 - alphas_cumprod_prev) *
                             torch.sqrt(alphas) / (1.0 - alphas_cumprod))

    def _extract(self, a: torch.Tensor, t: torch.Tensor,
                 x_shape: tuple) -> torch.Tensor:
        """从预计算的张量中按 t 索引提取值"""
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        # Reshape for broadcasting
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向扩散: q(x_t | x_0) = N(√ᾱ_t * x_0, (1-ᾱ_t) * I)
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alpha = self._extract(
            self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alpha = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        x_noisy = sqrt_alpha * x_start + sqrt_one_minus_alpha * noise
        return x_noisy, noise

    def predict_start_from_noise(self, x_t, t, noise):
        """从噪声预测 x_0"""
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x_0):
        """从 x_0 预测噪声"""
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            x_0
        ) / self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def q_posterior(self, x_start, x_t, t):
        """计算后验分布 q(x_{t-1} | x_t, x_0)"""
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(
            self.posterior_variance, t, x_t.shape)
        posterior_log_variance = self._extract(
            self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance

    def compute_loss(
        self,
        denoise_fn,
        x_start: torch.Tensor,
        t: torch.Tensor,
        condition: dict = None,
        noise: torch.Tensor = None
    ) -> torch.Tensor:
        """
        计算扩散损失
        Args:
            denoise_fn: 去噪网络 (接收 x_t, t, condition)
            x_start: [B, ...] 原始数据
            t: [B] 时间步
            condition: 条件信息 dict
            noise: 预设噪声
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy, _ = self.q_sample(x_start, t, noise)

        # 自条件化: 50% 概率使用上一步预测
        x_self_cond = None
        if self.self_condition and torch.rand(1).item() < 0.5:
            with torch.no_grad():
                x_self_cond = denoise_fn(x_noisy, t, condition).detach()

        # 预测
        model_out = denoise_fn(
            x_noisy, t, condition, x_self_cond=x_self_cond
        )

        # 计算目标
        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            # v-prediction: v = √ᾱ * ε - √(1-ᾱ) * x_0
            sqrt_alpha = self._extract(
                self.sqrt_alphas_cumprod, t, x_start.shape)
            sqrt_one_minus = self._extract(
                self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            target = sqrt_alpha * noise - sqrt_one_minus * x_start
        else:
            raise ValueError(f"Unknown objective: {self.objective}")

        # 损失函数
        if self.loss_type == 'l1':
            loss = F.l1_loss(model_out, target)
        elif self.loss_type == 'l2':
            loss = F.mse_loss(model_out, target)
        elif self.loss_type == 'huber':
            loss = F.smooth_l1_loss(model_out, target)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        return loss

    @torch.no_grad()
    def p_sample(self, denoise_fn, x_t, t, condition=None,
                 x_self_cond=None, guidance_scale=1.0):
        """单步反向采样 p(x_{t-1} | x_t)"""
        b = x_t.shape[0]
        t_batch = torch.full((b,), t, device=x_t.device, dtype=torch.long)

        # 模型预测
        model_out = denoise_fn(x_t, t_batch, condition,
                               x_self_cond=x_self_cond)

        # Classifier-free guidance
        if guidance_scale > 1.0 and condition is not None:
            uncond_out = denoise_fn(x_t, t_batch, None,
                                    x_self_cond=x_self_cond)
            model_out = uncond_out + guidance_scale * (model_out - uncond_out)

        # 从预测获取 x_0
        if self.objective == 'pred_noise':
            x_start = self.predict_start_from_noise(x_t, t_batch, model_out)
        elif self.objective == 'pred_x0':
            x_start = model_out
        elif self.objective == 'pred_v':
            sqrt_alpha = self._extract(
                self.sqrt_alphas_cumprod, t_batch, x_t.shape)
            sqrt_one_minus = self._extract(
                self.sqrt_one_minus_alphas_cumprod, t_batch, x_t.shape)
            x_start = sqrt_alpha * x_t - sqrt_one_minus * model_out

        x_start = torch.clamp(x_start, -1.0, 1.0)

        # 计算后验
        model_mean, _, model_log_variance = self.q_posterior(
            x_start, x_t, t_batch)

        noise = torch.randn_like(x_t) if t > 0 else 0.0
        pred = model_mean + (0.5 * model_log_variance).exp() * noise

        return pred, x_start

    @torch.no_grad()
    def sample(self, denoise_fn, shape, condition=None,
               guidance_scale=1.0, return_intermediates=False):
        """完整反向采样 (从噪声生成数据)"""
        device = next(iter(self.buffers())).device
        b = shape[0]

        x = torch.randn(shape, device=device)
        intermediates = [x.clone()]

        x_start = None
        for t in reversed(range(self.timesteps)):
            x_self_cond = x_start if self.self_condition else None
            x, x_start = self.p_sample(
                denoise_fn, x, t, condition,
                x_self_cond=x_self_cond,
                guidance_scale=guidance_scale
            )
            if return_intermediates and t % 100 == 0:
                intermediates.append(x.clone())

        if return_intermediates:
            return x, intermediates
        return x
