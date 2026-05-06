"""
Training Script for Multi-Scale Conditional Diffusion AMP Generator
===================================================================
Features:
  - Exponential Moving Average (EMA) of model weights
  - Gradient clipping
  - Cosine annealing with warmup
  - WandB / TensorBoard logging
  - Checkpoint save & resume
"""
import os
import sys
import time
import argparse
import copy
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import load_config
from data.dataset import build_dataloaders
from models.multi_scale_diffusion import MultiScaleConditionalDiffusion


class EMA:
    """指数移动平均 (Exponential Moving Average)"""

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    self.decay * self.shadow[name] +
                    (1 - self.decay) * param.data
                )

    def apply_shadow(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    """带 warmup 的余弦退火学习率调度器"""
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(
            max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


def train(config_path: str = None):
    """主训练函数"""
    # ---- 加载配置 ----
    config = load_config(config_path)
    tc = config.train

    # 设置随机种子
    torch.manual_seed(tc.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(tc.seed)

    device = torch.device(tc.device if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")

    # ---- 创建目录 ----
    os.makedirs(tc.checkpoint_dir, exist_ok=True)
    os.makedirs(tc.log_dir, exist_ok=True)

    # ---- 数据 ----
    print("📊 Loading data...")
    train_loader, val_loader, _ = build_dataloaders(config)
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")

    # ---- 模型 ----
    print("🧬 Building model...")
    model = MultiScaleConditionalDiffusion(config).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters()
                        if p.requires_grad)
    print(f"   Total params: {num_params:,}")
    print(f"   Trainable params: {num_trainable:,}")

    # ---- 优化器 ----
    optimizer = AdamW(
        model.parameters(),
        lr=tc.learning_rate,
        weight_decay=tc.weight_decay
    )

    total_steps = len(train_loader) * tc.num_epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, tc.warmup_steps, total_steps
    )

    # ---- EMA ----
    ema = EMA(model, decay=tc.ema_decay)

    # ---- Logger ----
    writer = SummaryWriter(tc.log_dir)

    if tc.use_wandb:
        import wandb
        wandb.init(project=tc.project_name, config=config.__dict__)

    # ---- Training Loop ----
    print("\n🚀 Starting training...")
    global_step = 0
    best_val_loss = float('inf')

    for epoch in range(1, tc.num_epochs + 1):
        model.train()
        epoch_losses = {}
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{tc.num_epochs}")

        for batch in pbar:
            # Move to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward
            losses = model.training_step(batch)
            total_loss = losses['total_loss']

            # Backward
            optimizer.zero_grad()
            total_loss.backward()

            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), tc.grad_clip_norm
            )

            optimizer.step()
            scheduler.step()
            ema.update(model)
            global_step += 1

            # Accumulate losses
            for k, v in losses.items():
                if k not in epoch_losses:
                    epoch_losses[k] = []
                val = v.item() if isinstance(v, torch.Tensor) else v
                epoch_losses[k].append(val)

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{total_loss.item():.4f}",
                'lr': f"{scheduler.get_last_lr()[0]:.2e}"
            })

            # Log
            if global_step % tc.log_every == 0:
                writer.add_scalar('train/total_loss',
                                  total_loss.item(), global_step)
                writer.add_scalar('train/grad_norm',
                                  grad_norm.item(), global_step)
                writer.add_scalar('train/lr',
                                  scheduler.get_last_lr()[0], global_step)

                for k, v in losses.items():
                    if k != 'total_loss':
                        val = v.item() if isinstance(v, torch.Tensor) else v
                        writer.add_scalar(f'train/{k}', val, global_step)

        # ---- Epoch Summary ----
        avg_losses = {
            k: sum(v) / len(v) for k, v in epoch_losses.items()
        }
        print(f"\n📈 Epoch {epoch} Summary:")
        for k, v in avg_losses.items():
            print(f"   {k}: {v:.4f}")

        # ---- Validation ----
        if epoch % tc.eval_every == 0:
            model.eval()
            ema.apply_shadow(model)

            val_losses = {}
            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    losses = model.training_step(batch)
                    for k, v in losses.items():
                        if k not in val_losses:
                            val_losses[k] = []
                        val = v.item() if isinstance(v, torch.Tensor) else v
                        val_losses[k].append(val)

            avg_val = {k: sum(v) / len(v) for k, v in val_losses.items()}
            print(f"\n📊 Validation:")
            for k, v in avg_val.items():
                print(f"   {k}: {v:.4f}")
                writer.add_scalar(f'val/{k}', v, epoch)

            # Save best model
            if avg_val['total_loss'] < best_val_loss:
                best_val_loss = avg_val['total_loss']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'ema_shadow': ema.shadow,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                    'config': config
                }, os.path.join(tc.checkpoint_dir, 'best_model.pt'))
                print(f"   ✅ Best model saved! (val_loss={best_val_loss:.4f})")

            ema.restore(model)

        # ---- Periodic Checkpoint ----
        if epoch % tc.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'ema_shadow': ema.shadow,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'config': config
            }, os.path.join(tc.checkpoint_dir, f'checkpoint_epoch{epoch}.pt'))

    writer.close()
    print("\n🎉 Training complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='C:/Users/Administrator/Desktop/amp_multi_layer_diffusion_new/configs/default.yaml',
                        help='Path to YAML config file')
    args = parser.parse_args()
    train(args.config)
