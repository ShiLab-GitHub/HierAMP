#!/bin/bash
# ============================================================
# Multi-Scale Conditional Diffusion AMP Generator
# Environment Setup Script
# ============================================================

echo "🧬 Setting up Multi-Scale Conditional Diffusion AMP Generator..."

# 1. Create conda environment
conda create -n amp_diffusion python=3.10 -y
conda activate amp_diffusion

# 2. Install PyTorch (CUDA 11.8)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 3. Install other dependencies
pip install -r requirements.txt

# 4. Create project directories
mkdir -p data
mkdir -p checkpoints
mkdir -p logs
mkdir -p results/generated_sequences
mkdir -p results/figures

echo "✅ Environment setup complete!"
echo "📌 Activate with: conda activate amp_diffusion"
echo "📌 Start training: python train.py --config configs/default.yaml"
echo "📌 Generate AMPs: python generate.py --checkpoint checkpoints/best_model.pt"