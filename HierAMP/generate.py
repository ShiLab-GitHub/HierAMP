"""
Generation Script for Multi-Scale Conditional Diffusion AMP Generator
=====================================================================
Features:
  - Unconditional generation
  - Conditional generation (target structure / properties)
  - Layer-fixed generation (e.g., fix structure, optimize sequence)
  - Result analysis & visualization
"""
import matplotlib
matplotlib.use('TkAgg')
import os
import argparse
import json
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import load_config, FullConfig
from data.dataset import (
    AminoAcidTokenizer, SecondaryStructureEncoder,
    PhysicochemicalCalculator
)
from models.multi_scale_diffusion import MultiScaleConditionalDiffusion


def load_model(checkpoint_path: str, device: str = 'cuda'):
    """加载训练好的模型"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']

    model = MultiScaleConditionalDiffusion(config).to(device)

    # 加载 EMA 权重
    if 'ema_shadow' in checkpoint:
        for name, param in model.named_parameters():
            if name in checkpoint['ema_shadow']:
                param.data = checkpoint['ema_shadow'][name].to(device)
    else:
        model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    return model, config


def generate_unconditional(
    model, config, num_samples=2593,
    seq_len=30, guidance_scale=3.0, device='cuda'
):
    """无条件生成"""
    print(f"🧬 Generating {num_samples} unconditional AMP sequences...")

    results = model.generate(
        num_samples=num_samples,
        seq_len=seq_len,
        guidance_scale=guidance_scale,
        device=device
    )

    return results


def generate_with_fixed_structure(
    model, config, target_ss: str,
    num_samples=50, seq_len=50, guidance_scale=3.0, device='cuda'
):
    """
    固定全局二级结构类型, 只生成序列和物化性质。
    target_ss: 'H' (α-helix) 或 'E' (β-sheet)
    """
    target_ss = target_ss.strip().upper()
    assert target_ss in ('H', 'E'),         f"target_ss 只接受 'H' 或 'E'，收到: {target_ss!r}"
    print(f"🔒 Generating with fixed structure type: {target_ss} "
          f"({'α-helix' if target_ss == 'H' else 'β-sheet'})")

    # 将全局标签编码为 [B] 整数 tensor
    ss_data_encoder = SecondaryStructureEncoder()   # 数据层编码器
    ss_label_single = ss_data_encoder.encode(target_ss)          # 标量 tensor
    ss_labels = ss_label_single.unsqueeze(0).repeat(num_samples) # [B]
    ss_labels = ss_labels.to(device)
    print(num_samples)
    print(seq_len)
    # 用模型的 Layer-2 编码器把全局标签编码到潜在空间
    with torch.no_grad():
        mask = torch.ones(int(num_samples), int(seq_len), device=device)
        z2_mean, z2_logvar = model.encoder_l2(ss_labels, mask)
        z2_fixed = z2_mean   # 使用均值（确定性）

    results = model.generate(
        num_samples=num_samples,
        seq_len=seq_len,
        guidance_scale=guidance_scale,
        fix_layer2=True,
        fixed_z2=z2_fixed,
        device=device
    )

    return results


def generate_with_target_properties(
    model, config,
    target_charge: float = None,
    target_hydrophobicity: float = None,
    num_samples=50, seq_len=50,
    guidance_scale=3.0, device='cuda'
):
    """
    固定目标物化性质, 生成满足条件的序列
    """
    print(f"🎯 Generating with target properties:")
    if target_charge is not None:
        print(f"   Charge: {target_charge}")
    if target_hydrophobicity is not None:
        print(f"   Hydrophobicity: {target_hydrophobicity}")

    # 构建目标物化性质向量
    target_props = torch.zeros(8)
    if target_charge is not None:
        target_props[0] = target_charge / 10.0
    if target_hydrophobicity is not None:
        target_props[1] = target_hydrophobicity / 4.5

    target_props = target_props.unsqueeze(0).repeat(num_samples, 1).to(device)
    residue_props = torch.zeros(num_samples, seq_len, 3).to(device)

    with torch.no_grad():
        mask = torch.ones(num_samples, seq_len, device=device)
        z3_mean, z3_logvar = model.encoder_l3(target_props, residue_props, mask)
        z3_fixed = z3_mean

    results = model.generate(
        num_samples=num_samples,
        seq_len=seq_len,
        guidance_scale=guidance_scale,
        fix_layer3=True,
        fixed_z3=z3_fixed,
        device=device
    )

    return results


def analyze_results(results, config, output_dir='results'):
    """分析和可视化生成结果"""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'figures'), exist_ok=True)

    tokenizer = AminoAcidTokenizer(max_len=results['seq_tokens'].shape[1])
    physchem = PhysicochemicalCalculator()

    # 解码序列
    sequences = []
    for tokens in results['seq_tokens']:
        seq = tokenizer.decode(tokens)
        sequences.append(seq)

    # 解码全局二级结构标签（ss_label: [B] 整数向量）
    structures = []
    for label in results['ss_label']:           # label: 标量 tensor
        ss = SecondaryStructureEncoder.decode(label)   # 返回 'H' 或 'E'
        structures.append(ss)

    # 计算物化性质
    properties = []
    for seq in sequences:
        if len(seq) > 0:
            props = physchem.compute_properties(seq)
            properties.append({
                'sequence': seq,
                'length': len(seq),
                'net_charge': props[0].item() * 10,
                'hydrophobicity': props[1].item() * 4.5,
                'amphipathicity': props[2].item() * 3.0,
                'molecular_weight': props[3].item() * 5000,
            })

    df = pd.DataFrame(properties)

    # 保存到 CSV
    df.to_csv(os.path.join(output_dir, 'generated_amps_second_structure_alpha_len50.csv'), index=False)
    print(f"\n📄 Saved {len(df)} sequences to {output_dir}/generated_amps.csv")

    # ---- 可视化 ----
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Generated AMP Analysis', fontsize=16)

    # 1. 序列长度分布
    axes[0, 0].hist(df['length'], bins=20, color='steelblue', edgecolor='white')
    axes[0, 0].set_xlabel('Sequence Length')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Length Distribution')

    # 2. 净电荷分布
    axes[0, 1].hist(df['net_charge'], bins=20, color='coral', edgecolor='white')
    axes[0, 1].set_xlabel('Net Charge')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Charge Distribution')

    # 3. 疏水性 vs 两亲性
    axes[1, 0].scatter(df['hydrophobicity'], df['amphipathicity'],
                       alpha=0.5, c='seagreen', s=20)
    axes[1, 0].set_xlabel('Hydrophobicity')
    axes[1, 0].set_ylabel('Amphipathicity')
    axes[1, 0].set_title('Hydrophobicity vs Amphipathicity')

    # 4. 氨基酸组成
    all_seqs = ''.join(sequences)
    aa_counts = {aa: all_seqs.count(aa) for aa in 'ACDEFGHIKLMNPQRSTVWY'}
    total = sum(aa_counts.values()) or 1
    aa_freq = {k: v / total for k, v in aa_counts.items()}
    axes[1, 1].bar(aa_freq.keys(), aa_freq.values(), color='mediumpurple')
    axes[1, 1].set_xlabel('Amino Acid')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Amino Acid Composition')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'figures', 'analysis.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # 统计摘要
    print("\n📊 Generation Statistics:")
    print(f"   Sequences: {len(sequences)}")
    print(f"   Avg length: {df['length'].mean():.1f} ± {df['length'].std():.1f}")
    print(f"   Avg charge: {df['net_charge'].mean():.2f} ± {df['net_charge'].std():.2f}")
    print(f"   Avg hydrophobicity: {df['hydrophobicity'].mean():.3f}")

    # 序列多样性
    unique_seqs = len(set(sequences))
    print(f"   Unique sequences: {unique_seqs}/{len(sequences)} "
          f"({unique_seqs/len(sequences)*100:.1f}%)")

    return df


def main():
    parser = argparse.ArgumentParser(
        description='Generate AMPs with Multi-Scale Diffusion')
    parser.add_argument('--checkpoint', type=str, default='C:/Users/Administrator/Desktop/amp_multi_layer_diffusion_new/checkpoints/best_model.pt')
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--seq_len', type=int, default=50)
    parser.add_argument('--guidance_scale', type=float, default=3.0)
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--device', type=str, default='cuda')

    # Layer control
    parser.add_argument('--fix_structure', action='store_true',default=True,
                        help='Fix secondary structure layer')
    parser.add_argument('--target_ss', type=str, default='H',
                        help='Target secondary structure string')
    parser.add_argument('--target_charge', type=float, default=None)
    parser.add_argument('--target_hydro', type=float, default=None)

    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # 加载模型
    model, config = load_model(args.checkpoint, device)

    # 生成
    if args.fix_structure and args.target_ss:
        results = generate_with_fixed_structure(
            model, config, args.target_ss,
            args.num_samples, args.seq_len, args.guidance_scale, device
        )
    elif args.target_charge or args.target_hydro:
        results = generate_with_target_properties(
            model, config, args.target_charge, args.target_hydro,
            args.num_samples, args.seq_len, args.guidance_scale, device
        )
    else:
        results = generate_unconditional(
            model, config, args.num_samples,
            args.seq_len, args.guidance_scale, device
        )

    # 分析
    df = analyze_results(results, config, args.output_dir)

    # 打印前 10 个生成的序列
    print("\n🧬 Top 10 Generated AMPs:")
    for i, row in df.head(10).iterrows():
        print(f"   {i+1}. {row['sequence']}")
        print(f"      Length={row['length']}, Charge={row['net_charge']:.1f}, "
              f"Hydro={row['hydrophobicity']:.3f}")


if __name__ == '__main__':
    main()
