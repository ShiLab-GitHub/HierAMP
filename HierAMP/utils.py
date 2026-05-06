"""
Utility Functions
- Sequence validation & filtering
- Property-based filtering
- Model statistics
- Visualization helpers
"""
import torch
import numpy as np
import re
from typing import List, Dict, Tuple, Optional
from data.dataset import PhysicochemicalCalculator, HYDROPHOBICITY, CHARGE


# ============================================================
# 序列验证与过滤
# ============================================================

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


def validate_sequence(seq: str) -> Tuple[bool, str]:
    """
    验证抗菌肽序列是否有效
    Returns: (is_valid, reason)
    """
    seq = seq.upper().strip()

    if not seq:
        return False, "Empty sequence"

    if len(seq) < 5:
        return False, f"Too short ({len(seq)} < 5)"

    if len(seq) > 100:
        return False, f"Too long ({len(seq)} > 100)"

    invalid_chars = set(seq) - VALID_AA
    if invalid_chars:
        return False, f"Invalid characters: {invalid_chars}"

    # 检查是否有过多重复
    for aa in VALID_AA:
        if aa * 5 in seq:  # 连续 5 个相同氨基酸
            return False, f"Excessive repeat of {aa}"

    return True, "Valid"


def filter_by_properties(
    sequences: List[str],
    min_charge: float = None,
    max_charge: float = None,
    min_hydrophobicity: float = None,
    max_hydrophobicity: float = None,
    min_length: int = None,
    max_length: int = None
) -> List[str]:
    """基于物化性质过滤序列"""
    calc = PhysicochemicalCalculator()
    filtered = []

    for seq in sequences:
        if min_length and len(seq) < min_length:
            continue
        if max_length and len(seq) > max_length:
            continue

        props = calc.compute_properties(seq)
        charge = props[0].item() * 10
        hydro = props[1].item() * 4.5

        if min_charge and charge < min_charge:
            continue
        if max_charge and charge > max_charge:
            continue
        if min_hydrophobicity and hydro < min_hydrophobicity:
            continue
        if max_hydrophobicity and hydro > max_hydrophobicity:
            continue

        filtered.append(seq)

    return filtered


def compute_sequence_diversity(sequences: List[str]) -> Dict:
    """计算序列集合的多样性指标"""
    n = len(sequences)
    if n == 0:
        return {'unique_ratio': 0, 'avg_similarity': 0}

    unique = len(set(sequences))
    unique_ratio = unique / n

    # 成对编辑距离 (采样计算)
    from itertools import combinations
    import random
    pairs = list(combinations(range(min(n, 100)), 2))
    if len(pairs) > 500:
        pairs = random.sample(pairs, 500)

    similarities = []
    for i, j in pairs:
        s1, s2 = sequences[i], sequences[j]
        sim = sequence_similarity(s1, s2)
        similarities.append(sim)

    return {
        'num_sequences': n,
        'unique_sequences': unique,
        'unique_ratio': unique_ratio,
        'avg_pairwise_similarity': np.mean(similarities) if similarities else 0,
        'std_pairwise_similarity': np.std(similarities) if similarities else 0
    }


def sequence_similarity(s1: str, s2: str) -> float:
    """简单序列相似度 (基于匹配位置)"""
    min_len = min(len(s1), len(s2))
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    matches = sum(1 for a, b in zip(s1, s2) if a == b)
    return matches / max_len


def count_parameters(model: torch.nn.Module) -> Dict:
    """统计模型参数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Per-module breakdown
    breakdown = {}
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        breakdown[name] = params

    return {
        'total': total,
        'trainable': trainable,
        'frozen': total - trainable,
        'breakdown': breakdown
    }


def print_model_summary(model: torch.nn.Module):
    """打印模型摘要"""
    stats = count_parameters(model)
    print("=" * 60)
    print("Model Summary")
    print("=" * 60)
    print(f"Total parameters:     {stats['total']:>12,}")
    print(f"Trainable parameters: {stats['trainable']:>12,}")
    print(f"Frozen parameters:    {stats['frozen']:>12,}")
    print("-" * 60)
    print("Module Breakdown:")
    for name, count in stats['breakdown'].items():
        pct = count / stats['total'] * 100
        print(f"  {name:30s} {count:>10,} ({pct:5.1f}%)")
    print("=" * 60)


# ============================================================
# AMP-specific Analysis
# ============================================================

def compute_amp_score(seq: str) -> float:
    """
    简化的 AMP 评分 (0-1)
    考虑: 电荷、疏水性、两亲性、长度
    """
    seq = seq.upper()
    n = len(seq)

    if n < 5 or n > 50:
        return 0.0

    # 正电荷 (AMP 通常带正电)
    charge = sum(CHARGE.get(aa, 0) for aa in seq)
    charge_score = min(max(charge / 6.0, 0), 1)  # 理想 +2~+6

    # 疏水性 (AMP 通常有适中疏水性)
    hydro = np.mean([HYDROPHOBICITY.get(aa, 0) for aa in seq])
    hydro_score = 1.0 - abs(hydro) / 4.5  # 适中最好

    # 两亲性
    angle = 100.0 * np.pi / 180.0
    hx = sum(HYDROPHOBICITY.get(aa, 0) * np.cos(i * angle)
             for i, aa in enumerate(seq))
    hy = sum(HYDROPHOBICITY.get(aa, 0) * np.sin(i * angle)
             for i, aa in enumerate(seq))
    amphi = np.sqrt(hx**2 + hy**2) / n
    amphi_score = min(amphi / 2.0, 1)

    # 长度 (理想 10-40)
    if 10 <= n <= 40:
        len_score = 1.0
    elif 5 <= n < 10 or 40 < n <= 50:
        len_score = 0.5
    else:
        len_score = 0.0

    # 加权平均
    score = (
        0.3 * charge_score +
        0.2 * hydro_score +
        0.3 * amphi_score +
        0.2 * len_score
    )

    return round(score, 3)
