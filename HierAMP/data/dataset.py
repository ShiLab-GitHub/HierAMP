"""
AMP Dataset Loading & Preprocessing
CSV columns: name, seq, source, type, second_structure, variant_type
"""
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

# ============================================================
# 氨基酸物化性质表
# ============================================================
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

# Kyte-Doolittle 疏水性量表
HYDROPHOBICITY = {
    'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
    'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
    'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
    'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
}

# 氨基酸电荷 (pH 7.0)
CHARGE = {
    'A': 0, 'C': 0, 'D': -1, 'E': -1, 'F': 0,
    'G': 0, 'H': 0.1, 'I': 0, 'K': 1, 'L': 0,
    'M': 0, 'N': 0, 'P': 0, 'Q': 0, 'R': 1,
    'S': 0, 'T': 0, 'V': 0, 'W': 0, 'Y': 0
}

# 氨基酸分子量
MOLECULAR_WEIGHT = {
    'A': 89.1, 'C': 121.2, 'D': 133.1, 'E': 147.1, 'F': 165.2,
    'G': 75.0, 'H': 155.2, 'I': 131.2, 'K': 146.2, 'L': 131.2,
    'M': 149.2, 'N': 132.1, 'P': 115.1, 'Q': 146.2, 'R': 174.2,
    'S': 105.1, 'T': 119.1, 'V': 117.1, 'W': 204.2, 'Y': 181.2
}

# 二级结构倾向性
SS_PROPENSITY = {
    'H': {'A': 1.42, 'C': 0.70, 'D': 1.01, 'E': 1.51, 'F': 1.13,
          'G': 0.57, 'H': 1.00, 'I': 1.08, 'K': 1.16, 'L': 1.21,
          'M': 1.45, 'N': 0.67, 'P': 0.57, 'Q': 1.11, 'R': 0.98,
          'S': 0.77, 'T': 0.83, 'V': 1.06, 'W': 1.08, 'Y': 0.69},
    'E': {'A': 0.83, 'C': 1.19, 'D': 0.54, 'E': 0.37, 'F': 1.38,
          'G': 0.75, 'H': 0.87, 'I': 1.60, 'K': 0.74, 'L': 1.30,
          'M': 1.05, 'N': 0.89, 'P': 0.55, 'Q': 1.10, 'R': 0.93,
          'S': 0.75, 'T': 1.19, 'V': 1.70, 'W': 1.37, 'Y': 1.47},
}


class AminoAcidTokenizer:
    """氨基酸序列编码器"""

    def __init__(self, max_len: int = 50):
        self.max_len = max_len
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"

        # 构建词汇表: PAD=0, UNK=1, A=2, C=3, ...
        self.vocab = {self.pad_token: 0, self.unk_token: 1}
        for i, aa in enumerate(AMINO_ACIDS):
            self.vocab[aa] = i + 2
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)

    def encode(self, seq: str) -> torch.Tensor:
        """将氨基酸序列编码为整数 tensor"""
        tokens = []
        for aa in seq.upper():
            tokens.append(self.vocab.get(aa, self.vocab[self.unk_token]))
        # Padding / Truncation
        if len(tokens) > self.max_len:
            tokens = tokens[:self.max_len]
        else:
            tokens += [self.vocab[self.pad_token]] * (self.max_len - len(tokens))
        return torch.tensor(tokens, dtype=torch.long)

    def decode(self, tokens: torch.Tensor) -> str:
        """将整数 tensor 解码为氨基酸序列"""
        seq = []
        for t in tokens.cpu().numpy():
            if t == 0:  # PAD
                break
            seq.append(self.inv_vocab.get(int(t), 'X'))
        return ''.join(seq)


class SecondaryStructureEncoder:
    """
    二级结构编码器 —— 全局单标签版本
    second_structure 列只有两种取值：
      'H'  →  α-helix  (label = 0)
      'E'  →  β-sheet  (label = 1)
    缺失/无效值默认归为 H (label = 0)
    """

    SS_MAP = {'H': 0, 'E': 1}

    def encode(self, ss_label: str) -> torch.Tensor:
        """
        将单字符全局标签编码为整数标量 tensor。
        Args:
            ss_label: 'H' 或 'E'
        Returns:
            shape [] 的 torch.long 标量
        """
    #    if pd.isna(ss_label) or str(ss_label).strip() == '':
   #         label = 0  # 缺失时默认 H
     #   else:
        label = self.SS_MAP.get(str(ss_label).strip().upper(), 0)
        return torch.tensor(label, dtype=torch.long)

    @staticmethod
    def decode(label_tensor: torch.Tensor) -> str:
        """将整数标量 tensor 解码回字符标签。"""
        inv_map = {0: 'H', 1: 'E'}
        return inv_map.get(int(label_tensor.item()), 'H')


class PhysicochemicalCalculator:
    """物化性质计算器"""

    @staticmethod
    def compute_properties(seq: str) -> torch.Tensor:
        """计算序列的 8 维物化性质向量"""
        seq = seq.upper()
        n = len(seq)
        if n == 0:
            return torch.zeros(8, dtype=torch.float32)

        # 1. 净电荷
        net_charge = sum(CHARGE.get(aa, 0) for aa in seq)

        # 2. 平均疏水性
        hydrophobicity = np.mean([HYDROPHOBICITY.get(aa, 0) for aa in seq])

        # 3. 两亲性 (基于螺旋轮投影，假设 α-helix, 100°/残基)
        angle_per_residue = 100.0 * np.pi / 180.0
        hx = sum(HYDROPHOBICITY.get(aa, 0) * np.cos(i * angle_per_residue)
                 for i, aa in enumerate(seq))
        hy = sum(HYDROPHOBICITY.get(aa, 0) * np.sin(i * angle_per_residue)
                 for i, aa in enumerate(seq))
        amphipathicity = np.sqrt(hx**2 + hy**2) / n

        # 4. 分子量
        mw = sum(MOLECULAR_WEIGHT.get(aa, 110) for aa in seq) - 18.02 * (n - 1)

        # 5. 等电点 (简化计算)
        positive = sum(1 for aa in seq if aa in 'KRH')
        negative = sum(1 for aa in seq if aa in 'DE')
        pI = 7.0 + (positive - negative) * 0.5  # 粗略估计

        # 6. 不稳定性指数 (简化)
        instability = sum(1 for aa in seq if aa in 'MSDN') / n * 100

        # 7. 芳香性
        aromaticity = sum(1 for aa in seq if aa in 'FWY') / n

        # 8. 脂肪族指数
        ala = seq.count('A') / n
        val = seq.count('V') / n
        ile = seq.count('I') / n
        leu = seq.count('L') / n
        aliphatic_index = 100 * ala + 2.9 * 100 * val + 3.9 * 100 * (ile + leu)

        # 归一化
        properties = torch.tensor([
            net_charge / 10.0,           # 归一化到 ~[-1, 1]
            hydrophobicity / 4.5,        # Kyte-Doolittle 范围
            amphipathicity / 3.0,
            mw / 5000.0,                 # 归一化分子量
            (pI - 7.0) / 7.0,
            instability / 100.0,
            aromaticity,
            aliphatic_index / 300.0
        ], dtype=torch.float32)

        return properties

    @staticmethod
    def compute_per_residue(seq: str, max_len: int = 50) -> torch.Tensor:
        """计算每个残基的物化性质 [max_len, 3]"""
        features = []
        for aa in seq.upper():
            features.append([
                CHARGE.get(aa, 0),
                HYDROPHOBICITY.get(aa, 0) / 4.5,
                MOLECULAR_WEIGHT.get(aa, 110) / 204.2
            ])

        # Pad / Truncate
        while len(features) < max_len:
            features.append([0.0, 0.0, 0.0])
        features = features[:max_len]

        return torch.tensor(features, dtype=torch.float32)


class AMPDataset(Dataset):
    """
    AMP 数据集
    CSV columns: name, seq, source, type, second_structure, variant_type
    """

    def __init__(
        self,
        csv_path: str,
        max_seq_len: int = 50,
        split: str = 'train',
        train_ratio: float = 0.85,
        val_ratio: float = 0.10,
        seed: int = 42
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.tokenizer = AminoAcidTokenizer(max_len=max_seq_len)
        self.ss_encoder = SecondaryStructureEncoder()   # 无需 max_len，单标签
        self.physchem = PhysicochemicalCalculator()

        # 读取并清洗数据
        df = pd.read_csv(csv_path)
        df = df.dropna(subset=['sequence'])
        df = df[df['sequence'].str.len().between(5, max_seq_len)]
        df = df.reset_index(drop=True)

        # 划分数据集
        np.random.seed(seed)
        indices = np.random.permutation(len(df))
        n_train = int(len(df) * train_ratio)
        n_val = int(len(df) * val_ratio)

        if split == 'train':
            self.df = df.iloc[indices[:n_train]].reset_index(drop=True)
        elif split == 'val':
            self.df = df.iloc[indices[n_train:n_train+n_val]].reset_index(drop=True)
        else:
            self.df = df.iloc[indices[n_train+n_val:]].reset_index(drop=True)

        # 构建类型编码映射
        all_types = df['type'].dropna().unique().tolist()
        self.type_to_idx = {t: i for i, t in enumerate(all_types)}

        all_variants = df['variant_type'].dropna().unique().tolist() \
            if 'variant_type' in df.columns else []
        self.variant_to_idx = {v: i for i, v in enumerate(all_variants)}

        print(f"[{split}] Loaded {len(self.df)} sequences, "
              f"{len(all_types)} AMP types, {len(all_variants)} variant types")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        seq = str(row['sequence']).upper()

        # Layer 1: 序列编码
        seq_tokens = self.tokenizer.encode(seq)
        seq_mask = (seq_tokens != 0).float()  # [max_len]

        # Layer 2: 二级结构编码（全局单标签：H=0, E=1）
        ss_raw = row.get('second_structure', '')
        ss_label = self.ss_encoder.encode(ss_raw)   # shape: [] (标量)

        # Layer 3: 物化性质
        global_props = self.physchem.compute_properties(seq)  # [8]
        residue_props = self.physchem.compute_per_residue(seq, self.max_seq_len)

        # 条件标签
        amp_type = self.type_to_idx.get(row.get('type', ''), 0)
        variant_type = self.variant_to_idx.get(
            row.get('variant_type', ''), 0)

        return {
            # Layer 1 data
            'seq_tokens': seq_tokens,            # [max_len]
            'seq_mask': seq_mask,                 # [max_len]
            'seq_len': torch.tensor(len(seq), dtype=torch.long),

            # Layer 2 data
            'ss_label': ss_label,                # [] 标量，H=0 / E=1

            # Layer 3 data
            'global_properties': global_props,   # [8]
            'residue_properties': residue_props,  # [max_len, 3]

            # Condition labels
            'amp_type': torch.tensor(amp_type, dtype=torch.long),
            'variant_type': torch.tensor(variant_type, dtype=torch.long),
        }


def build_dataloaders(config) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """构建训练/验证/测试数据加载器"""
    train_ds = AMPDataset(
        config.data.csv_path,
        max_seq_len=config.data.max_seq_len,
        split='train',
        train_ratio=config.data.train_ratio,
        val_ratio=config.data.val_ratio,
        seed=config.train.seed
    )
    val_ds = AMPDataset(
        config.data.csv_path,
        max_seq_len=config.data.max_seq_len,
        split='val',
        train_ratio=config.data.train_ratio,
        val_ratio=config.data.val_ratio,
        seed=config.train.seed
    )
    test_ds = AMPDataset(
        config.data.csv_path,
        max_seq_len=config.data.max_seq_len,
        split='test',
        train_ratio=config.data.train_ratio,
        val_ratio=config.data.val_ratio,
        seed=config.train.seed
    )

    train_loader = DataLoader(
        train_ds, batch_size=config.train.batch_size,
        shuffle=True, num_workers=config.data.num_workers,
        pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.train.batch_size,
        shuffle=False, num_workers=config.data.num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=config.train.batch_size,
        shuffle=False, num_workers=config.data.num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader
