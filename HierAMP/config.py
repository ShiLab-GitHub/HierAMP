"""
Multi-Scale Conditional Diffusion AMP Generator
Global Configuration
"""
import yaml
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DataConfig:
    """数据集相关配置"""
    csv_path: str = "data/amp_extended_dataset_v2.csv"
    max_seq_len: int = 50          # 最大序列长度
    min_seq_len: int = 5           # 最小序列长度
    vocab_size: int = 22           # 20标准氨基酸 + PAD + UNK
    train_ratio: float = 0.85
    val_ratio: float = 0.10
    test_ratio: float = 0.05
    num_workers: int = 4

    # 氨基酸词汇表
    amino_acids: str = "ACDEFGHIKLMNPQRSTVWY"
    pad_token: str = "<PAD>"
    unk_token: str = "<UNK>"


@dataclass
class DiffusionConfig:
    """扩散过程配置"""
    num_timesteps: int = 1000       # 扩散步数 T
    beta_start: float = 1e-4        # β 起始值
    beta_end: float = 0.02          # β 终止值
    beta_schedule: str = "cosine"   # 'linear' | 'cosine' | 'quadratic'
    loss_type: str = "huber"        # 'l1' | 'l2' | 'huber'
    self_condition: bool = True     # 自条件化
    objective: str = "pred_x0"     # 'pred_noise' | 'pred_x0' | 'pred_v'


@dataclass
class Layer1Config:
    """Layer 1: 序列 Motif 层 (局部模式)"""
    latent_dim: int = 128           # 潜在空间维度
    hidden_dim: int = 256
    num_heads: int = 8
    num_layers: int = 4
    dropout: float = 0.1
    kernel_sizes: List[int] = field(default_factory=lambda: [3, 5, 7])
    # 局部 motif 检测的卷积核大小
    motif_embed_dim: int = 64


@dataclass
class Layer2Config:
    """Layer 2: 二级结构层 (α-helix, β-sheet, loop)"""
    latent_dim: int = 64
    hidden_dim: int = 128
    num_heads: int = 4
    num_layers: int = 3
    dropout: float = 0.1
    num_ss_types: int = 2           # H(helix=0), E(sheet=1)，全局单标签
    ss_embed_dim: int = 32


@dataclass
class Layer3Config:
    """Layer 3: 物化性质层 (电荷、疏水性、两亲性)"""
    latent_dim: int = 32
    hidden_dim: int = 64
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1
    num_properties: int = 8
    # [net_charge, hydrophobicity, amphipathicity,
    #  molecular_weight, pI, instability_index,
    #  aromaticity, aliphatic_index]


@dataclass
class CrossLayerConfig:
    """跨层 Attention 耦合配置"""
    num_heads: int = 8
    hidden_dim: int = 256
    dropout: float = 0.1
    coupling_strength: float = 0.5  # 跨层耦合强度 [0,1]
    bidirectional: bool = True       # 双向耦合


@dataclass
class TrainConfig:
    """训练配置"""
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    num_epochs: int = 500
    warmup_steps: int = 1000
    grad_clip_norm: float = 1.0
    ema_decay: float = 0.9999
    save_every: int = 50
    eval_every: int = 10
    log_every: int = 100
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    device: str = "cuda"
    seed: int = 42
    use_wandb: bool = False
    project_name: str = "amp_diffusion"


@dataclass
class GenerateConfig:
    """生成配置"""
    num_samples: int = 100
    guidance_scale: float = 3.0     # Classifier-free guidance scale
    temperature: float = 1.0
    # 层级控制开关
    fix_layer1: bool = False        # 固定序列层
    fix_layer2: bool = False        # 固定结构层
    fix_layer3: bool = False        # 固定物化性质层
    # 条件输入 (可选)
    target_ss: Optional[str] = None          # 目标二级结构 e.g. "HHHHHCCCCEEEEE"
    target_charge: Optional[float] = None     # 目标净电荷
    target_hydrophobicity: Optional[float] = None
    target_amphipathicity: Optional[float] = None
    output_dir: str = "results/generated_sequences"


@dataclass
class FullConfig:
    """完整配置"""
    data: DataConfig = field(default_factory=DataConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    layer1: Layer1Config = field(default_factory=Layer1Config)
    layer2: Layer2Config = field(default_factory=Layer2Config)
    layer3: Layer3Config = field(default_factory=Layer3Config)
    cross_layer: CrossLayerConfig = field(default_factory=CrossLayerConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    generate: GenerateConfig = field(default_factory=GenerateConfig)


def load_config(path: str = None) -> FullConfig:
    """加载配置，支持从 YAML 文件覆盖默认值"""
    config = FullConfig()
    if path:
        with open(path, 'r',encoding='utf-8') as f:
            overrides = yaml.safe_load(f)
        # Recursively update config from YAML
        for section_name, section_vals in overrides.items():
            if hasattr(config, section_name):
                section = getattr(config, section_name)
                for k, v in section_vals.items():
                    if hasattr(section, k):
                        setattr(section, k, v)
    return config
