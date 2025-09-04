from dataclasses import dataclass


@dataclass
class VQVAEDatasetConfig:
    """
    Configuration class for the VQVAE dataset settings.
    """

    target_img_dir: str = "data/target"
    reference_img_dir: str = "data/reference"

    splits_root: str = "charsets"
    split_ratios: tuple[float, float] = (0.8, 0.2)
    random_seed: int = 2025
    batch_size: int = 8
    num_workers: int = 4


@dataclass
class VQVAEModelConfig:
    """
    Configuration class for the VQVAE architecture settings.
    """

    input_img_channels: int = 1
    encoder_base_channels: int = 112  # 回退到批次#5的稳定配置
    latent_dim: int = 4              # 回退到批次#5的稳定配置
    codebook_size: int = 192         # 回退到批次#5的稳定配置
    commitment_cost: float = 0.25


@dataclass
class VQVAETrainingConfig:
    """
    Configuration class for VQ-VAE training settings.
    """

    # 针对小数据集优化：增加训练轮数，优化学习率策略
    learning_rate: float = 8e-4        # 从6e-4增加到8e-4，加快收敛
    min_learning_rate: float = 1e-6    # 保持默认值
    num_epochs: int = 400              # 从200增加到400，充分学习小数据集
    warmup_epochs: int = 50            # 新增：预热训练轮数
    early_stopping_patience: int = 30  # 新增：早停耐心值

    model_save_path: str = "checkpoints/vqvae.pth"
    tensorboard_log_dir: str = "runs/VQVAE"
