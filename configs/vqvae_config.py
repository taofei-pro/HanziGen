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
    encoder_base_channels: int = 128  # 从256降到128，平衡性能和速度
    latent_dim: int = 4              # 从8降到4，减少显存占用
    codebook_size: int = 256         # 从512降到256，减少显存占用
    commitment_cost: float = 0.25


@dataclass
class VQVAETrainingConfig:
    """
    Configuration class for the VQVAE training settings.
    """

    learning_rate: float = 1e-3
    min_learning_rate: float = 1e-6
    num_epochs: int = 100

    model_save_path: str = "checkpoints/vqvae.pth"

    tensorboard_log_dir: str = "runs/VQVAE"
