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
    encoder_base_channels: int = 112  # 从96增加到112，小幅提升编码能力
    latent_dim: int = 4              # 从3增加到4，扩大潜在空间
    codebook_size: int = 192         # 从128增加到192，增加码本容量
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
