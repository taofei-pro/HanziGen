import argparse

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from configs import VQVAEDatasetConfig, VQVAEModelConfig, VQVAETrainingConfig
from datasets.loader import Loader
from models import VQVAE
from utils.argparse.argparse_utils import update_config_from_args
from utils.hardware.hardware_utils import print_model_params, select_device


def parse_args() -> argparse.Namespace:
    """ """
    parser = argparse.ArgumentParser(description="Train VQ-GAN model")
    parser.add_argument(
        "--split_ratios", type=float, nargs=2, help="Train/val split ratios"
    )
    parser.add_argument("--random_seed", type=int, help="Random seed")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, help="Number of epochs")
    parser.add_argument("--model_save_path", type=str, help="Model save path")
    parser.add_argument("--device", type=str, help="Training device (mps, cpu, cuda)")
    
    # VQ-GAN related arguments (默认启用VQ-GAN)
    parser.add_argument("--use_vqgan", action="store_true", default=True, help="Enable VQ-GAN mode (default: True)")
    parser.add_argument("--discriminator_lr", type=float, help="Discriminator learning rate")
    parser.add_argument("--perceptual_weight", type=float, help="Perceptual loss weight")
    parser.add_argument("--adversarial_weight", type=float, help="Adversarial loss weight")
    
    # Additional training arguments
    parser.add_argument("--font_path", type=str, help="Font file path")
    parser.add_argument("--log_dir", type=str, help="Log directory")
    parser.add_argument("--sample_dir", type=str, help="Sample directory")
    parser.add_argument("--save_every_n_epochs", type=int, help="Save model every N epochs")
    parser.add_argument("--log_every_n_steps", type=int, help="Log every N steps")
    parser.add_argument("--val_every_n_epochs", type=int, help="Validate every N epochs")

    return parser.parse_args()


def train_vqgan(
    dataset_config: VQVAEDatasetConfig,
    model_config: VQVAEModelConfig,
    training_config: VQVAETrainingConfig,
    device: torch.device,
    use_vqgan: bool = True,  # 默认启用VQ-GAN
    discriminator_lr: float = 1e-4,
    perceptual_weight: float = 0.1,
    adversarial_weight: float = 0.05,
):
    """ """
    loader = Loader.from_dataset_config(
        dataset_config=dataset_config,
        device=device,
    )

    # 强制启用VQ-GAN模式
    model_config.use_vqgan = True
    model_config.discriminator_lr = discriminator_lr
    model_config.perceptual_weight = perceptual_weight
    model_config.adversarial_weight = adversarial_weight

    print("🚀 启用VQ-GAN模式训练...")
    print(f"   - 判别器学习率: {discriminator_lr}")
    print(f"   - 感知损失权重: {perceptual_weight}")
    print(f"   - 对抗损失权重: {adversarial_weight}")

    vqvae = VQVAE(
        model_config=model_config,
        device=device,
    )

    # Main optimizer for generator
    optimizer = optim.Adam(
        vqvae.parameters(),
        lr=training_config.learning_rate,
    )

    # Discriminator optimizer for VQ-GAN
    discriminator_optimizer = None
    if hasattr(vqvae, 'discriminator') and vqvae.discriminator is not None:
        discriminator_optimizer = optim.Adam(
            vqvae.discriminator.parameters(),
            lr=discriminator_lr,
        )
        print("✅ 判别器优化器已初始化")

    scheduler = CosineAnnealingLR(
        optimizer=optimizer,
        T_max=training_config.num_epochs,
        eta_min=training_config.min_learning_rate,
    )

    print_model_params(
        model=vqvae,
    )

    vqvae.fit(
        loader=loader,
        optimizer=optimizer,
        scheduler=scheduler,
        training_config=training_config,
        discriminator_optimizer=discriminator_optimizer,
    )


def main() -> None:
    """ """
    args = parse_args()
    dataset_config = update_config_from_args(
        converting_config=VQVAEDatasetConfig(),
        args=args,
    )
    model_config = update_config_from_args(
        converting_config=VQVAEModelConfig(),
        args=args,
    )
    training_config = update_config_from_args(
        converting_config=VQVAETrainingConfig(),
        args=args,
    )
    device = select_device(args.device)

    train_vqgan(
        dataset_config=dataset_config,
        model_config=model_config,
        training_config=training_config,
        device=device,
        use_vqgan=getattr(args, 'use_vqgan', True),  # 默认启用VQ-GAN
        discriminator_lr=getattr(args, 'discriminator_lr', 1e-4),
        perceptual_weight=getattr(args, 'perceptual_weight', 0.1),
        adversarial_weight=getattr(args, 'adversarial_weight', 0.05),
    )


if __name__ == "__main__":
    main()
