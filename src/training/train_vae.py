import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
from pathlib import Path
import argparse
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.vae import VAE
from src.data.dataset import create_dataloaders


class VAETrainer:
    def __init__(
        self,
        model: VAE,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: optim.Optimizer,
        device: torch.device,
        checkpoint_dir: str = "checkpoints",
        use_wandb: bool = True
    ):
        """
        Initialize VAE trainer.

        Parameters
        ----------
        model : VAE
            VAE model
        train_loader : DataLoader
            Training data loader
        val_loader : DataLoader
            Validation data loader
        optimizer : optim.Optimizer
            Optimizer
        device : torch.device
            Device to train on
        checkpoint_dir : str
            Directory to save checkpoints
        use_wandb : bool
            Whether to use Weights & Biases for logging
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.use_wandb = use_wandb

        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch_idx, (data, _) in enumerate(pbar):
            data = data.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(data)
            loss_dict = self.model.loss_function(
                data,
                outputs['recon'],
                outputs['mu'],
                outputs['logvar']
            )

            # Backward pass
            loss_dict['loss'].backward()
            self.optimizer.step()

            # Track losses
            total_loss += loss_dict['loss'].item()
            total_recon_loss += loss_dict['recon_loss'].item()
            total_kl_loss += loss_dict['kl_loss'].item()

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss_dict['loss'].item():.4f}",
                'recon': f"{loss_dict['recon_loss'].item():.4f}",
                'kl': f"{loss_dict['kl_loss'].item():.4f}"
            })

        # Average losses
        avg_loss = total_loss / len(self.train_loader)
        avg_recon_loss = total_recon_loss / len(self.train_loader)
        avg_kl_loss = total_kl_loss / len(self.train_loader)

        return {
            'train_loss': avg_loss,
            'train_recon_loss': avg_recon_loss,
            'train_kl_loss': avg_kl_loss
        }

    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0

        for data, _ in tqdm(self.val_loader, desc="Validation"):
            data = data.to(self.device)

            # Forward pass
            outputs = self.model(data)
            loss_dict = self.model.loss_function(
                data,
                outputs['recon'],
                outputs['mu'],
                outputs['logvar']
            )

            # Track losses
            total_loss += loss_dict['loss'].item()
            total_recon_loss += loss_dict['recon_loss'].item()
            total_kl_loss += loss_dict['kl_loss'].item()

        # Average losses
        avg_loss = total_loss / len(self.val_loader)
        avg_recon_loss = total_recon_loss / len(self.val_loader)
        avg_kl_loss = total_kl_loss / len(self.val_loader)

        return {
            'val_loss': avg_loss,
            'val_recon_loss': avg_recon_loss,
            'val_kl_loss': avg_kl_loss
        }

    @torch.no_grad()
    def visualize_reconstructions(self, epoch: int, num_samples: int = 4):
        self.model.eval()

        # Get a batch of validation data
        data, _ = next(iter(self.val_loader))
        data = data[:num_samples].to(self.device)

        # Reconstruct
        outputs = self.model(data)
        recon = outputs['recon']

        # Convert to numpy
        data_np = data.cpu().numpy()
        recon_np = recon.cpu().numpy()

        # Create visualization
        fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))

        for i in range(num_samples):
            # Original - Real part (middle slice)
            mid_z = data_np.shape[-1] // 2
            axes[i, 0].imshow(data_np[i, 0, :, :, mid_z], cmap='RdBu')
            axes[i, 0].set_title(f"Original Real (z={mid_z})")
            axes[i, 0].axis('off')

            # Original - Imaginary part
            axes[i, 1].imshow(data_np[i, 1, :, :, mid_z], cmap='RdBu')
            axes[i, 1].set_title(f"Original Imag (z={mid_z})")
            axes[i, 1].axis('off')

            # Reconstruction - Real part
            axes[i, 2].imshow(recon_np[i, 0, :, :, mid_z], cmap='RdBu')
            axes[i, 2].set_title(f"Recon Real (z={mid_z})")
            axes[i, 2].axis('off')

            # Reconstruction - Imaginary part
            axes[i, 3].imshow(recon_np[i, 1, :, :, mid_z], cmap='RdBu')
            axes[i, 3].set_title(f"Recon Imag (z={mid_z})")
            axes[i, 3].axis('off')

        plt.tight_layout()

        if self.use_wandb:
            wandb.log({"reconstructions": wandb.Image(fig)}, step=epoch)

        plt.close(fig)

        return fig

    @torch.no_grad()
    def visualize_samples(self, epoch: int, num_samples: int = 4):
        self.model.eval()

        # Sample from latent space
        samples = self.model.sample(num_samples, self.device)
        samples_np = samples.cpu().numpy()

        # Create visualization
        fig, axes = plt.subplots(num_samples, 2, figsize=(8, 4 * num_samples))

        for i in range(num_samples):
            mid_z = samples_np.shape[-1] // 2

            # Real part
            axes[i, 0].imshow(samples_np[i, 0, :, :, mid_z], cmap='RdBu')
            axes[i, 0].set_title(f"Sample {i+1} Real (z={mid_z})")
            axes[i, 0].axis('off')

            # Imaginary part
            axes[i, 1].imshow(samples_np[i, 1, :, :, mid_z], cmap='RdBu')
            axes[i, 1].set_title(f"Sample {i+1} Imag (z={mid_z})")
            axes[i, 1].axis('off')

        plt.tight_layout()

        if self.use_wandb:
            wandb.log({"samples": wandb.Image(fig)}, step=epoch)

        plt.close(fig)

        return fig

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }

        # Save latest checkpoint
        checkpoint_path = self.checkpoint_dir / "latest_checkpoint.pt"
        torch.save(checkpoint, checkpoint_path)

        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "best_checkpoint.pt"
            torch.save(checkpoint, best_path)
            print(f"Saved best checkpoint with val_loss: {self.best_val_loss:.4f}")

    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.best_val_loss = checkpoint['best_val_loss']
        return checkpoint['epoch']

    def train(self, num_epochs: int, visualize_every: int = 10):
        """
        Train the VAE.

        Parameters
        ----------
        num_epochs : int
            Number of epochs to train
        visualize_every : int
            Visualize reconstructions and samples every N epochs
        """
        print(f"Starting training for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(1, num_epochs + 1):
            # Train
            train_metrics = self.train_epoch(epoch)
            self.train_losses.append(train_metrics['train_loss'])

            # Validate
            val_metrics = self.validate(epoch)
            self.val_losses.append(val_metrics['val_loss'])

            # Combine metrics
            metrics = {**train_metrics, **val_metrics, 'epoch': epoch}

            # Log to wandb
            if self.use_wandb:
                wandb.log(metrics, step=epoch)

            # Print epoch summary
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"Train Loss: {train_metrics['train_loss']:.4f} "
                  f"(Recon: {train_metrics['train_recon_loss']:.4f}, "
                  f"KL: {train_metrics['train_kl_loss']:.4f})")
            print(f"Val Loss: {val_metrics['val_loss']:.4f} "
                  f"(Recon: {val_metrics['val_recon_loss']:.4f}, "
                  f"KL: {val_metrics['val_kl_loss']:.4f})")

            # Save checkpoint
            is_best = val_metrics['val_loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['val_loss']

            self.save_checkpoint(epoch, is_best=is_best)

            # Visualize
            if epoch % visualize_every == 0:
                self.visualize_reconstructions(epoch)
                self.visualize_samples(epoch)

        print("\nTraining complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")


def train_vae(
    data_dir: str,
    output_dir: str = "outputs/vae",
    latent_dim: int = 128,
    batch_size: int = 32,
    num_epochs: int = 100,
    learning_rate: float = 1e-4,
    beta: float = 1.0,
    grid_size: tuple = (10, 10, 10),
    use_wandb: bool = True,
    wandb_project: str = "gpe-vae",
    device: Optional[str] = None
):
    """
    Train VAE on GPE initial conditions.

    Parameters
    ----------
    data_dir : str
        Directory containing the dataset
    output_dir : str
        Directory to save outputs
    latent_dim : int
        Latent dimension
    batch_size : int
        Batch size
    num_epochs : int
        Number of training epochs
    learning_rate : float
        Learning rate
    beta : float
        Beta parameter for beta-VAE
    grid_size : tuple
        Grid size (Nx, Ny, Nz)
    use_wandb : bool
        Whether to use Weights & Biases
    wandb_project : str
        W&B project name
    device : str, optional
        Device to use (cuda/cpu)
    """
    # Setup device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    print(f"Using device: {device}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize wandb
    if use_wandb:
        wandb.init(
            project=wandb_project,
            config={
                'latent_dim': latent_dim,
                'batch_size': batch_size,
                'num_epochs': num_epochs,
                'learning_rate': learning_rate,
                'beta': beta,
                'grid_size': grid_size
            }
        )

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        mode="initial_conditions",
        num_workers=4
    )

    # Create model
    model = VAE(
        input_shape=grid_size,
        latent_dim=latent_dim,
        beta=beta
    ).to(device)

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Create trainer
    trainer = VAETrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        checkpoint_dir=output_path / "checkpoints",
        use_wandb=use_wandb
    )

    # Train
    trainer.train(num_epochs=num_epochs)

    if use_wandb:
        wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Train VAE on GPE initial conditions")
    parser.add_argument("--data-dir", type=str, required=True, help="Dataset directory")
    parser.add_argument("--output-dir", type=str, default="outputs/vae", help="Output directory")
    parser.add_argument("--latent-dim", type=int, default=128, help="Latent dimension")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--num-epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--beta", type=float, default=1.0, help="Beta for beta-VAE")
    parser.add_argument("--grid-size", type=int, nargs=3, default=[10, 10, 10], help="Grid size")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--wandb-project", type=str, default="gpe-vae", help="W&B project name")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")

    args = parser.parse_args()

    train_vae(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        latent_dim=args.latent_dim,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.lr,
        beta=args.beta,
        grid_size=tuple(args.grid_size),
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
        device=args.device
    )


if __name__ == "__main__":
    main()
