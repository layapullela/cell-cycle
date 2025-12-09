"""
Simple training script for contact matrix decomposition.

The model learns to decompose summed contact matrices into their constituent phases.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt

# Add parent directory to path to import Dataloader
sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocess.Dataloader import CellCycleDataLoader


class ContactMatrixDataset(Dataset):
    """PyTorch Dataset wrapper for CellCycleDataLoader."""
    
    def __init__(self, data_loader):
        """
        Args:
            data_loader: CellCycleDataLoader instance
        """
        self.data_loader = data_loader
        self.phases = ['earlyG1', 'midG1', 'lateG1']  # Required phases
    
    def __len__(self):
        return len(self.data_loader)
    
    def __getitem__(self, idx):
        """Get a sample and convert to tensors."""
        sample = self.data_loader[idx]
        
        # Get matrices for each phase
        matrices = []
        for phase in self.phases:
            if phase not in sample:
                raise ValueError(f"Phase {phase} not found in sample")
            matrices.append(sample[phase])
        
        # Stack into tensor: [earlyG1, midG1, lateG1]
        # Shape: (3, 100, 100) for 1MB regions
        matrices_tensor = torch.FloatTensor(np.stack(matrices))
        
        # Sum all three matrices: (100, 100)
        summed_matrix = torch.sum(matrices_tensor, dim=0)
        
        # Add channel dimension for conv layers: (1, 100, 100)
        input_matrix = summed_matrix.unsqueeze(0)
        
        return {
            'input': input_matrix,  # (1, 100, 100)
            'targets': matrices_tensor,  # (3, 100, 100)
            'region': sample['region']
        }


class SimpleDecompositionModel(nn.Module):
    """Variational Autoencoder (VAE) to decompose summed contact matrices."""
    
    def __init__(self, matrix_size=100, hidden_size1=1024, hidden_size2=512, latent_dim=256):
        """
        Args:
            matrix_size: Size of input matrix (100x100 for 1MB regions)
            hidden_size1: Size of first hidden layer in encoder
            hidden_size2: Size of second hidden layer in encoder
            latent_dim: Dimension of latent vectors for each phase
        """
        super(SimpleDecompositionModel, self).__init__()
        
        self.matrix_size = matrix_size
        self.input_size = matrix_size * matrix_size  # 10000 for 1MB regions
        self.output_size = self.input_size
        self.latent_dim = latent_dim
        
        # ========== ENCODER ==========
        # Convolutional layers for spatial feature extraction
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)  # (batch, 1, 100, 100) -> (batch, 16, 100, 100)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # (batch, 16, 100, 100) -> (batch, 32, 100, 100)
        
        # Flatten after conv layers: 32 * 100 * 100 = 320000
        conv_output_size = 32 * matrix_size * matrix_size
        
        # Shared processing layers in encoder
        self.encoder_layer1 = nn.Linear(conv_output_size, hidden_size1)
        self.encoder_layer2 = nn.Linear(hidden_size1, hidden_size2)
        
        # VAE: Encoder outputs both mean (mu) and log variance (logvar) for each phase
        # Each phase gets its own latent distribution parameters
        self.latent_mu_earlyG1 = nn.Linear(hidden_size2, latent_dim)
        self.latent_logvar_earlyG1 = nn.Linear(hidden_size2, latent_dim)
        
        self.latent_mu_midG1 = nn.Linear(hidden_size2, latent_dim)
        self.latent_logvar_midG1 = nn.Linear(hidden_size2, latent_dim)
        
        self.latent_mu_lateG1 = nn.Linear(hidden_size2, latent_dim)
        self.latent_logvar_lateG1 = nn.Linear(hidden_size2, latent_dim)
        
        # ========== DECODER ==========
        # Decoder layers for each phase (fully connected)
        self.decoder_earlyG1 = nn.Linear(latent_dim, self.output_size)
        self.decoder_midG1 = nn.Linear(latent_dim, self.output_size)
        self.decoder_lateG1 = nn.Linear(latent_dim, self.output_size)
        
        # Activation function
        self.relu = nn.ReLU()
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: z = mu + sigma * epsilon
        where epsilon ~ N(0, 1) and sigma = exp(0.5 * logvar)
        
        Args:
            mu: Mean tensor (batch, latent_dim)
            logvar: Log variance tensor (batch, latent_dim)
        
        Returns:
            Sampled latent vector z (batch, latent_dim)
        """
        # Clip logvar to prevent numerical instability (exp of very large values)
        logvar = torch.clamp(logvar, min=-10, max=10)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x, return_latent_params=False):
        """
        Args:
            x: Input tensor of shape (batch_size, 1, matrix_size, matrix_size)
            return_latent_params: If True, also return mu and logvar for KL loss
        
        Returns:
            If return_latent_params=False: Tuple of 3 tensors (outputs), each (batch_size, output_size)
            If return_latent_params=True: Tuple of (outputs, mu_dict, logvar_dict)
        """
        # ========== ENCODER ==========
        # Convolutional layers for spatial feature extraction
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        
        # Flatten for linear layers
        x = x.view(x.size(0), -1)  # (batch, 32*100*100)
        
        # Shared processing layers
        x = self.relu(self.encoder_layer1(x))
        x = self.relu(self.encoder_layer2(x))
        
        # VAE: Encode into mu and logvar for each phase
        mu_earlyG1 = self.latent_mu_earlyG1(x)  # (batch, latent_dim)
        logvar_earlyG1 = self.latent_logvar_earlyG1(x)  # (batch, latent_dim)
        
        mu_midG1 = self.latent_mu_midG1(x)  # (batch, latent_dim)
        logvar_midG1 = self.latent_logvar_midG1(x)  # (batch, latent_dim)
        
        mu_lateG1 = self.latent_mu_lateG1(x)  # (batch, latent_dim)
        logvar_lateG1 = self.latent_logvar_lateG1(x)  # (batch, latent_dim)
        
        # Reparameterization trick: sample z from N(mu, sigma^2)
        z_earlyG1 = self.reparameterize(mu_earlyG1, logvar_earlyG1)
        z_midG1 = self.reparameterize(mu_midG1, logvar_midG1)
        z_lateG1 = self.reparameterize(mu_lateG1, logvar_lateG1)
        
        # ========== DECODER ==========
        # Decode from sampled latent vectors
        out_earlyG1 = self.decoder_earlyG1(z_earlyG1)
        out_midG1 = self.decoder_midG1(z_midG1)
        out_lateG1 = self.decoder_lateG1(z_lateG1)
        
        if return_latent_params:
            mu_dict = {
                'earlyG1': mu_earlyG1,
                'midG1': mu_midG1,
                'lateG1': mu_lateG1
            }
            logvar_dict = {
                'earlyG1': logvar_earlyG1,
                'midG1': logvar_midG1,
                'lateG1': logvar_lateG1
            }
            return (out_earlyG1, out_midG1, out_lateG1), mu_dict, logvar_dict
        
        return out_earlyG1, out_midG1, out_lateG1


def kl_loss(mu, logvar):
    """
    Compute KL divergence: KL(N(mu, sigma^2) || N(0, 1))
    Formula: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    
    Args:
        mu: Mean tensor (batch, latent_dim)
        logvar: Log variance tensor (batch, latent_dim)
    
    Returns:
        KL divergence loss (scalar)
    """
    # Clip logvar to prevent numerical instability
    logvar = torch.clamp(logvar, min=-10, max=10)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return torch.mean(kl)


def train_epoch(model, dataloader, criterion, optimizer, device, kl_weight=1.0):
    """
    Train for one epoch.
    
    Args:
        kl_weight: Weight for KL divergence term in loss (default 1.0)
    """
    model.train()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_kl_loss = 0.0
    num_batches = 0
    
    for batch in dataloader:
        inputs = batch['input'].to(device)  # (batch, 1, 100, 100)
        targets = batch['targets'].to(device)  # (batch, 3, 100, 100)
        
        # Flatten targets for loss computation
        targets_flat = targets.view(targets.size(0), 3, -1)  # (batch, 3, 10000)
        
        # Forward pass with latent parameters for KL loss
        optimizer.zero_grad()
        outputs, mu_dict, logvar_dict = model(inputs, return_latent_params=True)
        
        # Reconstruction loss (MSE)
        recon_loss = 0.0
        phases = ['earlyG1', 'midG1', 'lateG1']
        for i, phase in enumerate(phases):
            recon_loss += criterion(outputs[i], targets_flat[:, i, :])
        
        # KL divergence loss (regularization to standard normal)
        kl = 0.0
        for phase in phases:
            kl += kl_loss(mu_dict[phase], logvar_dict[phase])
        
        # Total loss: reconstruction + KL divergence
        loss = recon_loss + kl_weight * kl
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping to prevent explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_kl_loss += kl.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_recon = total_recon_loss / num_batches if num_batches > 0 else 0.0
    avg_kl = total_kl_loss / num_batches if num_batches > 0 else 0.0
    
    return avg_loss, avg_recon, avg_kl


def evaluate(model, dataloader, criterion, device, kl_weight=1.0):
    """Evaluate on validation/test set."""
    model.eval()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_kl_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['input'].to(device)  # (batch, 1, 100, 100)
            targets = batch['targets'].to(device)  # (batch, 3, 100, 100)
            
            # Flatten targets
            targets_flat = targets.view(targets.size(0), 3, -1)  # (batch, 3, 10000)
            
            # Forward pass with latent parameters
            outputs, mu_dict, logvar_dict = model(inputs, return_latent_params=True)
            
            # Reconstruction loss
            recon_loss = 0.0
            phases = ['earlyG1', 'midG1', 'lateG1']
            for i, phase in enumerate(phases):
                recon_loss += criterion(outputs[i], targets_flat[:, i, :])
            
            # KL divergence loss
            kl = 0.0
            for phase in phases:
                kl += kl_loss(mu_dict[phase], logvar_dict[phase])
            
            # Total loss
            loss = recon_loss + kl_weight * kl
            
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_recon = total_recon_loss / num_batches if num_batches > 0 else 0.0
    avg_kl = total_kl_loss / num_batches if num_batches > 0 else 0.0
    
    return avg_loss, avg_recon, avg_kl


def print_maps(model, dataset, device, matrix_size=100, sample_idx=0):
    """
    Visualize true and predicted contact maps for one sample.
    This is a debugging function to check model predictions.
    
    Args:
        model: Trained model
        dataset: Dataset to get sample from (not DataLoader)
        device: Device to run on
        matrix_size: Size of matrices (default 100 for 1MB regions)
        sample_idx: Index of sample to visualize
    """
    model.eval()
    
    # Get one sample
    sample = dataset[sample_idx]
    inputs = sample['input'].unsqueeze(0).to(device)  # (1, 1, 100, 100)
    targets = sample['targets'].unsqueeze(0).to(device)  # (1, 3, 100, 100)
    region = sample['region']
    
    # Get predictions
    with torch.no_grad():
        outputs = model(inputs)  # Tuple of 3 tensors, each (1, 10000)
    
    # Reshape predictions back to matrices
    phases = ['earlyG1', 'midG1', 'lateG1']
    predictions = []
    for i in range(3):
        pred_flat = outputs[i].cpu().squeeze(0)  # (10000,)
        pred_matrix = pred_flat.view(matrix_size, matrix_size).numpy()
        predictions.append(pred_matrix)
    
    # Get true matrices
    true_matrices = []
    for i in range(3):
        true_matrix = targets[0, i].cpu().numpy()  # (100, 100)
        true_matrices.append(true_matrix)
    
    # Create visualization
    fig, axes = plt.subplots(3, 2, figsize=(12, 18))
    
    for i, phase in enumerate(phases):
        # True map
        im1 = axes[i, 0].imshow(np.log1p(true_matrices[i]), cmap='YlOrRd', aspect='auto')
        axes[i, 0].set_title(f'True {phase}')
        axes[i, 0].set_xlabel('Bin')
        axes[i, 0].set_ylabel('Bin')
        plt.colorbar(im1, ax=axes[i, 0], label='log(contacts + 1)')
        
        # Predicted map
        im2 = axes[i, 1].imshow(np.log1p(np.maximum(predictions[i], 0)), cmap='YlOrRd', aspect='auto')
        axes[i, 1].set_title(f'Predicted {phase}')
        axes[i, 1].set_xlabel('Bin')
        axes[i, 1].set_ylabel('Bin')
        plt.colorbar(im2, ax=axes[i, 1], label='log(contacts + 1)')
    
    plt.suptitle(f'True vs Predicted Contact Maps\nRegion: {region}', fontsize=14)
    plt.tight_layout()
    
    # Save plot
    output_path = Path(__file__).parent / "prediction_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved prediction comparison to {output_path}")
    
    plt.close()


def main():
    """Main training function."""
    # Configuration
    data_dir = "/nfs/turbo/umms-minjilab/lpullela/cell-cycle/raw_data/zhang_4dn"
    batch_size = 1
    num_epochs = 30
    learning_rate = 0.001
    hidden_size1 = 1024
    hidden_size2 = 512
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    cell_cycle_loader = CellCycleDataLoader(
        data_dir=data_dir,
        resolution=10000,
        region_size=1000000,  # 1MB regions
        normalization="VC"  # Vanilla Coverage normalization
    )
    
    print(f"Total samples: {len(cell_cycle_loader)}")
    
    # Create dataset
    dataset = ContactMatrixDataset(cell_cycle_loader)
    
    # Split into train (90%) and test (10%)
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model (1MB regions = 100x100 matrices)
    matrix_size = 100  # 1MB / 10kb resolution = 100 bins
    latent_dim =  1024  # Dimension of latent vectors
    model = SimpleDecompositionModel(
        matrix_size=matrix_size,
        hidden_size1=hidden_size1,
        hidden_size2=hidden_size2,
        latent_dim=latent_dim
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # VAE KL weight - can be adjusted (typical range: 0.1 to 1.0)
    # Higher values = stronger regularization, more compact latent space
    kl_weight = 0.1  # Start with lower weight, can increase if needed
    
    # Training loop
    print("\nStarting training...")
    print(f"KL divergence weight: {kl_weight}")
    for epoch in range(num_epochs):
        train_loss, train_recon, train_kl = train_epoch(model, train_loader, criterion, optimizer, device, kl_weight=kl_weight)
        test_loss, test_recon, test_kl = evaluate(model, test_loader, criterion, device, kl_weight=kl_weight)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.6f} (Recon: {train_recon:.6f}, KL: {train_kl:.6f})")
        print(f"  Test Loss: {test_loss:.6f} (Recon: {test_recon:.6f}, KL: {test_kl:.6f})")
    
    print("\nTraining completed!")
    
    # Save model
    model_path = Path(__file__).parent / "model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # Visualize predictions for debugging
    print("\nGenerating prediction visualization...")
    print_maps(model, test_dataset, device, matrix_size=matrix_size, sample_idx=0)


if __name__ == "__main__":
    main()

