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
    """VAE-based model to decompose summed contact matrices."""
    
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
        
        # Latent vector encoders (one for each phase)
        # Each phase gets its own latent vector
        self.latent_encoder_earlyG1 = nn.Linear(hidden_size2, latent_dim)
        self.latent_encoder_midG1 = nn.Linear(hidden_size2, latent_dim)
        self.latent_encoder_lateG1 = nn.Linear(hidden_size2, latent_dim)
        
        # ========== DECODER ==========
        # Decoder layers for each phase (fully connected)
        self.decoder_earlyG1 = nn.Linear(latent_dim, self.output_size)
        self.decoder_midG1 = nn.Linear(latent_dim, self.output_size)
        self.decoder_lateG1 = nn.Linear(latent_dim, self.output_size)
        
        # Activation function
        self.relu = nn.ReLU()
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, 1, matrix_size, matrix_size)
        
        Returns:
            Tuple of 3 tensors, each of shape (batch_size, output_size)
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
        
        # Encode into 3 latent vectors (one for each phase)
        latent_earlyG1 = self.latent_encoder_earlyG1(x)  # (batch, latent_dim)
        latent_midG1 = self.latent_encoder_midG1(x)      # (batch, latent_dim)
        latent_lateG1 = self.latent_encoder_lateG1(x)    # (batch, latent_dim)
        
        # ========== DECODER ==========
        # Decode from latent vectors (fully connected layers)
        out_earlyG1 = self.decoder_earlyG1(latent_earlyG1)
        out_midG1 = self.decoder_midG1(latent_midG1)
        out_lateG1 = self.decoder_lateG1(latent_lateG1)
        
        return out_earlyG1, out_midG1, out_lateG1


def weighted_mse_loss(pred, target, weight_method='squared', alpha=2.0):
    if weight_method == 'squared':
        # Square both predictions and targets before computing MSE
        # This amplifies the penalty for errors in high-contact regions
        pred_sq = pred ** 2
        target_sq = target ** 2
        squared_diff = (pred_sq - target_sq) ** 2
        loss = torch.mean(squared_diff)
    
    else:
        # Default: standard MSE
        loss = torch.mean((pred - target) ** 2)
    
    return loss


def train_epoch(model, dataloader, optimizer, device, weight_method='squared', alpha=2.0):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch in dataloader:
        inputs = batch['input'].to(device)  # (batch, 1, 100, 100)
        targets = batch['targets'].to(device)  # (batch, 3, 100, 100)
        
        # Flatten targets for loss computation
        targets_flat = targets.view(targets.size(0), 3, -1)  # (batch, 3, 10000)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)  # Tuple of 3 tensors, each (batch, 10000)
        
        # Compute weighted loss for each phase and sum
        loss = 0.0
        for i, phase in enumerate(['earlyG1', 'midG1', 'lateG1']):
            loss += weighted_mse_loss(
                outputs[i], 
                targets_flat[:, i, :],
                weight_method=weight_method,
                alpha=alpha
            )
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def evaluate(model, dataloader, device, weight_method='none', alpha=2.0):
    """Evaluate on validation/test set."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['input'].to(device)  # (batch, 1, 100, 100)
            targets = batch['targets'].to(device)  # (batch, 3, 100, 100)
            
            # Flatten targets
            targets_flat = targets.view(targets.size(0), 3, -1)  # (batch, 3, 10000)
            
            # Forward pass
            outputs = model(inputs)
            
            # Compute weighted loss
            loss = 0.0
            for i in range(3):
                loss += weighted_mse_loss(
                    outputs[i],
                    targets_flat[:, i, :],
                    weight_method=weight_method,
                    alpha=alpha
                )
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0


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
    batch_size = 4
    num_epochs = 10
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
    latent_dim = 256  # Dimension of latent vectors
    model = SimpleDecompositionModel(
        matrix_size=matrix_size,
        hidden_size1=hidden_size1,
        hidden_size2=hidden_size2,
        latent_dim=latent_dim
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss configuration (weighted MSE to emphasize high-contact regions)
    # Options: 'squared' (square values), 'linear' (linear weighting), 'adaptive' (adaptive weighting)
    weight_method = 'squared'  # Try 'squared', 'linear', or 'adaptive'
    alpha = 2.0  # Weight factor for linear/adaptive methods
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"Using weighted MSE loss method: {weight_method}")
    if weight_method in ['linear', 'adaptive']:
        print(f"  Weight factor (alpha): {alpha}")
    
    # Training loop
    print("\nStarting training...")
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device, weight_method=weight_method, alpha=alpha)
        test_loss = evaluate(model, test_loader, device, weight_method=weight_method, alpha=alpha)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Test Loss: {test_loss:.6f}")
    
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

