"""
Simple GAN training script for contact matrix decomposition.
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

# seed
torch.manual_seed(42)

# Add parent directory to path to import Dataloader
sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocess.Dataloader import CellCycleDataLoader
from datasets import ContactMatrixDataset
from visualization import print_maps


class Generator(nn.Module):
    def __init__(self, matrix_size=100):
        super(Generator, self).__init__()
        self.matrix_size = matrix_size
        self.conv1 = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        self.weight_earlyG1 = nn.Parameter(torch.empty(matrix_size, matrix_size))
        self.weight_midG1 = nn.Parameter(torch.empty(matrix_size, matrix_size))
        self.weight_lateG1 = nn.Parameter(torch.empty(matrix_size, matrix_size))
        self._init_identity_conv(self.conv1)
        self._init_identity_conv(self.conv2)
        self._initialize_weights()

    def _init_identity_conv(self, conv):
        with torch.no_grad():
            conv.weight.zero_()
            conv.bias.zero_()
            center = conv.weight.size(2) // 2
            conv.weight[0, 0, center, center] = 1.0

    def _initialize_weights(self):
        init = torch.full((self.matrix_size, self.matrix_size), 1.0 / 3.0)
        with torch.no_grad():
            self.weight_earlyG1.copy_(init)
            self.weight_midG1.copy_(init)
            self.weight_lateG1.copy_(init)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        w_early = self.weight_earlyG1.unsqueeze(0).unsqueeze(0)
        w_mid   = self.weight_midG1.unsqueeze(0).unsqueeze(0)
        w_late  = self.weight_lateG1.unsqueeze(0).unsqueeze(0)
        out_earlyG1 = (x * w_early).view(x.size(0), -1)
        out_midG1   = (x * w_mid).view(x.size(0), -1)
        out_lateG1  = (x * w_late).view(x.size(0), -1)
        return out_earlyG1, out_midG1, out_lateG1


class Discriminator(nn.Module):
    def __init__(self, matrix_size=100):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(matrix_size * matrix_size, 256)
        self.fc2 = nn.Linear(256, 1)
        self._initialize_weights()
    
    def _initialize_weights(self):
        nn.init.kaiming_uniform_(self.fc1.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.kaiming_uniform_(self.fc2.weight, a=0, mode='fan_in', nonlinearity='linear')
        nn.init.constant_(self.fc2.bias, 0)
    
    def forward(self, x):
        x = torch.nn.functional.leaky_relu(self.fc1(x), negative_slope=0.2)
        return self.fc2(x)


def bulk_consistency_loss(outputs, input_summed, matrix_size):
    if len(input_summed.shape) == 4:
        input_flat = input_summed.view(input_summed.size(0), -1)
    else:
        input_flat = input_summed
    output_sum = outputs[0] + outputs[1] + outputs[2]
    return nn.functional.mse_loss(output_sum, input_flat)


def train_epoch_gan(generator, discriminator, dataloader, gen_optimizer, disc_optimizer, 
                    device, matrix_size, bulk_weight=1.0, adv_weight=0.1, epoch=0):
    generator.train()
    discriminator.train()
    
    total_gen_loss = 0.0
    total_disc_loss = 0.0
    total_recon_loss = 0.0
    total_bulk_loss = 0.0
    total_adv_loss = 0.0
    num_batches = 0
    
    for batch in dataloader:
        inputs = batch['input'].to(device)  # (batch, 1, matrix_size, matrix_size)
        targets = batch['targets'].to(device)  # (batch, 3, matrix_size, matrix_size)
        
        batch_size = inputs.size(0)
        
        # Train Discriminator (skip first epoch)
        #if epoch >= 1:
        for param in generator.parameters():
            param.requires_grad = False
        
        disc_optimizer.zero_grad()
        disc_loss_real = 0.0
        real_labels = torch.full((batch_size, 1), 0.9, device=device)
        for i in range(3):
            real_phase = targets[:, i, :, :].view(batch_size, -1)
            disc_loss_real += nn.functional.mse_loss(discriminator(real_phase), real_labels)
        disc_loss_real = disc_loss_real / 3.0
        
        with torch.no_grad():
            gen_outputs = generator(inputs)
        
        disc_loss_fake = 0.0
        fake_labels = torch.full((batch_size, 1), 0.1, device=device)
        for i in range(3):
            disc_loss_fake += nn.functional.mse_loss(discriminator(gen_outputs[i]), fake_labels)
        disc_loss_fake = disc_loss_fake / 3.0
        
        disc_loss = (disc_loss_real + disc_loss_fake) / 2.0
        disc_loss.backward()
        disc_optimizer.step()
        total_disc_loss += disc_loss.item()
        
        # Train Generator
        for param in generator.parameters():
            param.requires_grad = True
        for param in discriminator.parameters():
            param.requires_grad = False
        
        gen_optimizer.zero_grad()
        gen_outputs = generator(inputs)
        
        recon_loss = 0.0
        for i in range(3):
            target_flat = targets[:, i, :, :].view(batch_size, -1)
            recon_loss += nn.functional.mse_loss(gen_outputs[i], target_flat)
        
        bulk_loss = bulk_consistency_loss(gen_outputs, inputs, matrix_size)
        
        adv_loss = 0.0
        true_labels = torch.full((batch_size, 1), 0.9, device=device)
        for i in range(3):
            adv_loss += nn.functional.mse_loss(discriminator(gen_outputs[i]), true_labels)
        adv_loss = adv_loss / 3.0
        
        gen_loss = recon_loss + bulk_weight * bulk_loss + adv_weight * adv_loss
        gen_loss.backward()
        gen_optimizer.step()
        
        for param in discriminator.parameters():
            param.requires_grad = True

        total_gen_loss += gen_loss.item()
        total_recon_loss += recon_loss.item()
        total_bulk_loss += bulk_loss.item()
        total_adv_loss += adv_loss.item()
        num_batches += 1
    
    avg_gen_loss = total_gen_loss / num_batches if num_batches > 0 else 0.0
    avg_disc_loss = total_disc_loss / num_batches if num_batches > 0 else 0.0
    avg_recon = total_recon_loss / num_batches if num_batches > 0 else 0.0
    avg_bulk = total_bulk_loss / num_batches if num_batches > 0 else 0.0
    avg_adv = total_adv_loss / num_batches if num_batches > 0 else 0.0
    
    return avg_gen_loss, avg_disc_loss, avg_recon, avg_bulk, avg_adv


def evaluate_gan(generator, discriminator, dataloader, device, matrix_size, 
                 bulk_weight=1.0, adv_weight=0.1):
    generator.eval()
    discriminator.eval()
    total_gen_loss = 0.0
    total_disc_loss = 0.0
    total_recon_loss = 0.0
    total_bulk_loss = 0.0
    total_adv_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['input'].to(device)
            targets = batch['targets'].to(device)
            batch_size = inputs.size(0)
            
            disc_loss_real = 0.0
            real_labels = torch.full((batch_size, 1), 0.9, device=device)
            for i in range(3):
                real_phase = targets[:, i, :, :].view(batch_size, -1)
                disc_loss_real += nn.functional.mse_loss(discriminator(real_phase), real_labels)
            disc_loss_real = disc_loss_real / 3.0
            
            gen_outputs = generator(inputs)
            disc_loss_fake = 0.0
            fake_labels = torch.full((batch_size, 1), 0.1, device=device)
            for i in range(3):
                disc_loss_fake += nn.functional.mse_loss(discriminator(gen_outputs[i]), fake_labels)
            disc_loss_fake = disc_loss_fake / 3.0
            disc_loss = (disc_loss_real + disc_loss_fake) / 2.0
            
            recon_loss = 0.0
            for i in range(3):
                target_flat = targets[:, i, :, :].view(batch_size, -1)
                recon_loss += nn.functional.mse_loss(gen_outputs[i], target_flat)
            
            bulk_loss = bulk_consistency_loss(gen_outputs, inputs, matrix_size)
            
            adv_loss = 0.0
            true_labels = torch.full((batch_size, 1), 0.9, device=device)
            for i in range(3):
                adv_loss += nn.functional.mse_loss(discriminator(gen_outputs[i]), true_labels)
            adv_loss = adv_loss / 3.0
            
            gen_loss = recon_loss + bulk_weight * bulk_loss + adv_weight * adv_loss
            
            total_gen_loss += gen_loss.item()
            total_disc_loss += disc_loss.item()
            total_recon_loss += recon_loss.item()
            total_bulk_loss += bulk_loss.item()
            total_adv_loss += adv_loss.item()
            num_batches += 1
    
    avg_gen_loss = total_gen_loss / num_batches if num_batches > 0 else 0.0
    avg_disc_loss = total_disc_loss / num_batches if num_batches > 0 else 0.0
    avg_recon = total_recon_loss / num_batches if num_batches > 0 else 0.0
    avg_bulk = total_bulk_loss / num_batches if num_batches > 0 else 0.0
    avg_adv = total_adv_loss / num_batches if num_batches > 0 else 0.0
    
    return avg_gen_loss, avg_disc_loss, avg_recon, avg_bulk, avg_adv


def train_model_gan(generator, discriminator, train_loader, test_loader, 
                    gen_optimizer, disc_optimizer, device, matrix_size, num_epochs, 
                    bulk_weight=1.0, adv_weight=0.1):
    for epoch in range(num_epochs):
        train_gen_loss, train_disc_loss, train_recon, train_bulk, train_adv = train_epoch_gan(
            generator, discriminator, train_loader, gen_optimizer, disc_optimizer, 
            device, matrix_size, bulk_weight=bulk_weight, adv_weight=adv_weight, epoch=epoch
        )
        test_gen_loss, test_disc_loss, test_recon, test_bulk, test_adv = evaluate_gan(
            generator, discriminator, test_loader, device, matrix_size,
            bulk_weight=bulk_weight, adv_weight=adv_weight
        )
        print(f"Epoch {epoch+1}/{num_epochs} - Gen: {train_gen_loss:.6f} (R:{train_recon:.4f} B:{train_bulk:.4f} A:{train_adv:.4f}) | Disc: {train_disc_loss:.6f}")
    return generator, discriminator


def main():
    data_dir = "/nfs/turbo/umms-minjilab/lpullela/cell-cycle/raw_data/zhang_4dn"
    batch_size = 1
    num_epochs = 50
    gen_lr = 0.001
    disc_lr = 0.0001  # Higher LR for discriminator to help it learn
    bulk_weight = 1.0
    adv_weight = 1.0
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    cell_cycle_loader = CellCycleDataLoader(
        data_dir=data_dir,
        resolution=10000,
        region_size=500000,
        normalization="VC"
    )
    
    dataset = ContactMatrixDataset(cell_cycle_loader, use_log_scale=True)
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    matrix_size = 50
    generator = Generator(matrix_size=matrix_size).to(device)
    discriminator = Discriminator(matrix_size=matrix_size).to(device)
    
    # Comment out visualization calls if not needed
    # print_maps(generator, test_dataset, device, matrix_size=matrix_size, sample_idx=0, tag_name='before_training')
    
    gen_optimizer = optim.Adam(generator.parameters(), lr=gen_lr, weight_decay=1e-5)
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=disc_lr, weight_decay=1e-5)
    
    generator, discriminator = train_model_gan(
        generator, discriminator, train_loader, test_loader, 
        gen_optimizer, disc_optimizer, device, matrix_size, num_epochs, 
        bulk_weight=bulk_weight, adv_weight=adv_weight
    )
    
    gen_path = Path(__file__).parent / "generator.pt"
    disc_path = Path(__file__).parent / "discriminator.pt"
    torch.save(generator.state_dict(), gen_path)
    torch.save(discriminator.state_dict(), disc_path)
    print(f"Models saved: {gen_path}, {disc_path}")
    
    # Comment out visualization calls if not needed
    print_maps(generator, test_dataset, device, matrix_size=matrix_size, sample_idx=0, tag_name='after_training')
    print_maps(generator, test_dataset, device, matrix_size=matrix_size, sample_idx=1, tag_name='after_training')
    print_maps(generator, test_dataset, device, matrix_size=matrix_size, sample_idx=2, tag_name='after_training')
    print_maps(generator, test_dataset, device, matrix_size=matrix_size, sample_idx=3, tag_name='after_training')
    print_maps(generator, test_dataset, device, matrix_size=matrix_size, sample_idx=4, tag_name='after_training')
    print_maps(generator, test_dataset, device, matrix_size=matrix_size, sample_idx=5, tag_name='after_training')


if __name__ == "__main__":
    main()

