"""
Training script for the segmentation model
"""

import os
import sys
import argparse
import numpy as np
import h5py
import time
import random
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from models.segmentation.model import SegmentationModel
from models.segmentation.config import SegmentationConfig

class SegmentationDataset(Dataset):
    """Dataset for segmentation model training"""
    
    def __init__(self, hdf5_files):
        """Initialize the dataset
        
        Args:
            hdf5_files (list): List of HDF5 files with segmentation data
        """
        self.hdf5_files = hdf5_files
        
        # Get total number of examples
        self.total_examples = 0
        self.file_indices = []
        
        for file_path in self.hdf5_files:
            with h5py.File(file_path, 'r') as f:
                num_examples = f['signals'].shape[0]
                self.total_examples += num_examples
                self.file_indices.append((file_path, num_examples))
    
    def __len__(self):
        return self.total_examples
    
    def __getitem__(self, idx):
        """Get a sample
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            tuple: (signal, mask) - signal tensor and mask tensor
        """
        # Find which file contains this index
        file_idx = 0
        offset = idx
        
        for file_path, num_examples in self.file_indices:
            if offset < num_examples:
                break
            offset -= num_examples
            file_idx += 1
        
        # Load data from the file
        with h5py.File(self.file_indices[file_idx][0], 'r') as f:
            signal = torch.from_numpy(f['signals'][offset]).float()
            mask = torch.from_numpy(f['masks'][offset]).float()
        
        return signal, mask

def dice_loss(pred, target, smooth=1.0):
    """Dice loss for segmentation
    
    Args:
        pred (torch.Tensor): Predicted segmentation mask [batch, 1, length]
        target (torch.Tensor): Ground truth mask [batch, 1, length]
        smooth (float): Smoothing factor to avoid division by zero
        
    Returns:
        torch.Tensor: Dice loss
    """
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    
    intersection = (pred * target).sum()
    dice = (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    return 1.0 - dice

def train_model(model, train_loader, val_loader, config, device):
    """Train the segmentation model
    
    Args:
        model (nn.Module): The segmentation model
        train_loader (DataLoader): DataLoader for training data
        val_loader (DataLoader): DataLoader for validation data
        config (SegmentationConfig): Configuration
        device (torch.device): Device to use for training
    """
    # Create directories for saving models and logs
    os.makedirs(config.model_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    
    # Define loss function and optimizer
    criterion = dice_loss
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)
    
    # Log file
    log_file = os.path.join(config.log_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    # Training loop
    best_val_loss = float('inf')
    
    with open(log_file, 'w') as log:
        log.write(f"Training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log.write(f"Configuration: {vars(config)}\n\n")
        
        for epoch in range(config.epochs):
            start_time = time.time()
            
            # Training
            model.train()
            train_loss = 0.0
            
            for i, (signals, masks) in enumerate(train_loader):
                signals = signals.to(device)
                masks = masks.to(device)
                
                # Forward pass
                outputs = model(signals)
                loss = criterion(outputs, masks)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
                # Print progress
                if (i + 1) % 10 == 0:
                    print(f'Epoch [{epoch+1}/{config.epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
            
            train_loss /= len(train_loader)
            
            # Validation
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for signals, masks in val_loader:
                    signals = signals.to(device)
                    masks = masks.to(device)
                    
                    outputs = model(signals)
                    loss = criterion(outputs, masks)
                    
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Log epoch results
            epoch_time = time.time() - start_time
            print(f'Epoch [{epoch+1}/{config.epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Time: {epoch_time:.2f}s')
            
            log.write(f'Epoch {epoch+1}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Time: {epoch_time:.2f}s\n')
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model_path = os.path.join(config.model_dir, 'best_model.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'config': vars(config)
                }, model_path)
                print(f'Best model saved to {model_path}')
                log.write(f'Best model saved at epoch {epoch+1} with validation loss {val_loss:.6f}\n')
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                checkpoint_path = os.path.join(config.model_dir, f'checkpoint_epoch_{epoch+1}.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'config': vars(config)
                }, checkpoint_path)
                
                # Export to ONNX format
                onnx_path = os.path.join(config.model_dir, f'model_epoch_{epoch+1}.onnx')
                model.export_onnx(onnx_path)
        
        # Save final model
        final_model_path = os.path.join(config.model_dir, 'final_model.pt')
        torch.save({
            'epoch': config.epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'config': vars(config)
        }, final_model_path)
        print(f'Final model saved to {final_model_path}')
        log.write(f'Training completed at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
        
        # Export final model to ONNX
        onnx_path = os.path.join(config.model_dir, 'final_model.onnx')
        model.export_onnx(onnx_path)
        print(f'Model exported to ONNX format at {onnx_path}')

def main(args):
    """Main function"""
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Load configuration
    config = SegmentationConfig()
    config.data_dir = args.data_dir
    config.model_dir = args.model_dir
    config.log_dir = args.log_dir
    
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.epochs:
        config.epochs = args.epochs
    
    # Save configuration
    os.makedirs(config.model_dir, exist_ok=True)
    config_path = os.path.join(config.model_dir, 'config.json')
    config.save(config_path)
    print(f'Configuration saved to {config_path}')
    
    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Find all HDF5 files in the data directory
    data_files = list(Path(config.data_dir).glob('*.h5'))
    print(f'Found {len(data_files)} HDF5 files')
    
    if len(data_files) == 0:
        print(f'No HDF5 files found in {config.data_dir}')
        return
    
    # Split into training and validation sets
    val_size = max(1, int(len(data_files) * 0.2))
    train_files = data_files[:-val_size]
    val_files = data_files[-val_size:]
    
    print(f'Training files: {len(train_files)}')
    print(f'Validation files: {len(val_files)}')
    
    # Create datasets and data loaders
    train_dataset = SegmentationDataset([str(f) for f in train_files])
    val_dataset = SegmentationDataset([str(f) for f in val_files])
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)
    
    print(f'Training examples: {len(train_dataset)}')
    print(f'Validation examples: {len(val_dataset)}')
    
    # Create model
    model = SegmentationModel(n_classes=config.n_classes, input_channels=config.input_channels, features=config.features)
    model = model.to(device)
    
    # Print model summary
    print(model)
    print(f'Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    
    # Train model
    train_model(model, train_loader, val_loader, config, device)
    
    print('Training completed')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train segmentation model for nanopore signal')
    parser.add_argument('--data-dir', type=str, required=True, help='Directory containing HDF5 files')
    parser.add_argument('--model-dir', type=str, default='models/segmentation/checkpoints', help='Directory to save model checkpoints')
    parser.add_argument('--log-dir', type=str, default='logs', help='Directory to save training logs')
    parser.add_argument('--batch-size', type=int, help='Batch size for training')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, help='Learning rate')
    
    args = parser.parse_args()
    main(args)