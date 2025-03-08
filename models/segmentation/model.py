"""Implementation of the Segmentation Model based on URnano

Based on: 
https://github.com/yaozhong/URnano
"""

import os
import sys
import torch
from torch import nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from layers.urnano import URNetDownBlock, URNetFlatBlock, URNetUpBlock, URNet


class SegmentationModel(nn.Module):
    """Segmentation model based on URnano architecture
    
    This model is designed for nanopore signal segmentation using the URnano architecture,
    which combines a U-Net structure with recurrent layers for improved temporal understanding.
    """
    def __init__(self, n_classes=1, input_channels=1, features=64):
        super(SegmentationModel, self).__init__()
        """
        Args:
            n_classes (int): Number of segmentation classes (default: 1 for binary segmentation)
            input_channels (int): Number of input channels (default: 1 for raw signal)
            features (int): Number of features in the first layer (default: 64)
        """
        self.n_classes = n_classes
        self.features = features
        
        # Build network components
        self.convolution = self.build_cnn(input_channels)
        self.segmentation_head = nn.Conv1d(features, n_classes, kernel_size=1)
        
    def forward(self, x):
        """Forward pass
        
        Args:
            x (tensor) : [batch, channels (1), len]
            
        Returns:
            x (tensor) : [batch, n_classes, len]
        """
        x = self.convolution(x)
        x = self.segmentation_head(x)
        
        # For binary segmentation, use sigmoid
        if self.n_classes == 1:
            return torch.sigmoid(x)
        # For multi-class segmentation, use softmax
        else:
            return F.softmax(x, dim=1)
            
    def build_cnn(self, input_channels=1):
        """Build the U-Net with recurrent components
        
        Args:
            input_channels (int): Number of input channels
            
        Returns:
            nn.Sequential: The CNN part of the model
        """
        padding = 'same'
        stride = 1
        n_channels = [64, 128, 256, 512]
        kernel = 11
        maxpooling = [2, 2, 2]
        
        down = nn.ModuleList([
            URNetDownBlock(input_channels, n_channels[0], kernel, maxpooling[0], stride, padding),
            URNetDownBlock(n_channels[0], n_channels[1], 3, maxpooling[1], stride, padding),
            URNetDownBlock(n_channels[1], n_channels[2], 3, maxpooling[2], stride, padding)
        ])
        
        flat = nn.ModuleList([
            URNetFlatBlock(n_channels[2], n_channels[3], 3, stride, padding)
        ])
        
        up = nn.ModuleList([
            URNetUpBlock(n_channels[3], n_channels[2], 3, maxpooling[2], maxpooling[2], stride, padding), 
            URNetUpBlock(n_channels[2], n_channels[1], 3, maxpooling[1], maxpooling[1], stride, padding),
            URNetUpBlock(n_channels[1], n_channels[0], 3, maxpooling[0], maxpooling[0], stride, padding)
        ])
        
        cnn = nn.ModuleList([
            URNet(down, flat, up), 
            nn.Conv1d(n_channels[0], self.features, 3, stride, padding), 
            nn.BatchNorm1d(self.features), 
            nn.ReLU()
        ])
        
        return nn.Sequential(*cnn)
        
    def export_onnx(self, save_path, input_shape=(1, 1, 4000)):
        """Export model to ONNX format
        
        Args:
            save_path (str): Path to save the ONNX model
            input_shape (tuple): Shape of the input tensor
        """
        dummy_input = torch.randn(*input_shape)
        torch.onnx.export(
            self, 
            dummy_input, 
            save_path,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size', 2: 'signal_length'},
                'output': {0: 'batch_size', 2: 'signal_length'}
            }
        )
        print(f"Model exported to ONNX format at {save_path}")