"""Configuration for the segmentation model

This file contains the configuration parameters for the segmentation model.
"""

import os
import sys

class SegmentationConfig:
    """Configuration for the segmentation model"""
    
    def __init__(self):
        # Data parameters
        self.window_size = 4000  # Signal window size
        self.window_overlap = 200  # Overlap between windows
        
        # Model parameters
        self.n_classes = 1  # Binary segmentation (event/non-event)
        self.input_channels = 1
        self.features = 64
        
        # Training parameters
        self.batch_size = 64
        self.epochs = 100
        self.learning_rate = 1e-3
        self.weight_decay = 1e-5
        
        # Normalization parameters
        self.normalization_method = 'noisiest'  # Options: 'noisiest', 'all'
        self.normalization_samples = 100
        self.normalization_threshold = 6.0
        self.normalization_factor = 1.4826
        
        # Paths
        self.data_dir = 'data'
        self.model_dir = 'models/segmentation/checkpoints'
        self.log_dir = 'logs'
        
    def save(self, filepath):
        """Save the configuration to a file"""
        import json
        
        config_dict = {k: v for k, v in self.__dict__.items()}
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=4)
            
    @classmethod
    def load(cls, filepath):
        """Load the configuration from a file"""
        import json
        
        config = cls()
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
            
        for k, v in config_dict.items():
            setattr(config, k, v)
            
        return config