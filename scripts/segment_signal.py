"""
Segmentation inference script for nanopore signal data
Supports both PyTorch and ONNX Runtime inference
"""

import os
import sys
import argparse
import time
import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm

import torch
import onnxruntime as ort

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from models.segmentation.model import SegmentationModel
from models.segmentation.config import SegmentationConfig
from src.read import read_fast5
from src.normalization import normalize_signal_wrapper

def load_torch_model(model_path):
    """Load a PyTorch model from checkpoint
    
    Args:
        model_path (str): Path to model checkpoint
        
    Returns:
        tuple: (model, config) - The loaded model and its configuration
    """
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    
    # Create configuration from saved config
    config_dict = checkpoint.get('config', {})
    config = SegmentationConfig()
    for k, v in config_dict.items():
        setattr(config, k, v)
    
    # Create model
    model = SegmentationModel(
        n_classes=config.n_classes,
        input_channels=config.input_channels,
        features=config.features
    )
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, config

def load_onnx_model(model_path):
    """Load an ONNX model
    
    Args:
        model_path (str): Path to ONNX model file
        
    Returns:
        ort.InferenceSession: ONNX Runtime inference session
    """
    # Set up ONNX Runtime session
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    # Create inference session
    session = ort.InferenceSession(model_path, sess_options)
    
    return session

def segment_signal_torch(model, signal, device, window_size=4000, overlap=200):
    """Segment a signal using PyTorch model
    
    Args:
        model (nn.Module): PyTorch segmentation model
        signal (np.array): Normalized signal to segment
        device (torch.device): Device to run inference on
        window_size (int): Size of windows to process
        overlap (int): Overlap between windows
        
    Returns:
        np.array: Segmentation mask (1D array of same length as signal)
    """
    # Move model to device
    model = model.to(device)
    
    # Initialize output mask
    segmentation_mask = np.zeros(len(signal))
    
    # Calculate window boundaries
    from src.normalization import regular_break_points
    break_points = regular_break_points(len(signal), window_size, overlap=overlap, align='left')
    
    with torch.no_grad():
        for i, j in break_points:
            # Extract window
            window = signal[i:j]
            
            # Pad window if needed
            if len(window) < window_size:
                padding = window_size - len(window)
                window = np.pad(window, (0, padding), 'constant')
            
            # Prepare input
            x = torch.from_numpy(window).float().unsqueeze(0).unsqueeze(0)
            x = x.to(device)
            
            # Run inference
            output = model(x)
            
            # Get mask from output (threshold > 0.5)
            mask = (output.squeeze().cpu().numpy() > 0.5).astype(np.float32)
            
            # Unpad if needed
            if len(window) < window_size:
                mask = mask[:-padding]
            
            # Handle overlapping regions with max operation
            segmentation_mask[i:j] = np.maximum(segmentation_mask[i:j], mask)
    
    return segmentation_mask

def segment_signal_onnx(session, signal, window_size=4000, overlap=200):
    """Segment a signal using ONNX Runtime
    
    Args:
        session (ort.InferenceSession): ONNX Runtime session
        signal (np.array): Normalized signal to segment
        window_size (int): Size of windows to process
        overlap (int): Overlap between windows
        
    Returns:
        np.array: Segmentation mask (1D array of same length as signal)
    """
    # Get input and output names
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    # Initialize output mask
    segmentation_mask = np.zeros(len(signal))
    
    # Calculate window boundaries
    from src.normalization import regular_break_points
    break_points = regular_break_points(len(signal), window_size, overlap=overlap, align='left')
    
    for i, j in break_points:
        # Extract window
        window = signal[i:j]
        
        # Pad window if needed
        if len(window) < window_size:
            padding = window_size - len(window)
            window = np.pad(window, (0, padding), 'constant')
        
        # Prepare input
        x = window.reshape(1, 1, -1).astype(np.float32)
        
        # Run inference
        outputs = session.run([output_name], {input_name: x})
        
        # Get mask from output (threshold > 0.5)
        mask = (outputs[0].squeeze() > 0.5).astype(np.float32)
        
        # Unpad if needed
        if len(window) < window_size:
            mask = mask[:-padding]
        
        # Handle overlapping regions with max operation
        segmentation_mask[i:j] = np.maximum(segmentation_mask[i:j], mask)
    
    return segmentation_mask

def process_read(read_file, model, device=None, use_onnx=False, window_size=4000, overlap=200):
    """Process a fast5 file and segment the signals
    
    Args:
        read_file (str): Path to fast5 file
        model: PyTorch model or ONNX session
        device (torch.device): Device to run PyTorch inference on
        use_onnx (bool): Whether to use ONNX Runtime
        window_size (int): Size of windows to process
        overlap (int): Overlap between windows
        
    Returns:
        dict: Dictionary with read_id as key and segmentation results as value
    """
    if not os.path.isfile(read_file):
        print(f'File not found: {read_file}')
        return {}
    
    multiread_obj = read_fast5(read_file)
    
    segmentation_results = {}
    
    for read_id, read_obj in multiread_obj.items():
        # Get raw signal
        raw_signal = read_obj.raw
        
        # Normalize signal
        normalized_signal = normalize_signal_wrapper(
            raw_signal, 
            read_obj.offset, 
            read_obj.range, 
            read_obj.digitisation,
            method='noisiest', 
            samples=100, 
            threshold=6.0, 
            factor=1.4826
        )
        
        # Segment signal
        start_time = time.time()
        
        if use_onnx:
            segmentation_mask = segment_signal_onnx(model, normalized_signal, window_size, overlap)
        else:
            segmentation_mask = segment_signal_torch(model, normalized_signal, device, window_size, overlap)
        
        inference_time = time.time() - start_time
        
        # Find event boundaries (transitions between 0->1 and 1->0)
        events = []
        in_event = False
        event_start = 0
        
        for i in range(len(segmentation_mask)):
            if segmentation_mask[i] > 0.5 and not in_event:
                # Start of event
                in_event = True
                event_start = i
            elif segmentation_mask[i] <= 0.5 and in_event:
                # End of event
                in_event = False
                events.append((event_start, i))
        
        # Add last event if it extends to the end of the signal
        if in_event:
            events.append((event_start, len(segmentation_mask)))
        
        segmentation_results[read_id] = {
            'signal_length': len(normalized_signal),
            'events': events,
            'inference_time': inference_time,
            'event_count': len(events)
        }
    
    return segmentation_results

def save_segmentation_results(results, output_file, format='tsv'):
    """Save segmentation results to file
    
    Args:
        results (dict): Dictionary with read_id as key and segmentation results as value
        output_file (str): Path to output file
        format (str): Output format ('tsv' or 'h5')
    """
    if format == 'tsv':
        with open(output_file, 'w') as f:
            # Write header
            f.write('read_id\tkmer_idx\tstart_raw_idx\tend_raw_idx\n')
            
            # Write events
            for read_id, data in results.items():
                for i, (start, end) in enumerate(data['events']):
                    f.write(f'{read_id}\t{i}\t{start}\t{end}\n')
                    
        print(f'Segmentation results saved to {output_file}')
    
    elif format == 'h5':
        with h5py.File(output_file, 'w') as f:
            for read_id, data in results.items():
                # Create group for read
                read_group = f.create_group(read_id)
                
                # Store events as dataset
                events = np.array(data['events'])
                read_group.create_dataset('events', data=events)
                
                # Store metadata
                read_group.attrs['signal_length'] = data['signal_length']
                read_group.attrs['event_count'] = data['event_count']
                read_group.attrs['inference_time'] = data['inference_time']
                
        print(f'Segmentation results saved to {output_file}')
    
    else:
        raise ValueError(f'Unsupported output format: {format}')

def main(args):
    """Main function"""
    # Set device for PyTorch
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    
    # Load model
    if args.use_onnx:
        print(f'Loading ONNX model from {args.model}')
        model = load_onnx_model(args.model)
    else:
        print(f'Loading PyTorch model from {args.model}')
        model, config = load_torch_model(args.model)
    
    # Collect fast5 files
    if os.path.isdir(args.input):
        # Find all fast5 files in directory
        fast5_files = list(Path(args.input).rglob('*.fast5'))
    else:
        # Single file
        fast5_files = [Path(args.input)]
    
    print(f'Found {len(fast5_files)} fast5 files')
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Process files
    all_results = {}
    total_events = 0
    total_time = 0
    
    for fast5_file in tqdm(fast5_files):
        results = process_read(
            str(fast5_file),
            model,
            device,
            args.use_onnx,
            args.window_size,
            args.overlap
        )
        
        all_results.update(results)
        
        # Collect statistics
        for read_id, data in results.items():
            total_events += data['event_count']
            total_time += data['inference_time']
    
    # Save results
    save_segmentation_results(all_results, args.output, args.format)
    
    # Print statistics
    read_count = len(all_results)
    if read_count > 0:
        print(f'Processed {read_count} reads')
        print(f'Total events detected: {total_events}')
        print(f'Average events per read: {total_events / read_count:.2f}')
        print(f'Average inference time per read: {total_time / read_count:.4f}s')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Segment nanopore signals using trained model')
    parser.add_argument('--input', type=str, required=True, help='Input fast5 file or directory')
    parser.add_argument('--output', type=str, required=True, help='Output file path')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint (.pt) or ONNX model (.onnx)')
    parser.add_argument('--use-onnx', action='store_true', help='Use ONNX Runtime for inference')
    parser.add_argument('--cpu', action='store_true', help='Force CPU inference for PyTorch')
    parser.add_argument('--window-size', type=int, default=4000, help='Size of signal windows')
    parser.add_argument('--overlap', type=int, default=200, help='Overlap between consecutive windows')
    parser.add_argument('--format', type=str, choices=['tsv', 'h5'], default='tsv', help='Output format')
    
    args = parser.parse_args()
    main(args)