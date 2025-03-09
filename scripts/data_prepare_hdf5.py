"""
This script prepares HDF5 files with the signal data and segmentation labels
for training and evaluating the segmentation model.
"""

import os
import sys
import h5py
import argparse
import numpy as np
import pandas as pd
import multiprocessing as mp
from pathlib import Path
from tqdm import tqdm
import random

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from read import read_fast5
from normalization import normalize_signal_wrapper
from seeds import DATA_PREPARE_READ_SHUFFLE

def regular_break_points(n, chunk_len, overlap=0, align='mid'):
    """Define the start and end points of the raw data based on the 
    window length and overlap
    
    Args:
        n (int): length of the raw data
        chunk_len (int): window size
        overlap (int): overlap between windows
        align (str): relative to the whole length, how should the windows align
    """
    num_chunks, remainder = divmod(n - overlap, chunk_len - overlap)
    start = {'left': 0, 'mid': remainder // 2, 'right': remainder}[align]
    starts = np.arange(start, start + num_chunks*(chunk_len - overlap), (chunk_len - overlap))
    return np.vstack([starts, starts + chunk_len]).T

def load_segmentation_tsv(tsv_file, cache_file=None):
    """Load segmentation data from TSV file with caching
    
    Args:
        tsv_file (str): Path to TSV file with segmentation data
        cache_file (str): Path to save/load cached version (default: None)
        
    Returns:
        dict: Dictionary with read_id as key and segmentation data as value
    """
    # Use cached data if available
    if cache_file and os.path.exists(cache_file):
        try:
            print(f"Loading cached segmentation data from {cache_file}")
            with h5py.File(cache_file, 'r') as f:
                segmentation_data = {}
                for read_id in f.keys():
                    segmentation_data[read_id] = {
                        'start': f[read_id]['start'][:],
                        'end': f[read_id]['end'][:]
                    }
                return segmentation_data
        except Exception as e:
            print(f"Error loading cached file: {e}")
            # Fall through to load from TSV
    
    try:
        print(f"Loading segmentation data from {tsv_file}")
        
        # Try to use faster methods if available
        try:
            import pyarrow.csv as pa_csv
            
            # Using PyArrow for faster CSV/TSV reading
            print("Using PyArrow for faster TSV loading")
            read_options = pa_csv.ReadOptions(use_threads=True)
            parse_options = pa_csv.ParseOptions(delimiter='\t')
            
            table = pa_csv.read_csv(tsv_file, read_options=read_options, parse_options=parse_options)
            df = table.to_pandas()
            
        except ImportError:
            # Fall back to pandas with chunking for memory efficiency
            print("PyArrow not available, using pandas with chunking")
            chunks = []
            chunksize = 1000000  # Adjust based on available memory
            
            for chunk in pd.read_csv(tsv_file, sep='\t', chunksize=chunksize):
                chunks.append(chunk)
            
            df = pd.concat(chunks)
        
        # Group by read_id
        segmentation_data = {}
        
        for read_id, group in df.sort_values(['read_id', 'kmer_idx']).groupby('read_id'):
            segmentation_data[read_id] = {
                'start': group['start_raw_idx'].to_numpy(),
                'end': group['end_raw_idx'].to_numpy()
            }
        
        # Cache the results if specified
        if cache_file:
            print(f"Caching segmentation data to {cache_file}")
            os.makedirs(os.path.dirname(cache_file) if os.path.dirname(cache_file) else '.', exist_ok=True)
            
            with h5py.File(cache_file, 'w') as f:
                for read_id, data in segmentation_data.items():
                    read_group = f.create_group(read_id)
                    read_group.create_dataset('start', data=data['start'], compression='gzip')
                    read_group.create_dataset('end', data=data['end'], compression='gzip')
                    
        return segmentation_data
    except Exception as e:
        print(f"Error loading TSV file: {e}")
        return {}

def create_segmentation_mask(signal_length, event_starts, event_ends, window_start=0):
    """Create binary segmentation mask from event boundaries
    
    Args:
        signal_length (int): Length of the signal
        event_starts (np.array): Start positions of events
        event_ends (np.array): End positions of events
        window_start (int): Start position of the window in the original signal
        
    Returns:
        np.array: Binary mask with 1s for events and 0s for non-events
    """
    # Initialize mask with zeros
    mask = np.zeros(signal_length, dtype=np.float32)
    
    # Convert to window-relative positions
    rel_starts = event_starts - window_start
    rel_ends = event_ends - window_start
    
    # Fill in the mask with 1s for events
    for start, end in zip(rel_starts, rel_ends):
        # Adjust start and end to be within window bounds
        start = max(0, start)
        end = min(signal_length, end)
        
        # Only process if the segment overlaps with the window
        if start < signal_length and end > 0:
            mask[start:end] = 1.0
        
    return mask

def segment_read(read_file, segmentation_data, window_length, overlap):
    """Process a single fast5 read into chunks with segmentation masks
    
    Args:
        read_file (str): Path to fast5 file
        segmentation_data (dict): Dictionary with segmentation data from TSV
        window_length (int): Size of the chunks in raw datapoints
        overlap (int): Overlap between windows
        
    Returns:
        tuple: (signals, masks) - normalized signals and segmentation masks
    """
    if not os.path.isfile(read_file):
        print(f'File not found, skipping: {read_file}')
        return None, None
    
    multiread_obj = read_fast5(read_file)
    
    # Lists to store all the segments
    signals = []
    masks = []
    
    for read_id in multiread_obj.keys():
        read_obj = multiread_obj[read_id]
        
        # Skip if no segmentation data is available for this read
        if read_id not in segmentation_data:
            continue
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
        
        # Get event boundaries
        event_starts = segmentation_data[read_id]['start']
        event_ends = segmentation_data[read_id]['end']
        
        # Calculate window boundaries
        break_points = regular_break_points(len(normalized_signal), window_length, overlap=overlap, align='left')
        
        # Process each window
        for i, j in break_points:
            # Extract signal for this window
            window_signal = normalized_signal[i:j]
            
            # Create segmentation mask for this window
            window_mask = create_segmentation_mask(j-i, event_starts, event_ends, window_start=i)
            
            # Store the data
            signals.append(window_signal)
            masks.append(window_mask)
    
    if len(signals) < 1:
        return None, None
    
    return np.array(signals), np.array(masks)

def process_batch(read_files, segmentation_file, output_file, window_length, overlap, cache_file=None):
    """Process a batch of reads and save to HDF5 file
    
    Args:
        read_files (list): List of fast5 files to process
        segmentation_file (str): Path to segmentation TSV file
        output_file (str): Path to output HDF5 file
        window_length (int): Size of the chunks in raw datapoints
        overlap (int): Overlap between windows
        cache_file (str): Optional path to cache segmentation data
    """
    # Load segmentation data with caching if specified
    segmentation_data = load_segmentation_tsv(segmentation_file, cache_file)
    
    all_signals = []
    all_masks = []
    
    for read_file in read_files:
        signals, masks = segment_read(read_file, segmentation_data, window_length, overlap)
        if signals is not None:
            all_signals.append(signals)
            all_masks.append(masks)
    
    if len(all_signals) == 0:
        print(f"No valid data for {output_file}")
        return
    
    # Concatenate all data
    X = np.vstack(all_signals)
    Y = np.vstack(all_masks)
    
    # Reshape for proper format [batch, channels, length]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    Y = Y.reshape(Y.shape[0], 1, Y.shape[1])
    
    # Save to HDF5 file
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('signals', data=X, compression='gzip', compression_opts=4)
        f.create_dataset('masks', data=Y, compression='gzip', compression_opts=4)
    
    print(f"Saved {X.shape[0]} examples to {output_file}")

def merge_hdf5_files(input_files, output_file, delete_input=True, verbose=True):
    """Merge multiple HDF5 files into a single file
    
    Args:
        input_files (list): List of HDF5 files to merge
        output_file (str): Path to output merged file
        delete_input (bool): Whether to delete input files after merging
        verbose (bool): Whether to show progress information
    """
    if verbose:
        print(f"Merging {len(input_files)} files into {output_file}")
    
    # Get total number of examples
    signals_total = 0
    masks_total = 0
    
    # First pass to get total shapes
    for file_path in input_files:
        try:
            with h5py.File(file_path, 'r') as f:
                if 'signals' in f and 'masks' in f:
                    signals_total += f['signals'].shape[0]
                    masks_total += f['masks'].shape[0]
        except:
            print(f"Warning: Could not read {file_path}, skipping")
    
    # Check if we have data to merge
    if signals_total == 0:
        print("No data to merge")
        return
    
    # Get shapes from first valid file
    signal_shape = None
    mask_shape = None
    
    for file_path in input_files:
        try:
            with h5py.File(file_path, 'r') as f:
                if 'signals' in f and 'masks' in f and f['signals'].shape[0] > 0:
                    signal_shape = f['signals'].shape
                    mask_shape = f['masks'].shape
                    break
        except:
            continue
    
    if signal_shape is None:
        print("No valid data found in any file")
        return
    
    # Create merged file with resizable datasets
    with h5py.File(output_file, 'w') as out_f:
        # Create datasets
        out_f.create_dataset(
            'signals', 
            shape=(0, signal_shape[1], signal_shape[2]),
            maxshape=(None, signal_shape[1], signal_shape[2]),
            dtype=np.float32,
            compression='gzip', 
            compression_opts=4
        )
        
        out_f.create_dataset(
            'masks', 
            shape=(0, mask_shape[1], mask_shape[2]),
            maxshape=(None, mask_shape[1], mask_shape[2]),
            dtype=np.float32,
            compression='gzip', 
            compression_opts=4
        )
        
        # Merge files
        current_idx = 0
        for file_path in tqdm(input_files, disable=not verbose):
            try:
                with h5py.File(file_path, 'r') as in_f:
                    if 'signals' in in_f and 'masks' in in_f and in_f['signals'].shape[0] > 0:
                        signals = in_f['signals'][:]
                        masks = in_f['masks'][:]
                        
                        # Resize output datasets
                        new_size = current_idx + signals.shape[0]
                        out_f['signals'].resize(new_size, axis=0)
                        out_f['masks'].resize(new_size, axis=0)
                        
                        # Copy data
                        out_f['signals'][current_idx:new_size] = signals
                        out_f['masks'][current_idx:new_size] = masks
                        
                        current_idx = new_size
            except:
                print(f"Warning: Could not merge data from {file_path}, skipping")
    
    if delete_input and verbose:
        print(f"Cleaning up temporary files")
    
    # Delete input files if requested
    if delete_input:
        for file_path in input_files:
            try:
                os.remove(file_path)
            except:
                print(f"Warning: Could not delete {file_path}")
    
    if verbose:
        print(f"Merged {current_idx} examples into {output_file}")

def main(fast5_dir, segmentation_file, output_dir, total_files, 
         window_length, window_overlap, cache_file=None, n_cores=1, verbose=True):
    """Process fast5 files and segmentation data into HDF5 files
    
    Args:
        fast5_dir (str): Directory with fast5 files
        segmentation_file (str): Path to segmentation TSV file
        output_dir (str): Directory to save HDF5 files
        total_files (int): Number of output files to create at the end
        window_length (int): Size of the chunks in raw datapoints
        window_overlap (int): Overlap between windows
        cache_file (str): Optional path to cache segmentation data
        n_cores (int): Number of processes to use
        verbose (bool): Show progress information
    """
    print('Finding files to process')
    
    # Find all fast5 files
    files_list = []
    for path in Path(fast5_dir).rglob('*.fast5'):
        files_list.append(str(path))
    
    if verbose:
        print(f'Found {len(files_list)} files to process')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Shuffle files for better distribution
    random.seed(DATA_PREPARE_READ_SHUFFLE)
    random.shuffle(files_list)
    
    # Use n_cores for parallel processing regardless of final total_files
    processing_chunks = max(n_cores, total_files)
    
    # Split files into batches for processing
    file_batches = np.array_split(files_list, processing_chunks)
    
    # Determine if we need to merge files at the end
    merge_needed = (total_files < processing_chunks)
    temp_dir = None
    
    if merge_needed:
        temp_dir = os.path.join(output_dir, 'temp')
        os.makedirs(temp_dir, exist_ok=True)
    
    # Process batches in parallel
    pool = mp.Pool(processes=n_cores)
    
    jobs = []
    temp_files = []
    
    for i, batch in enumerate(file_batches):
        if merge_needed:
            # Use temporary output files
            output_file = os.path.join(temp_dir, f'temp_segmentation_{i}.h5')
            temp_files.append(output_file)
        else:
            # Create final output files directly
            output_file = os.path.join(output_dir, f'segmentation_{i}.h5')
            
        jobs.append(pool.apply_async(
            process_batch, 
            (batch.tolist(), segmentation_file, output_file, window_length, window_overlap, cache_file)
        ))
    
    if verbose:
        print(f'Processing {len(jobs)} batches in parallel')
    
    for job in tqdm(jobs, disable=not verbose):
        job.get()
    
    pool.close()
    pool.join()
    
    # Merge files if needed
    if merge_needed:
        if verbose:
            print(f"Processing complete. Merging into {total_files} final file(s)")
        
        # Group temp files for merging
        merged_file_groups = np.array_split(temp_files, total_files)
        
        # Merge each group
        for i, group in enumerate(merged_file_groups):
            output_file = os.path.join(output_dir, f'segmentation_{i}.h5')
            merge_hdf5_files(group.tolist(), output_file, delete_input=True, verbose=verbose)
        
        # Remove temp directory if empty
        try:
            os.rmdir(temp_dir)
        except:
            pass
    
    print('Done')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare HDF5 files from fast5 files and segmentation data for training')
    parser.add_argument('--fast5-dir', type=str, required=True, help='Directory containing fast5 files')
    parser.add_argument('--segmentation-file', type=str, required=True, help='Path to segmentation TSV file')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save output HDF5 files')
    parser.add_argument('--cache-file', type=str, help='File to cache segmentation data for faster processing')
    parser.add_argument('--total-files', type=int, default=10, help='Number of final output files to create (processing will use n_cores for parallelism regardless)')
    parser.add_argument('--window-size', type=int, default=4000, help='Size of signal windows')
    parser.add_argument('--window-overlap', type=int, default=200, help='Overlap between consecutive windows')
    parser.add_argument('--n-cores', type=int, default=4, help='Number of processes to use')
    parser.add_argument('--verbose', action='store_true', help='Show progress information')
    
    args = parser.parse_args()
    
    main(
        fast5_dir=args.fast5_dir,
        segmentation_file=args.segmentation_file,
        output_dir=args.output_dir,
        total_files=args.total_files,
        window_length=args.window_size,
        window_overlap=args.window_overlap,
        cache_file=args.cache_file,
        n_cores=args.n_cores,
        verbose=args.verbose
    )
