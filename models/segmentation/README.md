# Nanopore Signal Segmentation Model

This module provides a state-of-the-art segmentation model for Oxford Nanopore signals based on the URnano architecture. The model can identify individual events (DNA bases) in raw nanopore signals, which is a crucial step in the basecalling process.

## Features

- U-Net architecture with recurrent GRU layers adapted for 1D signal segmentation
- Efficient data processing with HDF5 format
- Binary segmentation to identify event boundaries in the signal
- Support for both PyTorch and ONNX Runtime inference
- High performance with overlapping window processing

## Usage

### Data Preparation

First, prepare your data by converting fast5 files and segmentation labels (from f5c resquiggle) into HDF5 format:

```bash
python scripts/data_prepare_hdf5.py \
    --fast5-dir /path/to/fast5/files \
    --segmentation-file /path/to/segmentation.tsv \
    --output-dir /path/to/output/directory \
    --total-files 10 \
    --window-size 4000 \
    --window-overlap 200 \
    --n-cores 4 \
    --verbose
```

### Training

Train the segmentation model using the prepared HDF5 data:

```bash
python scripts/train_segmentation.py \
    --data-dir /path/to/hdf5/files \
    --model-dir models/segmentation/checkpoints \
    --log-dir logs \
    --batch-size 64 \
    --epochs 100 \
    --learning-rate 0.001
```

### Inference

Run segmentation on new fast5 files using a trained model:

```bash
# Using PyTorch
python scripts/segment_signal.py \
    --input /path/to/fast5/file/or/directory \
    --output /path/to/output.tsv \
    --model models/segmentation/checkpoints/best_model.pt \
    --window-size 4000 \
    --overlap 200 \
    --format tsv

# Using ONNX Runtime
python scripts/segment_signal.py \
    --input /path/to/fast5/file/or/directory \
    --output /path/to/output.tsv \
    --model models/segmentation/checkpoints/best_model.onnx \
    --use-onnx \
    --window-size 4000 \
    --overlap 200 \
    --format tsv
```

## Model Architecture

The segmentation model architecture is based on URnano with the following key components:

1. **Input**: 1D raw nanopore signal normalized using median and MAD
2. **Encoder**: U-Net encoder with convolutional blocks and GRU layers
3. **Bottleneck**: Flat block to process the most compressed features
4. **Decoder**: U-Net decoder with transposed convolutions and skip connections
5. **Output**: 1D binary segmentation mask indicating event regions

## Performance Considerations

- Window size of 4000 datapoints with 200 overlap works well for most signals
- Dice loss provides better training for segmentation compared to BCE loss
- Inference can be accelerated by using ONNX Runtime, especially on CPU

## Requirements

- PyTorch >= 1.7.0
- ONNX Runtime >= 1.7.0
- h5py
- numpy
- pandas
- ont_fast5_api

## Citation

If you use this segmentation model in your research, please cite the original URnano paper:

```
@article{urnano,
  title={URnano: nanopore basecalling via deep multitask learning},
  author={Yao Zong, etc.},
  journal={Bioinformatics},
  year={2021}
}
```