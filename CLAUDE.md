# Basecalling Architectures Project Guide

## Commands
- Install dependencies: `pip install -r requirements.txt`
- Process data: `python scripts/data_prepare_numpy.py`
- Train models:
  - Original: `python scripts/train_original.py`
  - Combined: `python scripts/train_comb.py`
  - Grid Recurrent: `python scripts/train_grid_recurrent.py`
- Run basecalling:
  - Original: `python scripts/basecall_original.py`
  - Combined: `python scripts/basecall_comb.py`

## Code Style Guidelines
- Indentation: 4 spaces, no tabs
- Imports: standard library → third-party → local modules
- Classes: CamelCase (BaseCaller, URNanoBasecaller)
- Functions/variables: snake_case (prepare_data, signal_length)
- Constants: UPPERCASE (DEFAULT_ALPHABET)
- Documentation: Google-style docstrings
- Error handling: Use try/except blocks with specific exceptions
- Structure: Keep model definitions separate from training logic

This project implements various architectures for nanopore basecalling using PyTorch.