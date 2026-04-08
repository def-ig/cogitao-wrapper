# DEPRECATION NOTICE⚠️

> [!CAUTION]
> **This repository is no longer maintained.**

All functionality and code from this wrapper have been merged directly into the core **[yassinetb/COGITAO](https://github.com/yassinetb/COGITAO)** repository. 

Please migrate your projects to use the main repository for the latest features, bug fixes, and support.

# cogitao-wrapper

A PyTorch Dataset and generator wrapper for COGITAO (Compositional Generalization in Task-Oriented Abstraction and Object-based reasoning).

## Installation

Using pip:
```bash
pip install git+https://github.com/def-ig/cogitao-wrapper.git
```

Using uv:
```bash
uv add git+https://github.com/def-ig/cogitao-wrapper.git
```

## Quick Start

### Generate a dataset

```python
from cogitao_wrapper import GeneratorConfig, DatasetGenerator

# Configure generator
config = GeneratorConfig(
    output_file="./data/my_dataset.h5",
    grid_size=32,
    num_tasks=10000,
    num_workers=8,
    image_size=224,
)

# Generate dataset
generator = DatasetGenerator(config)
generator.generate()
```

### Load and use dataset

```python
from cogitao_wrapper import CogitaoDataset
from torch.utils.data import DataLoader

# Load dataset
dataset = CogitaoDataset(path="./data/my_dataset.h5")

# Create DataLoader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Use in training
for batch in dataloader:
    images = batch["imgs"]  # Shape: [B, C, H, W]
    # Your training code here
```
