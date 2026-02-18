import os
import shutil

import h5py
import numpy as np
import pytest

from cogitao_wrapper import DatasetGenerator, GeneratorConfig
from cogitao_wrapper.dataset import CogitaoDataset, HDF5CogitaoStore


@pytest.fixture
def temp_gen_path(tmp_path):
    """Fixture for generated dataset path."""
    return str(tmp_path / "gen_dataset.h5")


def test_generator_batch_size_config(temp_gen_path):
    """Test that DatasetGenerator respects batch_size configuration."""
    batch_size = 42

    cfg = GeneratorConfig(
        output_file=temp_gen_path,
        batch_size=batch_size,
        num_workers=2,
        num_tasks=10,
        image_size=32,
    )

    # Generate calls arcworld internally. This might be slow or flaky if network is involved.
    # Assuming local generation logic or mocking is needed if too slow.
    # For now, we trust it runs reasonable fast for small num_tasks.

    generator = DatasetGenerator(cfg)
    generator.generate(num_samples=5)  # Generate fewer samples than tasks is fine

    assert os.path.exists(temp_gen_path)

    # Check if batch_size is stored correctly
    store = HDF5CogitaoStore(temp_gen_path)
    assert store.batch_size == batch_size


def test_single_mode_generation(temp_gen_path):
    """Test single mode dataset generation and validation."""

    cfg = GeneratorConfig(
        output_file=temp_gen_path,
        num_tasks=5,
        num_workers=2,
        min_transformation_depth=0,
        max_transformation_depth=0,
        image_size=32,
    )

    generator = DatasetGenerator(cfg)
    generator.generate(num_samples=5)

    # Load and verify
    dataset = CogitaoDataset(temp_gen_path)
    assert len(dataset) == 5

    sample = dataset[0]
    assert isinstance(sample, dict)
    assert "imgs" in sample
    # dataset returns tensors, not numpy arrays
    import torch

    assert isinstance(sample["imgs"], torch.Tensor)
    assert sample["imgs"].shape == (
        3,
        32,
        32,
    )

    # Check that iteration works
    count = 0
    for item in dataset:
        assert item["imgs"].shape == (3, 32, 32)
        count += 1
    assert count == 5
