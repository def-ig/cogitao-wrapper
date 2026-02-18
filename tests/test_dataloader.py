import os
import shutil

import numpy as np
import pytest
from torch.utils.data import DataLoader

from cogitao_wrapper.dataset import CogitaoDataset, HDF5CogitaoStore


@pytest.fixture
def dummy_dataset_path(tmp_path):
    """Create a dummy dataset for DataLoader testing."""
    path = str(tmp_path / "dataloader_test.h5")
    shape = (3, 32, 32)
    store = HDF5CogitaoStore(path, shape=shape, batch_size=10)

    # Create 100 dummy samples
    data = []
    for i in range(100):
        data.append(np.zeros(shape, dtype=np.float32) + i)
    store.save_batch(data)

    return path


def test_multiworker_loading(dummy_dataset_path):
    """Test DataLoader with multiple workers."""

    dataset = CogitaoDataset(dummy_dataset_path)
    assert len(dataset) == 100

    # Test with 0 and 2 workers
    for num_workers in [0, 2]:
        loader = DataLoader(
            dataset,
            batch_size=10,
            num_workers=num_workers,
            shuffle=True,
            persistent_workers=True if num_workers > 0 else False,
        )

        # Load all batches
        item_count = 0
        for i, batch in enumerate(loader):
            imgs = batch["imgs"]
            assert imgs.shape[0] == 10
            assert imgs.shape[1:] == (3, 32, 32)
            item_count += imgs.shape[0]

        assert item_count == 100


def test_dataloader_shuffle(dummy_dataset_path):
    """Verify shuffle functionality works (order is different)."""
    dataset = CogitaoDataset(dummy_dataset_path)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    # Collect first 5 items from two runs
    run1 = [batch["imgs"].mean().item() for i, batch in enumerate(loader) if i < 5]
    run2 = [batch["imgs"].mean().item() for i, batch in enumerate(loader) if i < 5]

    # statistically unlikely to be same if shuffled correctly
    # But wait, logic: if shuffle=True, order should be random. Run1 vs Run2.
    assert run1 != run2
