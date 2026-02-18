import os
import shutil

import h5py
import numpy as np
import pytest

from cogitao_wrapper.dataset import CogitaoDataset, HDF5CogitaoStore


@pytest.fixture
def temp_dataset_path(tmp_path):
    """Fixture to provide a temporary path for HDF5 store."""
    return str(tmp_path / "test_store.h5")


def test_store_creation_and_loading(temp_dataset_path):
    """Test creating a store and reloading it."""
    shape = (3, 32, 32)
    store = HDF5CogitaoStore(temp_dataset_path, shape=shape, batch_size=16)
    assert store.batch_size == 16
    assert os.path.exists(temp_dataset_path)

    # Reload
    del store
    store_reloaded = HDF5CogitaoStore(temp_dataset_path)
    assert store_reloaded.batch_size == 16

    # Check underlying HDF5 attributes
    with h5py.File(temp_dataset_path, "r") as f:
        assert "batch_size" in f.attrs
        assert f.attrs["batch_size"] == 16


def test_dataset_basic_access(temp_dataset_path):
    """Test accessing items from the dataset."""
    # Create store and add content
    shape = (3, 32, 32)
    store = HDF5CogitaoStore(temp_dataset_path, shape=shape, batch_size=2)

    # Add dummy data
    data = []
    for i in range(5):
        img = np.zeros(shape, dtype=np.float32) + i
        data.append(img)

    store.save_batch(data)

    # Use Dataset wrapper
    dataset = CogitaoDataset(temp_dataset_path)
    assert len(dataset) == 5

    # Test __getitem__ single
    item0 = dataset[0]
    assert "imgs" in item0
    # dataset returns tensors
    assert item0["imgs"].shape == shape
    # Convert to numpy for comparison or use torch.all
    assert (item0["imgs"] == 0).all()

    item4 = dataset[4]
    assert (item4["imgs"] == 4).all()


def test_dataset_batch_access(temp_dataset_path):
    """Test retrieving multiple items at once (smart slicing)."""
    shape = (3, 10, 10)
    store = HDF5CogitaoStore(temp_dataset_path, shape=shape, batch_size=2)
    data = [np.zeros(shape) + i for i in range(10)]
    store.save_batch(data)

    dataset = CogitaoDataset(temp_dataset_path)

    # Get a slice
    indices = [0, 1, 2]
    items = dataset.__getitems__(indices)
    assert len(items) == 3
    assert items[0]["imgs"].max() == 0
    assert items[1]["imgs"].max() == 1
    assert items[2]["imgs"].max() == 2


def test_dataset_iteration(temp_dataset_path):
    """Test iterating over the dataset."""
    shape = (3, 8, 8)
    store = HDF5CogitaoStore(temp_dataset_path, shape=shape, batch_size=2)
    # 4 items
    store.save_batch([np.zeros(shape)] * 4)

    dataset = CogitaoDataset(temp_dataset_path)
    count = 0
    for item in dataset:
        assert "imgs" in item
        count += 1
    assert count == 4
