import os
from pathlib import Path
from typing import Dict, Optional

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset


class HDF5CogitaoStore:
    """HDF5-based persistent storage for pre-generated training samples in optimized format.

    Uses a contiguous 'imgs' dataset [N, C, H, W] for fast loading.
    Supports incremental batch writing to build up the dataset.
    """

    def __init__(
        self,
        path: str | Path,
        shape: Optional[tuple[int, int, int]] = None,
        batch_size: Optional[int] = None,
    ):
        """Initialize sample store.

        Args:
            path: Path to HDF5 store file
            shape: (C, H, W) shape for samples. Required only when creating new store.
            batch_size: Optional batch size to store as metadata
        """
        if isinstance(path, str):
            path = Path(path)

        self.path = path

        self._read_handle = None
        self._write_handle = None

        self._pid = None  # Track process ID to detect forks
        self._length = 0  # Track number of samples in store
        self._shape = shape  # (C, H, W)
        self.batch_size = batch_size

        # Initialize or open existing store file
        if not self.path.exists():
            if shape is None:
                raise ValueError(
                    "shape (C, H, W) is required when creating a new store file"
                )
            self._create_h5(shape, batch_size)

        # Get dataset info from file
        handle = self._get_read_handle()
        if "imgs" not in handle:
            raise ValueError(
                f"File {path} uses old format. Regenerating dataset is recommended. Use older library version if necessary."
            )
        self._length = handle["imgs"].shape[0]
        self._shape = tuple(handle["imgs"].shape[1:])  # (C, H, W)

        if "batch_size" in handle.attrs:
            self.batch_size = int(handle.attrs["batch_size"])

    def _create_h5(self, shape: tuple[int, int, int], batch_size: Optional[int] = None):
        """Initialize store file with resizable dataset for incremental writing.

        Args:
            shape: (C, H, W) shape for samples
            batch_size: Optional batch size to store as metadata
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        C, H, W = shape
        with h5py.File(self.path, "w", libver="latest") as f:
            # Create resizable dataset starting with 0 samples
            f.create_dataset(
                "imgs",
                shape=(0, C, H, W),
                maxshape=(None, C, H, W),  # Resizable along first dimension
                chunks=(batch_size, C, H, W),  # One sample per chunk
                dtype="float32",
                # compression=None,  # No compression for speed
                compression="lzf",
            )
            if batch_size is not None:
                f.attrs["batch_size"] = batch_size
        print(f"Created new optimized sample store at {self.path}")
        print(f"  Sample shape: {shape}")

    def _get_read_handle(self):
        """Get persistent read handle.

        Opens a new handle if:
        - No handle exists yet
        - We're in a different process (after fork)

        Returns:
            h5py.File handle
        """
        current_pid = os.getpid()

        # Forked process protection
        if self._pid is not None and self._pid != current_pid:
            self._close_handle()

        if self._read_handle is None:
            # Close write handle first - can't have both open simultaneously
            if self._write_handle is not None:
                try:
                    self._write_handle.close()
                except Exception:
                    pass
                self._write_handle = None

            self._read_handle = h5py.File(self.path, "r", libver="latest")
            self._pid = current_pid

        return self._read_handle

    def _get_write_handle(self):
        """Get persistent write handle.

        Opens a new handle if:
        - No handle exists yet
        - We're in a different process (after fork)

        Returns:
            h5py.File handle opened in append mode
        """
        current_pid = os.getpid()

        # Forked process protection
        if self._pid is not None and self._pid != current_pid:
            self._close_handle()

        if self._write_handle is None:
            # Close read handle first - can't have both open simultaneously
            if self._read_handle is not None:
                try:
                    self._read_handle.close()
                except Exception:
                    pass
                self._read_handle = None

            self._write_handle = h5py.File(self.path, "a", libver="latest")
            self._pid = current_pid

        return self._write_handle

    def _close_handle(self):
        """Close persistent file handles."""
        # Close read handle
        if self._read_handle is not None:
            try:
                self._read_handle.close()
            except Exception:
                pass
            self._read_handle = None

        # Close write handle
        if self._write_handle is not None:
            try:
                self._write_handle.close()
            except Exception:
                pass
            self._write_handle = None

        self._pid = None

    def __del__(self):
        """Cleanup: close file handle on deletion."""
        self._close_handle()

    def __getstate__(self):
        """Prepare object for pickling - exclude file handles."""
        state = self.__dict__.copy()
        # Remove unpicklable h5py file handles
        state["_read_handle"] = None
        state["_write_handle"] = None
        state["_pid"] = None
        return state

    def __setstate__(self, state):
        """Restore object from pickle - reinitialize handles on demand."""
        self.__dict__.update(state)
        # Handles will be reopened on first access via _get_read_handle()/_get_write_handle()

    def __len__(self) -> int:
        """Get number of samples currently in store."""
        return self._length

    def __getitem__(self, idx: int) -> Optional[np.ndarray]:
        """Load a sample from store by index (supports indexing syntax).

        Args:
            idx: Sample index

        Returns:
            Sample array [C, H, W] or None if not found
        """
        return self.load_batch([idx])[0]

    def __getitems__(self, idxs: list[int]) -> list[Optional[np.ndarray]]:
        """Load multiple samples from store by index using persistent handle.

        Args:
            idxs: List of sample indices

        Returns:
            List of sample arrays [C, H, W] or None if not found
            idxs: List of sample indices. Must be valid and within bounds.

        Returns:
            List of sample arrays [C, H, W]
        """
        return self.load_batch(idxs).tolist()

    def load_batch(self, indices: list[int]) -> np.ndarray:
        """Read a batch of samples using efficient fancy indexing.

        Indices MUST be valid and within bounds.

        Args:
           indices: List of indices to read

        Returns:
           Numpy array of shape [B, C, H, W]
        """
        f = self._get_read_handle()

        # h5py requires sorted indices for best performance and compatibility.
        # Sort indices to be safe and efficient.

        indices = np.array(indices)
        sorted_idx_args = np.argsort(indices)
        sorted_indices = indices[sorted_idx_args]

        if sorted_indices[0] < 0 or sorted_indices[-1] >= self._length:
            raise IndexError("Index range out of bounds")

        # h5py selection doesn't support duplicates in selection list.
        # Handle duplicate indices, as DataLoader with replacement=True can yield duplicates.

        unique_sorted, inverse_map = np.unique(indices, return_inverse=True)

        if len(unique_sorted) == 0:
            # Return empty array with correct shape (0, C, H, W)
            return np.zeros((0,) + self._shape, dtype="float32")

        batch_data = f["imgs"][unique_sorted]

        # Reconstruct original order and handle duplicates
        return batch_data[inverse_map]

    def save_batch(
        self, samples: list[np.ndarray], start_idx: Optional[int] = None
    ) -> list[int]:
        """Save multiple samples efficiently to the dataset.

        This method appends samples to the dataset by resizing it.
        For best performance, write larger batches.

        Args:
            samples: List of sample arrays [C, H, W]
            start_idx: Optional starting index. If None, appends to end.
                      If specified, must equal current length (no gaps allowed).

        Returns:
            List of indices where samples were saved
        """
        if not samples:
            return []

        f = self._get_write_handle()
        current_length = f["imgs"].shape[0]

        if start_idx is None:
            start_idx = current_length
        elif start_idx != current_length:
            raise ValueError(
                f"start_idx must equal current length {current_length} (no gaps allowed). "
                f"Got start_idx={start_idx}"
            )

        # Validate sample shapes
        batch = np.array(samples, dtype="float32")  # [N, C, H, W]
        if batch.shape[1:] != self._shape:
            raise ValueError(
                f"Sample shape {batch.shape[1:]} doesn't match expected {self._shape}"
            )

        # Resize dataset to fit new samples
        new_length = start_idx + len(samples)
        f["imgs"].resize(new_length, axis=0)

        # Write batch
        f["imgs"][start_idx:new_length] = batch

        # Update internal length
        self._length = new_length

        # Flush to ensure data is written
        f.flush()

        return list(range(start_idx, new_length))

    def inspect(self):
        """Print information about the store."""
        if not self.path.exists():
            print(f"Store file not found: {self.path}")
            return

        f = self._get_read_handle()
        print(f"Store file: {self.path}")
        print(f"File size: {self.path.stat().st_size / (1024**2):.2f} MB")
        print("Format: Optimized (contiguous 'imgs' dataset)")
        print()

        # Print dataset info
        if "imgs" in f:
            dset = f["imgs"]
            print("Dataset 'imgs':")
            print(f"  Shape: {dset.shape}")
            print(f"  Dtype: {dset.dtype}")
            print(f"  Chunks: {dset.chunks}")
            print(f"  Compression: {dset.compression}")

            if self.batch_size is not None:
                print(f"  Batch size: {self.batch_size}")
            print()

            print(f"Total samples: {self._length}")
            if self._length > 0:
                print(f"Sample shape: {self._shape}")
                # Show value range of first sample
                first_sample = dset[0]
                print(
                    f"Sample range (first): [{first_sample.min():.3f}, {first_sample.max():.3f}]"
                )
        else:
            print("No 'imgs' dataset found!")

    def clear(self, confirm: bool = False):
        """Clear all samples from the store.

        Args:
            confirm: If True, skip confirmation prompt
        """
        if not self.path.exists():
            print(f"Store file not found: {self.path}")
            return

        if not confirm:
            response = input(f"Are you sure you want to clear {self.path}? (yes/no): ")
            if response.lower() != "yes":
                print("Cancelled.")
                return

        # Close persistent handle before deleting
        self._close_handle()

        # Delete the file
        self.path.unlink(missing_ok=True)
        self._length = 0
        print(f"Store cleared: {self.path}")

    def show_examples(self, num_examples: int = 5, output_dir: str = "."):
        """Show example samples from the store.

        Args:
            num_examples: Number of examples to show
            output_dir: Directory to save the visualization
        """
        if not self.path.exists():
            print(f"Store file not found: {self.path}")
            return

        f = self._get_read_handle()
        if "imgs" not in f:
            print("No 'imgs' dataset found in store.")
            return

        num_to_show = min(num_examples, self._length)

        if num_to_show == 0:
            print("No samples available to show.")
            return

        print(f"Showing {num_to_show} examples from store:")

        # Create subplots to show all images at once
        fig, axes = plt.subplots(1, num_to_show, figsize=(4 * num_to_show, 4))
        if num_to_show == 1:
            axes = [axes]

        for i in range(num_to_show):
            sample = f["imgs"][i]
            print(f"Sample {i}: shape={sample.shape}, dtype={sample.dtype}")
            # Transpose from (C, H, W) to (H, W, C) for matplotlib
            if len(sample.shape) == 3 and sample.shape[0] in [1, 3, 4]:
                sample = sample.transpose(1, 2, 0)
            axes[i].imshow(sample)
            axes[i].set_title(f"Sample {i}")
            axes[i].axis("off")

        plt.tight_layout()
        save_path = Path(output_dir) / f"dataset_examples_{num_to_show}.png"
        plt.savefig(save_path)
        print(f"Saved visualization to {save_path}")


class CogitaoDataset(Dataset):
    """PyTorch Dataset for optimized HDF5 format.

    Expects HDF5 structure:
    - imgs: [N, C, H, W] dataset with chunks=(1, C, H, W)
    - No compression

    This provides 3-5x faster loading than the old format.
    Old format (samples/0, samples/1, ...) is NOT supported.
    """

    def __init__(self, path: str | Path):
        """Initialize dataset from optimized store file.

        Args:
            path: Path to HDF5 store file (optimized format)
        """
        super().__init__()

        if isinstance(path, str):
            path = Path(path)

        self.path = path

        if not self.path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        # Get dataset info and validate format
        self.h5df_store = HDF5CogitaoStore(path)
        self._length = self.h5df_store._length
        self._shape = self.h5df_store._shape

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load a sample from dataset.

        Args:
            idx: Sample index

        Returns:
            Dictionary with 'imgs' key containing sample tensor [C, H, W]
        """

        sample = self.h5df_store[idx]

        # Convert to tensor
        tensor = torch.from_numpy(sample).float()
        return {"imgs": tensor}

    def __getitems__(self, idxs: list[int]) -> list[Dict[str, torch.Tensor]]:
        """Load multiple samples from dataset efficiently.

        This method is called by DataLoader when batch_sampler is used or
        when fetching a batch of indices.

        Args:
            idxs: List of sample indices

        Returns:
            List of dictionaries with 'imgs' key
        """

        if len(idxs) <= 0:
            return []

        try:
            # load_batch returns [N, C, H, W]
            batch_arr = self.h5df_store.load_batch(idxs)

            # Convert to list of dicts
            results = []
            for sample in batch_arr:
                results.append({"imgs": torch.from_numpy(sample).float()})
            return results

        except Exception as e:
            print(f"Error in batch reading: {e}")
            # Fallback to individual reading using new load_batch
            return []

    def __getstate__(self):
        """Prepare for pickling (DataLoader multiprocessing)."""
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        """Restore after unpickling."""
        self.__dict__.update(state)
