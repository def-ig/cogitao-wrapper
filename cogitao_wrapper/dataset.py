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

    def __init__(self, path: str, shape: Optional[tuple[int, int, int]] = None):
        """Initialize sample store.

        Args:
            path: Path to HDF5 store file
            shape: (C, H, W) shape for samples. Required only when creating new store.
        """
        self.cache_path = Path(path)

        self._read_handle = None
        self._write_handle = None

        self._pid = None  # Track process ID to detect forks
        self._length = 0  # Track number of samples in store
        self._shape = shape  # (C, H, W)

        # Initialize or open existing store file
        if not self.cache_path.exists():
            if shape is None:
                raise ValueError(
                    "shape (C, H, W) is required when creating a new store file"
                )
            self._create_h5(shape)

        # Get dataset info from file
        handle = self._get_read_handle()
        if "imgs" not in handle:
            raise ValueError(
                f"File {path} uses old format. Regenerating dataset is recommended. Use older library version if necessary."
            )
        self._length = handle["imgs"].shape[0]
        self._shape = tuple(handle["imgs"].shape[1:])  # (C, H, W)

    def _create_h5(self, shape: tuple[int, int, int]):
        """Initialize store file with resizable dataset for incremental writing.

        Args:
            shape: (C, H, W) shape for samples
        """
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        C, H, W = shape
        with h5py.File(self.cache_path, "w", libver="latest") as f:
            # Create resizable dataset starting with 0 samples
            f.create_dataset(
                "imgs",
                shape=(0, C, H, W),
                maxshape=(None, C, H, W),  # Resizable along first dimension
                chunks=(1, C, H, W),  # One sample per chunk
                dtype="float32",
                compression=None,  # No compression for speed
            )
        print(f"Created new optimized sample store at {self.cache_path}")
        print(f"  Sample shape: {shape}")

    def _get_read_handle(self):
        """Get persistent read handle with SWMR mode.

        Opens a new handle if:
        - No handle exists yet
        - We're in a different process (after fork)

        Returns:
            h5py.File handle opened in SWMR read mode
        """
        current_pid = os.getpid()

        # Forked process protection...
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

            # Open with SWMR mode for concurrent reads
            self._read_handle = h5py.File(
                self.cache_path, "r", libver="latest", swmr=True
            )
            self._pid = current_pid

        return self._read_handle

    def _get_write_handle(self):
        """Get persistent write handle.

        Opens a new handle if:
        - No handle exists yet
        - We're in a different process (after fork)

        Returns:
            h5py.File handle opened in append mode (without SWMR)
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

            # Open in append mode without SWMR (allows attribute updates)
            self._write_handle = h5py.File(self.cache_path, "a", libver="latest")
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
        return self.load_sample(idx)

    def load_sample(self, idx: int) -> Optional[np.ndarray]:
        """Load a sample from store by index using persistent handle.

        Args:
            idx: Sample index

        Returns:
            Sample array [C, H, W] or None if not found
        """
        try:
            f = self._get_read_handle()
            if idx < 0 or idx >= self._length:
                return None
            # Direct indexing into contiguous dataset - fast!
            return f["imgs"][idx]  # type: ignore[return-value]
        except Exception:
            return None

    def save_sample(self, sample: np.ndarray, idx: Optional[int] = None) -> int:
        """Save a single sample to store.

        For efficiency, prefer save_batch() for multiple samples.

        Args:
            sample: Sample array [C, H, W]
            idx: Optional index. If None, appends to end

        Returns:
            Index where sample was saved
        """
        # Use save_batch for consistency
        indices = self.save_batch([sample], start_idx=idx)
        return indices[0]

    def load_batch(self, indices: list[int]) -> list[Optional[np.ndarray]]:
        """Load multiple samples efficiently using persistent handle.

        Args:
            indices: List of sample indices

        Returns:
            List of samples (or None for missing samples)
        """
        samples = []
        f = self._get_read_handle()
        for idx in indices:
            if 0 <= idx < self._length:
                samples.append(f["imgs"][idx])
            else:
                samples.append(None)
        return samples

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
        if not self.cache_path.exists():
            print(f"Store file not found: {self.cache_path}")
            return

        f = self._get_read_handle()
        print(f"Store file: {self.cache_path}")
        print(f"File size: {self.cache_path.stat().st_size / (1024**2):.2f} MB")
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
        if not self.cache_path.exists():
            print(f"Store file not found: {self.cache_path}")
            return

        if not confirm:
            response = input(
                f"Are you sure you want to clear {self.cache_path}? (yes/no): "
            )
            if response.lower() != "yes":
                print("Cancelled.")
                return

        # Close persistent handle before deleting
        self._close_handle()

        # Delete the file
        self.cache_path.unlink(missing_ok=True)
        self._length = 0
        print(f"Store cleared: {self.cache_path}")

    def show_examples(self, num_examples: int = 5, output_dir: str = "."):
        """Show example samples from the store.

        Args:
            num_examples: Number of examples to show
            output_dir: Directory to save the visualization
        """
        if not self.cache_path.exists():
            print(f"Store file not found: {self.cache_path}")
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

    def __init__(self, path: str):
        """Initialize dataset from optimized store file.

        Args:
            path: Path to HDF5 store file (optimized format)
        """
        super().__init__()
        self.path = Path(path)

        if not self.path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        # File handle management (process-safe)
        self._file_handle = None
        self._pid = None

        # Get dataset info and validate format
        with h5py.File(self.path, "r", swmr=True) as f:
            if "imgs" not in f:
                raise ValueError(
                    f"File {path} doesn't have 'imgs' dataset. "
                    "This appears to be the old format (samples/N structure). "
                    "Please convert your dataset to the optimized format."
                )
            self._length = f["imgs"].shape[0]
            self._shape = f["imgs"].shape[1:]  # (C, H, W)

    def _get_file_handle(self):
        """Get persistent file handle (process-safe for DataLoader workers)."""
        current_pid = os.getpid()

        # Reopen handle if we forked to a new process
        if self._pid != current_pid:
            if self._file_handle is not None:
                try:
                    self._file_handle.close()
                except Exception:
                    pass
            self._file_handle = h5py.File(self.path, "r", swmr=True)
            self._pid = current_pid

        return self._file_handle

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load a sample from dataset.

        Args:
            idx: Sample index

        Returns:
            Dictionary with 'imgs' key containing sample tensor [C, H, W]
        """
        if idx < 0 or idx >= self._length:
            raise IndexError(f"Index {idx} out of range [0, {self._length})")

        # Get file handle (reopens if needed after fork)
        f = self._get_file_handle()

        # Direct array indexing - fast!
        sample = f["imgs"][idx]

        # Convert to tensor
        tensor = torch.from_numpy(sample).float()
        return {"imgs": tensor}

    def __del__(self):
        """Cleanup file handle."""
        if self._file_handle is not None:
            try:
                self._file_handle.close()
            except Exception:
                pass

    def __getstate__(self):
        """Prepare for pickling (DataLoader multiprocessing)."""
        state = self.__dict__.copy()
        state["_file_handle"] = None
        state["_pid"] = None
        return state

    def __setstate__(self, state):
        """Restore after unpickling."""
        self.__dict__.update(state)
