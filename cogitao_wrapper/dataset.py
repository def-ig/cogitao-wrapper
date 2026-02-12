from pathlib import Path
from typing import Dict, Optional

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset


class HDF5CogitaoStore:
    """HDF5-based persistent storage for pre-generated training samples."""

    def __init__(self, path: str):
        """Initialize sample store.

        Args:
            cache_path: Path to HDF5 store file
        """
        self.cache_path = Path(path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize or validate store file
        self._init_cache()

        if not self.cache_path.exists():
            self._length = 0
        with h5py.File(self.cache_path, "r") as f:
            self._length = f.attrs.get("sample_count", 0)

    def _init_cache(self):
        """Initialize store file or validate existing one."""
        if not self.cache_path.exists():
            # Create new store file
            with h5py.File(self.cache_path, "w") as f:
                f.attrs["sample_count"] = 0
                # Create samples group
                f.create_group("samples")
                print(f"Created new sample store at {self.cache_path}")

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
        """Load a sample from store by index.

        Args:
            idx: Sample index

        Returns:
            Sample array [C, H, W] or None if not found
        """
        try:
            with h5py.File(self.cache_path, "r") as f:
                if f"samples/{idx}" in f:
                    return f[f"samples/{idx}"][()]  # type: ignore[return-value]
                else:
                    return None
        except Exception:
            return None

    def save_sample(self, sample: np.ndarray, idx: Optional[int] = None) -> int:
        """Save a sample to store.

        Args:
            sample: Sample array [C, H, W]
            idx: Optional index. If None, uses next available index

        Returns:
            Index where sample was saved
        """
        with h5py.File(self.cache_path, "a") as f:
            if idx is None:
                # Get next index
                idx = int(f.attrs.get("sample_count", 0))

            # Save sample
            sample_key = f"samples/{idx}"
            if sample_key in f:
                del f[sample_key]  # Overwrite if exists

            f.create_dataset(
                sample_key,
                data=sample,
                dtype="float32",
                compression="gzip",
                compression_opts=4,
            )

            # Update count if this is a new sample
            current_count = int(f.attrs.get("sample_count", 0))
            if idx >= current_count:
                f.attrs["sample_count"] = idx + 1
                self._length = idx + 1

        return idx

    def load_batch(self, indices: list[int]) -> list[Optional[np.ndarray]]:
        """Load multiple samples efficiently.

        Args:
            indices: List of sample indices

        Returns:
            List of samples (or None for missing samples)
        """
        samples = []
        with h5py.File(self.cache_path, "r") as f:
            for idx in indices:
                if f"samples/{idx}" in f:
                    samples.append(f[f"samples/{idx}"][()])  # type: ignore[arg-type]
                else:
                    samples.append(None)
        return samples

    def save_batch(
        self, samples: list[np.ndarray], start_idx: Optional[int] = None
    ) -> list[int]:
        """Save multiple samples efficiently.

        Args:
            samples: List of sample arrays
            start_idx: Optional starting index. If None, uses next available

        Returns:
            List of indices where samples were saved
        """
        indices = []
        with h5py.File(self.cache_path, "a") as f:
            if start_idx is None:
                start_idx = int(f.attrs.get("sample_count", 0))

            for i, sample in enumerate(samples):
                idx = start_idx + i
                sample_key = f"samples/{idx}"

                if sample_key in f:
                    del f[sample_key]

                f.create_dataset(
                    sample_key,
                    data=sample,
                    dtype="float32",
                    compression="gzip",
                    compression_opts=4,
                )
                indices.append(idx)

            # Update count
            current_count = int(f.attrs.get("sample_count", 0))
            new_count = max(current_count, start_idx + len(samples))
            f.attrs["sample_count"] = new_count
            self._length = new_count
        return indices

    def inspect(self):
        """Print information about the store."""
        if not self.cache_path.exists():
            print(f"Store file not found: {self.cache_path}")
            return

        with h5py.File(self.cache_path, "r") as f:
            print(f"Store file: {self.cache_path}")
            print(f"File size: {self.cache_path.stat().st_size / (1024**2):.2f} MB")
            print()

            # Print attributes
            print("Attributes:")
            for key in f.attrs.keys():
                print(f"  {key}: {f.attrs[key]}")
            print()

            # Print groups
            print("Groups:")
            for key in f.keys():
                group = f[key]
                if isinstance(group, h5py.Group):
                    num_items = len(group.keys())
                    print(f"  {key}: {num_items} items")
            print()

            # Print sample info
            if "samples" in f:
                sample_count = f.attrs.get("sample_count", 0)
                print(f"Total samples: {sample_count}")

                # Check first sample
                if sample_count > 0:
                    first_sample = f["samples/0"][()]
                    print(f"Sample shape: {first_sample.shape}")
                    print(f"Sample dtype: {first_sample.dtype}")
                    print(
                        f"Sample range: [{first_sample.min():.3f}, {first_sample.max():.3f}]"
                    )

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

        with h5py.File(self.cache_path, "r") as f:
            if "samples" not in f:
                print("No samples found in store.")
                return

            sample_count = f.attrs.get("sample_count", 0)
            num_to_show = min(num_examples, sample_count)
        if num_to_show == 0:
            print("No samples available to show.")
            return

        print(f"Showing {num_to_show} examples from store:")

        # Create subplots to show all images at once
        fig, axes = plt.subplots(1, num_to_show, figsize=(4 * num_to_show, 4))
        if num_to_show == 1:
            axes = [axes]

        for i in range(num_to_show):
            sample = f[f"samples/{i}"][()]
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
    """PyTorch Dataset that wraps HDF5CogitaoStore for consistent interface.

    Uses the same loading mechanism as HDF5CogitaoStore to ensure consistency.
    """

    def __init__(self, path: str):
        """Initialize dataset from store file.

        Args:
            path: Path to HDF5 store file
        """
        super(CogitaoDataset, self).__init__()
        self.path = Path(path)

        if not self.path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        # Initialize store - this validates the file and provides the loading interface
        self.store = HDF5CogitaoStore(path=str(path))

    def __len__(self) -> int:
        return len(self.store)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load a sample from dataset using store interface.

        Args:
            idx: Sample index

        Returns:
            Dictionary with 'imgs' key containing sample tensor [C, H, W]
        """
        sample = self.store[idx]  # Use indexing syntax

        if sample is None:
            raise KeyError(f"Sample {idx} not found in dataset")

        # Convert to tensor and return in dict format for compatibility
        tensor = torch.from_numpy(sample).float()
        return {"imgs": tensor}
